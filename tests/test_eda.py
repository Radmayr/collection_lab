import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from collection_lab.eda import (
    add_days_since,
    eda_transactions,
    maturation_transactions,
    overview,
    plot_distribution,
    plot_target_rate_by_bins,
    plot_vintage,
    target_dynamics,
    target_summary,
    vintage,
    vintage_by_segment,
    vintage_by_type,
    wilson_ci,
)
from collection_lab.eda.vintage import Y_LABELS


@pytest.fixture(scope="module")
def tx_df():
    """Синтетические транзакции: 200 договоров, у части нет транзакций."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        send = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 90)))
        balance = float(rng.gamma(2, 50_000))
        n_tx = int(rng.poisson(3)) if i % 4 else 0
        if n_tx == 0:
            rows.append({"contract_number": i, "rtk_send_date": send, "rtk_balance": balance,
                         "real_transaction_dttm": None, "transaction_amt": np.nan,
                         "tr_direction": None})
        for _ in range(n_tx):
            rows.append({"contract_number": i, "rtk_send_date": send, "rtk_balance": balance,
                         "real_transaction_dttm": send + pd.Timedelta(
                             days=int(rng.integers(-5, 400))),
                         "transaction_amt": float(rng.gamma(1, 5_000)),
                         "tr_direction": rng.choice(["parent", "daughter"])})
    return add_days_since(pd.DataFrame(rows), "real_transaction_dttm", "rtk_send_date")


def test_overview(binary_df):
    ov = overview(binary_df)
    assert ov.loc["cat", "kind"] == "categorical"
    assert ov.loc["const", "kind"] == "constant"
    assert ov.loc["all_nan", "kind"] == "empty"
    assert ov.loc["report_date", "kind"] == "datetime"
    assert ov.loc["x2", "missing_share"] == pytest.approx(binary_df["x2"].isna().mean())
    ts = target_summary(binary_df, "target", by="segment")
    assert ts["n"].sum() == len(binary_df)


def test_plot_distribution(binary_df):
    fig = plot_distribution(binary_df, "x1")
    assert isinstance(fig, go.Figure) and "μ=" in fig.layout.title.text
    skewed = pd.Series(np.random.default_rng(0).lognormal(0, 2, 1000), name="money")
    assert "log10" in plot_distribution(skewed).layout.xaxis.title.text
    assert isinstance(plot_target_rate_by_bins(binary_df, "x1", "target"), go.Figure)


def test_wilson_ci():
    lo, hi = wilson_ci([10, 0], [100, 50])
    assert lo[0] < 0.1 < hi[0]
    assert lo[1] == pytest.approx(0, abs=1e-12) and 0 < hi[1] < 0.1


def test_target_dynamics(binary_df):
    res = target_dynamics(binary_df, "report_date", "target", "segment", freq="Q", verbose=False)
    t = res.table
    assert set(t["segment"]) == {"s1", "s2"}
    assert t["n_total"].sum() == len(binary_df)
    assert ((t["ci_low"] <= t["target_rate"]) & (t["target_rate"] <= t["ci_high"])).all()
    assert isinstance(res.plot(), go.Figure)
    single = target_dynamics(binary_df, "report_date", "target", verbose=False)
    assert set(single.table["segment"]) == {"all"}


def test_maturation_and_vintage(tx_df):
    res = maturation_transactions(tx_df, "train", horizon_days=365, step=7,
                                  processed_dt="2026-01-01")
    assert res["day"].iloc[-1] <= 365 and (res["day"] % 7 == 0).all()
    assert res["N_total"].iloc[0] == 200
    assert res["cum_share_clients"].is_monotonic_increasing
    assert res["cum_share_clients"].iloc[-1] <= 0.75 + 1e-9  # у 1/4 договоров нет транзакций
    assert isinstance(plot_vintage(res), go.Figure)
    short = maturation_transactions(tx_df, "x", horizon_days=365,
                                    processed_dt=tx_df["rtk_send_date"].max() + pd.Timedelta(
                                        days=30))
    assert short["day"].max() == 30  # горизонт урезан до «созревшего»

    data, fig = vintage_by_type(tx_df, "tr_direction", horizon_days=200, step=10,
                                processed_dt="2026-01-01")
    # договоры без транзакций — не отдельный тип, они входят только в базу
    assert set(data["sample"]) == {"parent", "daughter"}
    assert isinstance(fig, go.Figure)


def test_vintage_by_type_shares_base_and_sums_to_total(tx_df):
    """У всех типов одна база, и по типам доля/сумма/число транзакций складываются в общий."""
    kw = dict(horizon_days=365, step=7, processed_dt="2026-01-01")
    total = maturation_transactions(tx_df, "ALL", **kw).set_index("day")
    data, _ = vintage_by_type(tx_df, "tr_direction", **kw)
    explicit_none, _ = vintage_by_type(tx_df, "tr_direction", population=None, **kw)
    pd.testing.assert_frame_equal(data, explicit_none)            # явный None = та же общая база
    assert (data["N_total"] == total["N_total"].iloc[0]).all()
    assert np.allclose(data["total_balance"], total["total_balance"].iloc[0])
    summed = data.groupby("day")[["cum_share_of_balance", "cum_tx_sum", "cum_tx_cnt"]].sum()
    assert summed.index.equals(total.index)                       # общий горизонт и сетка дней
    for col in summed.columns:
        assert np.allclose(summed[col], total[col]), col


def test_vintage_by_type_common_horizon(tx_df):
    """Тип, у которого последняя ретро-дата раньше, не получает более длинный горизонт."""
    df = tx_df.copy()
    late = df["rtk_send_date"] == df["rtk_send_date"].max()
    df.loc[late & df["transaction_amt"].notna(), "tr_direction"] = "late_only"
    processed = df["rtk_send_date"].max() + pd.Timedelta(days=60)
    data, _ = vintage_by_type(df, "tr_direction", horizon_days=365, processed_dt=processed)
    assert data.groupby("sample")["day"].max().eq(60).all()


@pytest.fixture
def seg_df(tx_df):
    """tx_df с продуктом на уровне договора; у части договоров продукт не указан."""
    df = tx_df.copy()
    df["product"] = np.where(df["contract_number"] % 3 == 0, "A",
                             np.where(df["contract_number"] % 3 == 1, "B", "C"))
    df.loc[df["contract_number"] % 17 == 0, "product"] = None
    return df


def test_vintage_by_segment_has_own_base_per_segment(seg_df):
    """Каждый сегмент — своя база: результат равен vintage по строкам этого сегмента."""
    kw = dict(horizon_days=365, step=7, processed_dt="2026-01-01")
    data, fig = vintage_by_segment(seg_df, "product", **kw)
    assert isinstance(fig, go.Figure)
    assert set(data["sample"]) == {"A", "B", "C", "NA"}           # пустой продукт — сегмент NA
    filled = seg_df.assign(product=seg_df["product"].fillna("NA"))
    for s, part in data.groupby("sample"):
        expected, _ = vintage(filled[filled["product"] == s], s, **kw)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), expected)
        assert part["N_total"].iloc[0] == filled.loc[filled["product"] == s,
                                                     "contract_number"].nunique()


def test_vintage_by_segment_selection(seg_df):
    kw = dict(horizon_days=100, processed_dt="2026-01-01")
    counts = seg_df.assign(p=seg_df["product"].fillna("NA")).drop_duplicates(
        "contract_number")["p"].value_counts()
    sizes = counts.sort_index().sort_values(ascending=False, kind="stable")  # равенство — алфавит
    top2, _ = vintage_by_segment(seg_df, "product", top_k=2, **kw)
    assert list(dict.fromkeys(top2["sample"])) == sizes.index[:2].tolist()   # крупнейшие первыми
    big, _ = vintage_by_segment(seg_df, "product", min_contracts=int(sizes.min()) + 1, **kw)
    assert sizes.idxmin() not in set(big["sample"])
    chosen, _ = vintage_by_segment(seg_df, "product", segments=["C", "A"], **kw)
    assert list(dict.fromkeys(chosen["sample"])) == ["C", "A"]


def test_vintage_by_segment_population_and_errors(seg_df):
    kw = dict(horizon_days=100, processed_dt="2026-01-01")
    base = seg_df.drop_duplicates("contract_number")[["contract_number", "rtk_balance",
                                                       "product"]]
    extra = pd.DataFrame({"contract_number": [10_000], "rtk_balance": [1e6], "product": ["A"]})
    data, _ = vintage_by_segment(seg_df, "product", population=pd.concat([base, extra]), **kw)
    n = data.groupby("sample")["N_total"].first()
    no_pop, _ = vintage_by_segment(seg_df, "product", **kw)
    n0 = no_pop.groupby("sample")["N_total"].first()
    assert n["A"] == n0["A"] + 1 and n["B"] == n0["B"]           # база делится по сегменту
    with pytest.raises(KeyError, match="population"):
        vintage_by_segment(seg_df, "product", population=base.drop(columns="product"), **kw)
    with pytest.raises(KeyError, match="нет колонки"):
        vintage_by_segment(seg_df, "no_such_col", **kw)


def test_vintage_by_segment_labels(seg_df):
    _, fig = vintage_by_segment(seg_df, "product", y="cum_share_clients", horizon_days=100,
                                processed_dt="2026-01-01")
    assert fig.layout.title.text == "Винтаж по «product»: конверсия в транзакцию"
    assert fig.layout.legend.title.text == "product"


def test_vintage_by_type_transaction_without_type_goes_to_na(tx_df):
    df = tx_df.copy()
    first_tx = df.index[df["transaction_amt"].notna()][0]
    df.loc[first_tx, "tr_direction"] = None
    data, _ = vintage_by_type(df, "tr_direction", horizon_days=365, processed_dt="2026-01-01")
    assert "NA" in set(data["sample"])                             # транзакция без типа не теряется


def test_vintage_combines_maturation_and_plot(tx_df):
    """vintage() — то же, что maturation_transactions() + plot_vintage(), но одним вызовом."""
    data, fig = vintage(tx_df, "train", horizon_days=365, step=7, processed_dt="2026-01-01")
    expected = maturation_transactions(tx_df, "train", horizon_days=365, step=7,
                                       processed_dt="2026-01-01")
    pd.testing.assert_frame_equal(data, expected)
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Дней от rtk_send_date"       # retro_dt_col по умолчанию


def test_vintage_russian_labels_on_title_and_yaxis():
    """Заголовок и подпись Y — человекочитаемые, не сырое имя колонки."""
    res = pd.DataFrame({"day": [0, 1], "sample": ["a", "a"], "N_total": [10, 10],
                        "cum_tx_cnt": [1, 2], "cum_tx_sum": [100, 200],
                        "cum_clients_with_tx": [1, 2], "cum_share_of_balance": [0.1, 0.2],
                        "cum_share_clients": [0.05, 0.1], "other_metric": [1, 2]})
    for y, label in Y_LABELS.items():
        fig = plot_vintage(res, y)
        assert label in fig.layout.title.text
        assert fig.layout.yaxis.title.text == label
    # колонка без готовой подписи — используется её собственное имя
    assert plot_vintage(res, "other_metric").layout.yaxis.title.text == "other_metric"


def test_eda_transactions(tx_df):
    rep = eda_transactions(tx_df, tx_dt_col="real_transaction_dttm", verbose=False)
    assert rep["overview"]["clients"] == 200
    assert {"balance", "tx_amount", "activity_by_day", "by_month"} <= set(rep["figures"])
    assert len(rep["top_clients"]) == 10


@pytest.mark.slow
def test_vintage_on_real_transactions(transactions_df):
    df = add_days_since(transactions_df, "real_transaction_dttm", "rtk_send_date")
    phx = df[df["financial_account_subtype_cd"] == "PHX"]
    res = maturation_transactions(phx, "PHX", horizon_days=730, processed_dt="2026-09-01")
    assert res["cum_share_clients"].between(0, 1).all()
    data, fig = vintage(phx, "PHX", horizon_days=730, processed_dt="2026-09-01")
    pd.testing.assert_frame_equal(data, res)
    assert isinstance(fig, go.Figure)
    data, _ = vintage_by_type(phx, "tr_direction", horizon_days=740, step=7,
                              processed_dt="2026-09-01")
    assert data["sample"].nunique() >= 1
    rep = eda_transactions(phx, tx_dt_col="real_transaction_dttm", verbose=False)
    assert rep["overview"]["clients"] == phx["contract_number"].nunique()
