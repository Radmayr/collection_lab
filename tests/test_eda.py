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
    vintage_by_type,
    wilson_ci,
)


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
    # договоры без транзакций попадают в тип "NA" (как в исходной функции)
    assert set(data["sample"]) == {"parent", "daughter", "NA"}
    assert isinstance(fig, go.Figure)


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
    data, _ = vintage_by_type(phx, "tr_direction", horizon_days=740, step=7,
                              processed_dt="2026-09-01")
    assert data["sample"].nunique() >= 1
    rep = eda_transactions(phx, tx_dt_col="real_transaction_dttm", verbose=False)
    assert rep["overview"]["clients"] == phx["contract_number"].nunique()
