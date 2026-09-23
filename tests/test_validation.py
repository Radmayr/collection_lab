import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.special import expit, logit

from collection_lab.data import random_split, time_split
from collection_lab.metrics import (
    calibration_offset,
    full_calibration,
    gain_chart,
    gain_chart_metrics,
    gain_chart_table,
    information_value,
    iv_table,
    prob_to_logit,
    quantile_buckets,
)
from collection_lab.modeling import train_model
from collection_lab.validation import (
    feature_metric_dynamics,
    learning_curve,
    metric_dynamics,
    model_report,
    plot_stab,
    stability_table,
)

FAST = {"n_estimators": 150, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(0)
    lg = np.round(rng.normal(-2, 0.8, 5000), 2)  # одинаковые логиты
    y = rng.binomial(1, expit(1.2 * lg + 0.3))
    return lg, y


def test_quantile_buckets_keep_ties():
    x = np.array([0] * 86 + [1] * 8 + [2] * 3 + [3] * 2 + [4, 25])
    b = quantile_buckets(x, 5)
    groups = pd.Series(x).groupby(b).agg(["min", "max"]).to_numpy().tolist()
    expected = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 25]]
    assert groups == expected
    assert (quantile_buckets(np.array([1.0, np.nan]), 2) == [0, -1]).all()


def test_calibration_offset_and_full(scored):
    lg, y = scored
    b = calibration_offset(lg, y)
    assert expit(lg + b).mean() == pytest.approx(y.mean(), abs=1e-9)
    k, c = full_calibration(lg, y)
    assert k == pytest.approx(1.2, abs=0.15) and c == pytest.approx(0.3, abs=0.3)


def test_gain_chart_table_properties(scored):
    lg, y = scored
    t, m = gain_chart_table(lg, y, n_buckets=10)
    assert t["n"].sum() == len(lg)
    assert t["bucket"].min() == 1
    # бакет 1 — самый рисковый
    assert t.set_index("bucket")["pred"].idxmax() == 1
    # калибровка на офсет — постоянный сдвиг в логитах (как в risk_instruments)
    assert np.ptp(logit(t["calibrated"]) - logit(t["pred"])) < 0.01
    # ДИ — нормальный 99%
    ci = 2.5758 * np.sqrt(t["badrate"] * (1 - t["badrate"]) / t["n"])
    assert np.allclose(t["ci"], ci, atol=1e-4)
    assert set(m) == {"roc_auc", "hl", "n_buck", "offset", "full calib coefs"}
    assert m["n_buck"] == int(t.loc[t["bucket"].idxmax(), "n"])


def test_gain_chart_return_modes(scored):
    lg, y = scored
    fig, m = gain_chart(lg, y, n_buckets=10, calib=True, calib_full=True,
                        return_plotly_fig=True)
    assert isinstance(fig, go.Figure) and "Group" in m
    names = {tr.name for tr in fig.data}
    assert {"Model Prediction", "Badrate", "Calibrated Model", "Full Calibrated Model",
            "Confidence Interval"} <= names
    only_fig = gain_chart(lg, y, return_plotly_fig=True, return_dict=False)
    assert isinstance(only_fig, go.Figure)
    groups = np.where(np.arange(len(lg)) % 2, "a", "b")
    fig_g, m_g = gain_chart(lg, y, groups=groups, n_buckets=[5, 8], return_plotly_fig=True)
    assert set(m_g) == {"a", "b"}
    table = gain_chart_metrics([m, m_g], ["train", "test"])
    assert list(table.index) == ["train", "test / b", "test / a"]


def test_information_value(binary_df):
    iv_x1 = information_value(binary_df["x1"], binary_df["target"])
    iv_noise = information_value(binary_df["noise"], binary_df["target"])
    assert iv_x1 > 0.1 > iv_noise
    assert information_value(binary_df["x1"], binary_df["target"], method="IV_auc") > 0.1
    t = iv_table(binary_df, "target", ["x1", "noise"])
    assert t["feature"].iloc[0] == "x1"


def test_plot_stab(binary_df, capsys):
    df = binary_df.assign(cnt=np.where(np.arange(len(binary_df)) % 10 == 0, 1, 0))
    figs = plot_stab(df["x2"], df["target"], df["report_date"], 5, feature_nm="x2",
                     period="Q", add_psi=True, return_plotly_fig=True,
                     line_dt={"2024-01-01": "end of train"})
    assert len(figs) == 2 and all(isinstance(f, go.Figure) for f in figs)
    assert "nulls have been deleted" in capsys.readouterr().out
    legend = [tr.name for tr in figs[0].data if tr.showlegend]
    assert all(name.startswith("x2 in [") for name in legend)
    table, periods = stability_table(df["cnt"], df["target"], df["report_date"], 5, period="Q",
                                     verbose=False)
    assert set(table["label"]) == {"var in [0, 0]", "var in [1, 1]"}
    assert {"iv", "psi"} <= set(periods.columns)
    tbl_null, _ = stability_table(df["x2"], df["target"], df["report_date"], 3, period="Q",
                                  null_bucket=True, verbose=False)
    assert "var is null" in set(tbl_null["label"])
    cont = plot_stab(df["x1"], df["x1"] * 2 + 1, df["report_date"], 4, period="Q",
                     binary_target=False, return_plotly_fig=True)
    assert len(cont) == 2


def test_metric_dynamics(binary_df):
    df = binary_df.assign(score=binary_df["x1"])
    res = metric_dynamics(df, "target", "score", "report_date", "segment", freq="Q")
    assert set(res.table["segment"]) == {"s1", "s2"}
    assert res.table["auc"].between(0.5, 1).all()
    assert isinstance(res.plot(), go.Figure)


def test_feature_metric_dynamics(binary_df):
    split = random_split(binary_df, "target")
    res = feature_metric_dynamics(split.train, binary_df, ["x1", "cat"], "target",
                                  "report_date", freq="Q")
    assert set(res.table["feature"]) == {"x1", "cat"}
    assert isinstance(res.plot(), go.Figure)


def test_model_report(binary_df, tmp_path):
    split = time_split(binary_df, "target", "report_date", oot_from="2024-03-01")
    model = train_model(split, ["x1", "x2", "cat", "noise"], params=FAST)
    rep = model_report(model, split, date_col="report_date", segment="segment", n_buckets=10)
    assert list(rep.metrics.index) == ["train", "val", "test"]
    assert rep.metrics.loc["train", "score_psi"] == 0
    assert list(rep.calibration.index) == ["train", "val", "test"]
    assert {"gain_charts", "feature_importance", "auc_dynamics"} <= set(rep.figures)
    assert rep.segments is not None
    rep.save(tmp_path)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Model report" in html and "plotly" in html
    assert isinstance(learning_curve(model), go.Figure)


def test_prob_to_logit_roundtrip():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(expit(prob_to_logit(p)), p)
