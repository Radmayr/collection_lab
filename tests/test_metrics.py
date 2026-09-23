import numpy as np
import pandas as pd
import pytest

from collection_lab.metrics import (
    feature_psi,
    get_metric,
    gini,
    ks,
    metrics_by_segment,
    psi,
    psi_by_period,
    psi_label,
    psi_table,
    quantile_edges,
    roc_auc,
)


def test_auc_gini_ks_basic():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert roc_auc(y, p) == pytest.approx(0.75)
    assert gini(y, p) == pytest.approx(0.5)
    assert ks(y, p) == pytest.approx(0.5)
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))


def test_metric_direction():
    auc, ll = get_metric("auc"), get_metric("logloss")
    assert auc.delta(0.7, 0.6) > 0
    assert ll.delta(0.3, 0.4) > 0
    with pytest.raises(ValueError):
        get_metric("unknown")
    custom = get_metric(lambda y, p: 1.0)
    assert custom([0], [0]) == 1.0


def test_metrics_by_segment():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = y * 0.3 + rng.random(400)
    seg = np.where(np.arange(400) < 300, "big", "small")
    out = metrics_by_segment(y, p, seg, min_size=150)
    assert list(out.index) == ["ALL", "big"]
    assert out.loc["ALL", "n"] == 400
    assert {"auc", "gini", "target_rate"} <= set(out.columns)


def _psi_manual(e_share, a_share):
    e, a = np.asarray(e_share), np.asarray(a_share)
    return float(((a - e) * np.log(a / e)).sum())


def test_psi_identical_is_zero():
    x = np.random.default_rng(1).normal(size=1000)
    assert psi(x, x) == pytest.approx(0.0)


def test_psi_categorical_matches_formula():
    e = pd.Series(["a"] * 50 + ["b"] * 50)
    a = pd.Series(["a"] * 70 + ["b"] * 30)
    assert psi(e, a) == pytest.approx(_psi_manual([0.5, 0.5], [0.7, 0.3]))


def test_psi_missing_category_gets_count_one():
    e = pd.Series(["a"] * 99 + ["b"])
    a = pd.Series(["a"] * 100)
    table = psi_table(e, a)
    assert table.set_index("bin").loc["b", "actual_count"] == 1


def test_psi_nan_is_own_bin_and_edges_sorted():
    rng = np.random.default_rng(2)
    e = pd.Series(rng.normal(size=1000))
    a = e.copy()
    a[:200] = np.nan
    table = psi_table(e, a, bins=5)
    assert "NaN" in set(table["bin"])
    edges = quantile_edges(e, bins=5)
    assert np.all(np.diff(edges) > 0)
    labels = [b for b in table["bin"] if b != "NaN"]
    assert labels == sorted(labels)  # подписи идут по порядку границ
    assert psi(e, a, bins=5) > 0.1


def test_psi_label():
    assert psi_label(0.05) == "stable"
    assert psi_label(0.2) == "moderate"
    assert psi_label(0.3) == "significant"


def test_feature_psi_and_by_period(binary_df):
    train = binary_df.iloc[:2000]
    test = binary_df.iloc[2000:].copy()
    test["x1"] = test["x1"] + 1.0  # сдвиг
    out = feature_psi(train, test, ["x1", "noise", "cat"], cat_features=["cat"])
    assert out.iloc[0]["feature"] == "x1"
    assert out.set_index("feature").loc["noise", "status"] == "stable"

    by_p = psi_by_period(binary_df, "x1", "report_date", freq="Q")
    assert by_p["psi"].iloc[0] == pytest.approx(0.0)
    adj = psi_by_period(binary_df, "x1", "report_date", freq="Q", mode="adjacent")
    assert np.isnan(adj["psi"].iloc[0])
