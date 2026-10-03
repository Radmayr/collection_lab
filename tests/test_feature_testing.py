import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from collection_lab import feature_testing as ft
from collection_lab.data import time_split
from collection_lab.metrics import get_metric, lift
from collection_lab.modeling import train_model
from collection_lab.tracking import Experiment

FAST = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 8}
QUICK = {"params": FAST, "n_boot": 20, "verbose": False}


@pytest.fixture(scope="module")
def split(binary_df):
    return time_split(binary_df, "target", "report_date", oot_from="2024-03-01")


def test_new_metrics():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(0.9, 0.1, 10)
    assert get_metric("pr_auc")(y, p) == pytest.approx(1.0)
    assert get_metric("brier").greater_is_better is False
    assert lift(y, p, share=0.2) == pytest.approx(5.0)
    m = get_metric("lift@20%")
    assert m.name == "lift@20%" and m(y, p) == pytest.approx(5.0)


def test_features_retrain(split, tmp_path):
    t = ft.test_features(split, ["x2", "cat"], ["x1", "noise"], segment="segment",
                         date_col="report_date", **QUICK)
    s = t.summary.set_index("name")
    assert set(s.index) == {"x1", "noise", "ALL"}
    assert t.info["how"] == "retrain" and t.info["last"] == "test"
    assert s.loc["x1", "verdict"] == "better" and s.loc["x1", "delta_test"] > 0
    assert s.loc["noise", "verdict"] != "better"
    assert s.loc["x1", "cv_ci_low"] < s.loc["x1", "delta_cv"] < s.loc["x1", "cv_ci_high"]
    assert s.loc["x1", "test_ci_low"] <= s.loc["x1", "delta_test"] <= s.loc["x1", "test_ci_high"]
    assert s.loc["ALL", "n_features"] == 2 and s.loc["x1", "coverage"] == 1
    assert {"psi_max", "filled_from", "worst_segment", "uni_auc"} <= set(s.columns)
    assert set(t.metrics["part"]) == {"cv", "val", "test"}
    assert set(t.metrics["metric"]) == set(ft.DEFAULT_METRICS)
    assert set(t.segments["segment"]) == {"s1", "s2"}
    logloss = t.metrics.query("name == 'x1' and part == 'test' and metric == 'logloss'").iloc[0]
    assert logloss["new"] < logloss["base"] and logloss["delta"] > 0  # дельта — «стало лучше»
    for fig in (t.plot(), t.plot_segments(), t.plot_metrics()):
        assert isinstance(fig, go.Figure)
    t.save(tmp_path)
    assert (tmp_path / "feature_test_summary.csv").exists()
    with pytest.raises(ValueError):
        t.plot_domain("x")


def test_features_validation(split):
    with pytest.raises(ValueError, match="уже есть в базе"):
        ft.test_features(split, ["x1", "x2"], ["x1"], **QUICK)
    with pytest.raises(KeyError):
        ft.test_features(split, ["x1"], ["нет_такого"], **QUICK)
    with pytest.raises(ValueError, match="нет такой колонки"):
        ft.test_features(split, "нет_скора", ["x1"], **QUICK)


def test_features_on_top_of_score(binary_df, split):
    model = train_model(split, ["x2", "cat"], params=FAST)
    scored = binary_df.assign(score=model.predict(binary_df))
    scored_split = time_split(scored, "target", "report_date", oot_from="2024-03-01")
    t = ft.test_features(scored_split, "score", ["x1", "noise"], metrics=("gini", "ks"), **QUICK)
    s = t.summary.set_index("name")
    assert t.info["how"] == "on_top" and t.info["metric"] == "gini"
    assert s.loc["x1", "delta_test"] > 0.05 and s.loc["x1", "corr_with"] == "base_score"
    assert set(t.metrics["metric"]) == {"gini", "ks"}
    with pytest.raises(ValueError, match="retrain"):
        t.replacement()
    with pytest.raises(ValueError, match="retrain"):
        ft.test_features(scored_split, "score", ["x1"], how="retrain", **QUICK)
    with pytest.raises(ValueError, match="вероятностью"):
        ft.test_features(scored_split, "x1", ["noise"], **QUICK)


def test_base_as_model(split, tmp_path):
    model = train_model(split, ["x2", "cat"], params=FAST)
    quick = {k: v for k, v in QUICK.items() if k != "params"}

    by_model = ft.test_features(split, model, ["x1"], **quick)
    assert by_model.info["how"] == "retrain" and "lgbm" in by_model.info["base"]
    on_top = ft.test_features(split, model, ["x1"], how="on_top", **quick)
    assert on_top.summary.loc[0, "delta_test"] > 0

    with Experiment("ft", root=tmp_path) as exp:
        pkl = exp.save_model(model)
    by_pkl = ft.test_features(split, pkl, ["x1"], **quick)
    export = model.export(tmp_path / "export")["directory"]
    by_export = ft.test_features(split, export, ["x1"], **quick)
    assert by_pkl.summary.loc[0, "delta_cv"] == pytest.approx(by_model.summary.loc[0, "delta_cv"])
    assert by_export.summary.loc[0, "delta_test"] > 0

    native = lgb.LGBMClassifier(**FAST, verbosity=-1).fit(split.train[["x2", "noise"]],
                                                          split.train["target"])
    by_native = ft.test_features(split, native, ["x1"], **quick)
    assert "2 признаков" in by_native.info["base"]
    native.booster_.save_model(str(tmp_path / "model.txt"))
    by_file = ft.test_features(split, tmp_path / "model.txt", ["x1"], **quick)
    assert by_file.summary.loc[0, "verdict"] == "better"


def test_screening(split):
    t = ft.test_features(split, ["x2"], ["x1", "noise", "cat", "const"], max_exact=1, **QUICK)
    s = t.summary.set_index("name")
    assert (s["verdict"] == "screened out").sum() == 3
    assert s.loc["x1", "verdict"] == "better" and s["screen_delta"].idxmax() == "x1"
    assert s.loc["ALL", "n_features"] == 4 and not np.isnan(s.loc["ALL", "delta_cv"])
    assert isinstance(t.plot(), go.Figure)


def test_replacement(split):
    t = ft.test_features(split, ["x1_copy", "x2"], ["x1", "noise"], **QUICK)
    r = t.replacement(threshold=0.7)
    assert r.table[["candidate", "replaces"]].to_numpy().tolist() == [["x1", "x1_copy"]]
    assert r.table.loc[0, "corr"] > 0.9
    assert isinstance(r.plot(), go.Figure)
    assert len(t.replacement(threshold=0.9999).table) == 0


def test_domains(binary_df, tmp_path):
    df = binary_df.copy()
    df.loc[df["report_date"] < "2023-04-01", ["x1", "x1_copy"]] = np.nan
    split = time_split(df, "target", "report_date", oot_from="2024-03-01")
    t = ft.test_domains(split, ["x2", "cat"], {"сильный": ["x1", "x1_copy"], "шум": ["noise"]},
                        segment="segment", date_col="report_date", **QUICK)
    s = t.summary.set_index("name")
    assert list(t.summary["name"]) == ["сильный", "шум"]
    assert s.loc["сильный", "verdict"] == "better" and s.loc["сильный", "coverage"] < 1
    assert s.loc["сильный", "filled_from"] >= pd.Timestamp("2023-04-01")
    assert s.loc["сильный", "gain_share"] > s.loc["шум", "gain_share"]
    assert s.loc["сильный", "alone_auc_test"] > s.loc["шум", "alone_auc_test"]
    shares = t.features.groupby("domain")["share"].sum()
    assert shares["сильный"] == pytest.approx(1.0)
    assert isinstance(t.plot_domain("сильный"), go.Figure)
    with pytest.raises(ValueError):
        t.plot_domain("нет")

    with Experiment("ft", root=tmp_path) as exp:
        exp.log(t)
        assert (exp.logs_path / "domain_test_summary.csv").exists()
        assert (exp.logs_path / "domain_test_delta.html").exists()
