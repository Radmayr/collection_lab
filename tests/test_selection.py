import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from collection_lab.data import random_split
from collection_lab.selection import (
    RFE,
    CorrelationFilter,
    CumulativeImportance,
    Custom,
    QualityFilter,
    SelectionPipeline,
    UnivariateFilter,
    backward_elimination,
    correlation_filter,
    correlation_groups,
    cumulative_importance_selection,
    direct_auc,
    drop_column_importance,
    forward_addition,
    incremental_feature_eval,
    quality_filter,
    rfe,
    univariate_scores,
)

FAST = {"n_estimators": 150, "learning_rate": 0.1}
FEATURES = ["x1", "x2", "x1_copy", "noise", "cat"]


def test_quality_filter(binary_df):
    res = quality_filter(binary_df, ["x1", "x2", "const", "all_nan", "cat"])
    t = res.table.set_index("feature")
    assert set(res.dropped) == {"const", "all_nan"}
    assert t.loc["cat", "type"] == "categorical"
    assert "Пропущено" in t.loc["all_nan", "reason"]
    assert isinstance(res.plot(), go.Figure)


def test_quality_filter_cardinality():
    df = pd.DataFrame({"many": [f"id_{i}" for i in range(200)], "one": ["a"] * 200})
    res = quality_filter(df, cat_unique_max=100)
    assert set(res.dropped) == {"many", "one"}


def test_direct_auc_direction_invariant(binary_df):
    y = binary_df["target"]
    assert direct_auc(binary_df["x1"], y) == pytest.approx(direct_auc(-binary_df["x1"], y))
    assert direct_auc(binary_df["x1"], y) > direct_auc(binary_df["noise"], y)


def test_univariate_scores(binary_df):
    res = univariate_scores(binary_df, binary_df["target"], ["x1", "noise", "cat", "const"],
                            n_splits=3, n_jobs=1, verbose=False)
    t = res.table.set_index("feature")
    assert t.loc["const", "status"] == "constant_or_all_nan"
    assert t.loc["x1", "auc_mean"] > t.loc["noise", "auc_mean"]
    assert t.loc["cat", "auc_mean"] > 0.55
    assert res.table["feature"].iloc[0] == "x1"
    assert "const" not in res.selected


def test_correlation_groups_and_filter(binary_df):
    groups, corr = correlation_groups(binary_df[["x1", "x1_copy", "x2", "noise"]], 0.8)
    assert groups == [["x1", "x1_copy"]]
    for strategy in ("direct", "model", "hybrid"):
        res = correlation_filter(binary_df, binary_df["target"], FEATURES, strategy=strategy,
                                 n_splits=3, params=FAST, verbose=False)
        assert len(res.dropped) == 1 and res.dropped[0] in {"x1", "x1_copy"}
        assert "cat" in res.selected  # нечисловые проходят без изменений
    assert isinstance(res.plot(), go.Figure)


def test_cumulative_importance(binary_df):
    res = cumulative_importance_selection(binary_df, "target", ["x1", "x2", "noise", "cat"],
                                          threshold=0.8, params=FAST)
    assert res.selected[0] in {"x1", "cat", "x2"}
    assert 1 <= len(res.selected) <= 4
    assert res.info["auc_full"] > 0.7
    assert res.table["importance_cum"].iloc[-1] == pytest.approx(1.0)
    perm = cumulative_importance_selection(binary_df, "target", ["x1", "noise"],
                                           importance="permutation", params=FAST, n_repeats=2)
    assert perm.selected[0] == "x1"


def test_drop_column_importance(binary_df):
    split = random_split(binary_df, "target", test_size=0.2)
    res = drop_column_importance(split.train, split.val, ["x1", "x2", "noise"], "target",
                                 test=split.test, params=FAST, verbose=False)
    t = res.table.set_index("feature")
    assert t.loc["x1", "delta_val"] > t.loc["noise", "delta_val"]
    assert "delta_test" in t.columns
    assert isinstance(res.plot(), go.Figure)


def test_rfe_drops_noise_keeps_signal(binary_df):
    res = rfe(binary_df, binary_df["target"], ["x1", "x2", "noise", "const"], n_splits=3,
              tol=0.002, params=FAST, verbose=False)
    assert "x1" in res.selected
    assert "const" in res.dropped  # префильтр константы
    assert res.table["action"].iloc[-1] == "final"
    assert isinstance(res.plot(), go.Figure)


def test_rfe_segments_and_catboost(binary_df):
    res = rfe(binary_df, binary_df["target"], ["x1", "x2", "noise"], segments="segment",
              segment_tol=0.01, tol=0.01, n_splits=2, params=FAST, verbose=False)
    assert {"min_segment_delta", "worst_segment"} <= set(res.table.columns)
    assert any(c.startswith("seg_before_") for c in res.table.columns)
    pytest.importorskip("catboost")
    res_cb = rfe(binary_df, binary_df["target"], ["x1", "noise", "cat"], model="catboost",
                 params={"iterations": 100, "learning_rate": 0.1}, n_splits=2, tol=0.005,
                 min_features=2, verbose=False)
    assert "x1" in res_cb.selected


def test_backward_and_forward(binary_df):
    split = random_split(binary_df, "target")
    Xtr, Xva = split.train, split.val
    ytr, yva = Xtr["target"], Xva["target"]
    back = backward_elimination(Xtr, ytr, Xva, yva, ["x1", "x2", "noise", "cat"],
                                segments="segment", min_segment_size=50, params=FAST,
                                verbose=False)
    assert len(back.table) == 4
    assert "delta_after_remove_valid" in back.table.columns
    assert isinstance(back.plot(), go.Figure)
    assert isinstance(back.plot(kind="per_feature"), go.Figure)

    fwd = forward_addition(Xtr, ytr, Xva, yva, ["x2"], ["x1", "noise"], params=FAST,
                           segments="segment", min_segment_size=50, verbose=False)
    t = fwd.table.set_index("added")
    assert t.loc["x1", "delta_auc_valid"] > t.loc["noise", "delta_auc_valid"]
    groups = forward_addition(Xtr, ytr, Xva, yva, ["x2"], {"g": ["x1", "cat"]}, mode="groups",
                              params=FAST, verbose=False)
    assert list(groups.table["added"]) == ["BASE", "g"]
    assert isinstance(fwd.plot(), go.Figure)


@pytest.mark.parametrize("direction,mode", [("forward", "ordered"), ("backward", "greedy")])
def test_incremental(binary_df, direction, mode):
    res = incremental_feature_eval(binary_df, binary_df["target"], ["x1", "x2", "noise"],
                                   direction=direction, mode=mode, n_splits=2, params=FAST,
                                   n_jobs=1, verbose=False)
    # backward не считает шаг с нулём признаков (как исходная функция)
    assert len(res.table) == (3 if direction == "forward" else 2)
    assert {"auc_mean", "auc_signal_ratio", "features_set"} <= set(res.table.columns)
    assert res.selected
    assert isinstance(res.plot(), go.Figure)


def test_pipeline(binary_df, tmp_path):
    feats = ["x1", "x2", "x1_copy", "noise", "cat", "const", "all_nan"]
    pipe = SelectionPipeline([
        QualityFilter(),
        CorrelationFilter(threshold=0.8, n_splits=2, verbose=False),
        UnivariateFilter(min_auc=0.51, n_splits=2, n_jobs=1, verbose=False),
        CumulativeImportance(threshold=0.99, params=FAST),
        RFE(tol=0.005, n_splits=2, params=FAST, verbose=False),
        Custom(lambda df, t, f, c: [x for x in f if x != "noise"], name="no_noise"),
    ]).fit(binary_df, "target", feats, verbose=False)
    assert "x1" in pipe.selected_ or "x1_copy" in pipe.selected_
    assert not {"const", "all_nan", "noise"} & set(pipe.selected_)
    s = pipe.summary()
    assert s["n_in"].iloc[0] == len(feats)
    assert (s["n_in"] - s["n_dropped"] == s["n_out"]).all()
    assert set(pipe.log_["feature"]) == set(feats) - set(pipe.selected_)
    assert isinstance(pipe.plot(), go.Figure)
    pipe.save(tmp_path)
    assert (tmp_path / "selection_log.csv").exists()
    assert np.isfinite(pipe.results_["1_quality_filter"].info["n_dropped"])


def test_incremental_eval_sets_columns_and_plot(binary_df):
    split = random_split(binary_df, "target", test_size=0.25)
    feats = ["x1", "x2", "noise", "cat"]
    res = incremental_feature_eval(
        split.train, split.train["target"], feats, eval_sets=split.eval_sets(),
        n_splits=2, params=FAST, n_jobs=1, verbose=False)
    t = res.table
    expected = {"auc_val", "auc_test", "auc_delta_val", "auc_delta_test", "auc_mean"}
    assert expected <= set(t.columns)
    assert t["auc_val"].between(0.4, 1).all() and t["auc_test"].between(0.4, 1).all()
    assert np.isnan(t["auc_delta_val"].iloc[0])        # у первого шага дельты нет
    assert res.info["eval_sets"] == ["val", "test"] and "cv" in res.info["best_step"]
    fig = res.plot()
    names = {tr.name for tr in fig.data}
    assert {"train (CV)", "val", "test"} <= names


def test_incremental_eval_sets_any_number_and_names(binary_df):
    split = random_split(binary_df, "target", test_size=0.25)
    Xa, Xb = split.val, split.test
    sets = {"a": (Xa, Xa["target"]), "b": (Xb, Xb["target"]), "c_oot": (Xb, Xb["target"])}
    res = incremental_feature_eval(split.train, split.train["target"], ["x1", "noise"],
                                   eval_sets=sets, n_splits=2, params=FAST, n_jobs=1,
                                   verbose=False)
    assert {"auc_a", "auc_b", "auc_c_oot"} <= set(res.table.columns)
    only_val = incremental_feature_eval(split.train, split.train["target"], ["x1", "noise"],
                                        eval_sets=split.eval_sets(["val"]), n_splits=2,
                                        params=FAST, n_jobs=1, verbose=False)
    assert "auc_test" not in only_val.table.columns
    with pytest.raises(ValueError, match="нет признаков"):
        incremental_feature_eval(split.train, split.train["target"], ["x1", "noise"],
                                 eval_sets={"bad": (Xa[["x1"]], Xa["target"])}, n_splits=2,
                                 params=FAST, verbose=False)


@pytest.mark.parametrize("mode", ["ordered", "greedy"])
def test_incremental_eval_sets_do_not_affect_selection(binary_df, mode):
    """Наборы только оцениваются: порядок признаков и CV-метрика те же, что без наборов."""
    split = random_split(binary_df, "target", test_size=0.25)
    kw = dict(direction="forward", mode=mode, n_splits=2, params=FAST, n_jobs=1, verbose=False)
    feats = ["x1", "x2", "noise"]
    without = incremental_feature_eval(split.train, split.train["target"], feats, **kw)
    with_sets = incremental_feature_eval(split.train, split.train["target"], feats,
                                         eval_sets=split.eval_sets(), **kw)
    assert list(without.table["feature_changed"]) == list(with_sets.table["feature_changed"])
    assert np.allclose(without.table["auc_mean"], with_sets.table["auc_mean"])


def test_datasplit_eval_sets(binary_df):
    with_test = random_split(binary_df, "target", test_size=0.2)
    assert list(with_test.eval_sets()) == ["val", "test"]
    no_test = random_split(binary_df, "target")
    assert list(no_test.eval_sets()) == ["val"]           # отсутствующий test пропускается
    with pytest.raises(KeyError):
        no_test.eval_sets(["nope"])


def test_univariate_tree_scores(binary_df):
    from collection_lab.selection import univariate_tree_scores

    df = binary_df.assign(sparse=np.where(binary_df.index % 4 == 0, 1.0, np.nan))
    res = univariate_tree_scores(df, df["target"], ["x1", "noise", "cat", "const", "sparse"],
                                 max_depth=2, min_leaf_share=0.1, n_splits=3, verbose=False)
    t = res.table.set_index("feature")
    assert t.index[0] == "x1" and t.loc["x1", "auc_mean"] > t.loc["noise", "auc_mean"]
    assert t.loc["const", "status"] == "constant_or_all_nan"
    assert t.loc["sparse", "status"] == "ok"           # «заполнен / пропуск» — тоже разбиение
    assert 2 <= t.loc["x1", "n_leaves"] <= 4 and bool(t.loc["x1", "monotonic"])
    assert np.isnan(t.loc["cat", "monotonic"])
    assert set(res.selected) == {"x1", "noise", "cat", "sparse"}

    leaves = res.info["leaves"]
    x1 = leaves[leaves["feature"] == "x1"]
    assert x1["n"].sum() == len(df) and x1["share"].sum() == pytest.approx(1.0)
    assert (x1["share"] >= 0.1 * 2 / 3 - 1e-9).all()   # не мельче min_leaf_share от фолда
    assert x1["target_rate"].is_monotonic_increasing    # x1 повышает риск
    cat = leaves[leaves["feature"] == "cat"]
    assert cat["target_rate"].is_monotonic_increasing   # категории — по доле таргета
    assert isinstance(res.plot(), go.Figure)
    assert isinstance(res.plot(feature="x1"), go.Figure)
    with pytest.raises(ValueError):
        res.plot(feature="const")
    with pytest.raises(ValueError):
        univariate_tree_scores(df, df["target"], ["x1"], min_leaf_share=0.7)


def _chain_df(weights=(2.0, 0.8)):
    """A ~ B ~ C: B связан с A и с C, A и C между собой независимы."""
    rng = np.random.default_rng(0)
    n = 6000
    a, c = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"A": a, "B": (a + c) / np.sqrt(2) + rng.normal(scale=0.2, size=n), "C": c})
    logit = weights[0] * a + weights[1] * c - 1
    df["target"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


@pytest.mark.parametrize("strategy", ["direct", "model", "hybrid"])
def test_correlation_filter_pairwise_breaks_chains(strategy):
    df = _chain_df()
    kwargs = {"threshold": 0.6, "strategy": strategy, "verbose": False, "n_splits": 3}
    groups = correlation_filter(df, df["target"], ["A", "B", "C"], **kwargs)
    pairs = correlation_filter(df, df["target"], ["A", "B", "C"], grouping="pairwise", **kwargs)
    assert groups.selected == ["A"]                   # одна группа — остаётся один признак
    assert pairs.selected == ["A", "C"]               # A и C не связаны — остаются оба
    t = pairs.table.set_index("feature")
    assert t.loc["B", "action"] == "drop" and t.loc["B", "kept"] == "A"
    assert t.loc["C", "action"] == "keep" and t.loc["C", "kept"] == "C"
    assert pairs.info["grouping"] == "pairwise"
    with pytest.raises(ValueError):
        correlation_filter(df, df["target"], ["A", "B"], grouping="graph")


def test_pairwise_hybrid_lets_model_overturn_close_pair():
    from collection_lab.selection.correlation import _pairwise_rows

    corr = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], index=["p", "q"], columns=["p", "q"])
    direct = {"p": 0.700, "q": 0.695}                 # по direct p чуть сильнее
    model = {"p": 0.71, "q": 0.73}.__getitem__        # а по модели — q
    rows = {r["feature"]: r for r in _pairwise_rows([["p", "q"]], corr, 0.8, "hybrid", direct,
                                                    model, auc_gap_threshold=0.01)}
    assert rows["q"]["action"] == "keep" and rows["p"]["action"] == "drop"
    assert rows["p"]["kept"] == "q" and "method=model" in rows["p"]["reason"]
    rows = {r["feature"]: r for r in _pairwise_rows([["p", "q"]], corr, 0.8, "direct", direct,
                                                    model, auc_gap_threshold=0.01)}
    assert rows["p"]["action"] == "keep" and rows["q"]["action"] == "drop"
