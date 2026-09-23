import numpy as np
import pandas as pd
import pytest

from collection_lab.core import (
    FeaturePreparer,
    LGBMModel,
    cross_validate,
    make_folds,
    make_lgbm_ready,
    make_model,
)

FEATURES = ["x1", "x2", "noise", "cat"]
FAST = {"n_estimators": 200, "learning_rate": 0.1}


def test_preparer_fixes_categories_on_train():
    train = pd.DataFrame({"c": ["a", "b", None], "n": ["1", "2", "3"]})
    new = pd.DataFrame({"c": ["b", "z"], "n": ["5", "nan"]})
    prep = FeaturePreparer().fit(train)
    assert prep.cat_features_ == ["c"]
    out = prep.transform(new)
    assert list(out["c"].cat.categories) == ["a", "b"]
    assert out["c"].isna().tolist() == [False, True]  # неизвестная категория → NaN
    assert out["n"].dtype == "float64"


def test_make_lgbm_ready_compat():
    X, cats = make_lgbm_ready(pd.DataFrame({"c": ["a", "b"], "n": [1, 2]}))
    assert cats == ["c"] and str(X["c"].dtype) == "category"


def test_make_folds():
    y = np.array([0, 1] * 50)
    folds = make_folds(y, 5)
    assert len(folds) == 5
    all_va = np.concatenate([va for _, va in folds])
    assert sorted(all_va) == list(range(100))
    (tr, va), = make_folds(y, 1, test_size=0.3)
    assert len(va) == 30 and not set(tr) & set(va)


def test_lgbm_model_fit_predict(binary_df):
    X, y = binary_df[FEATURES], binary_df["target"]
    m = make_model("lgbm", FAST, early_stopping_rounds=20)
    m.fit(X.iloc[:3000], y.iloc[:3000], eval_set=(X.iloc[3000:], y.iloc[3000:]))
    p = m.predict(X.iloc[3000:])
    assert p.shape == (1000,) and ((0 <= p) & (p <= 1)).all()
    assert m.cat_features_ == ["cat"]
    imp = m.feature_importance()
    assert set(imp.index) == set(FEATURES)
    assert imp.index[0] in {"x1", "cat", "x2"}
    assert m.best_iteration_ is None or m.best_iteration_ <= 200
    assert isinstance(m.clone(), LGBMModel) and m.clone().estimator_ is None


def test_regression_task(binary_df):
    X = binary_df[["x1", "noise"]]
    m = make_model("lgbm", FAST, task="regression").fit(X, binary_df["x1"] * 2)
    assert np.corrcoef(m.predict(X), binary_df["x1"])[0, 1] > 0.9


def test_cross_validate_with_segments(binary_df):
    res = cross_validate(binary_df, binary_df["target"], features=FEATURES, params=FAST,
                         n_splits=3, segments="segment")
    assert 0.7 < res.mean < 0.95
    assert len(res.fold_scores) == 3
    assert set(res.segment_scores.index) == {"s1", "s2"}
    assert res.oof.notna().all() and res.oof.index.equals(binary_df.index)
    assert "auc_seg=s1" in res.summary()


def test_cross_validate_same_folds_reproducible(binary_df):
    folds = make_folds(binary_df["target"], 3)
    kw = dict(features=FEATURES, params=FAST, folds=folds)
    a = cross_validate(binary_df, binary_df["target"], **kw)
    b = cross_validate(binary_df, binary_df["target"], n_jobs=3, **kw)
    assert a.fold_scores == pytest.approx(b.fold_scores)


def test_cross_validate_noise_worse_than_signal(binary_df):
    folds = make_folds(binary_df["target"], 3)
    good = cross_validate(binary_df, binary_df["target"], features=["x1"], params=FAST,
                          folds=folds)
    bad = cross_validate(binary_df, binary_df["target"], features=["noise"], params=FAST,
                         folds=folds)
    assert good.mean > bad.mean + 0.1


def test_catboost_adapter(binary_df):
    pytest.importorskip("catboost")
    res = cross_validate(binary_df, binary_df["target"], features=FEATURES, model="catboost",
                         params={"iterations": 150, "learning_rate": 0.1}, n_splits=2)
    assert res.mean > 0.7
    assert res.best_iterations[0] is not None
