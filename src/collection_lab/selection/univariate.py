"""Однофакторная оценка предсказательной силы признаков."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed

from collection_lab.config import LGBM_UNIVARIATE_PARAMS, RANDOM_STATE, merge_params
from collection_lab.core.cv import Folds, cross_validate, make_folds
from collection_lab.core.results import Result
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import roc_auc
from collection_lab.plotting.theme import style
from collection_lab.utils.parallel import tqdm_joblib


def direct_auc(x, y, folds: Folds | None = None) -> float:
    """AUC признака как скора без модели, инвариантный к направлению: ``max(AUC, 1 - AUC)``.

    С ``folds`` — среднее по валидационным фолдам, пропуски заполняются медианой
    train-части фолда (как в ``select_features_by_auc_cv_hybrid``). Без ``folds`` —
    по всей выборке с заполнением медианой.
    """
    x = pd.Series(np.asarray(x, dtype="float64"))
    y = np.asarray(y)
    if folds is None:
        folds = [(np.arange(len(x)), np.arange(len(x)))]
    aucs = []
    for tr, va in folds:
        x_tr, x_va = x.iloc[tr], x.iloc[va]
        if x_tr.nunique(dropna=False) < 2 or x_va.nunique(dropna=False) < 2:
            aucs.append(0.5)
            continue
        s = x_va.fillna(x_tr.median())
        a = roc_auc(y[va], s)
        aucs.append(0.5 if np.isnan(a) else max(a, 1 - a))
    return float(np.mean(aucs))


def _score_feature(
    feature: str, X: pd.DataFrame, y: np.ndarray, is_cat: bool, folds: Folds,
    params: dict[str, Any], early_stopping_rounds: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {"feature": feature, "auc_mean": np.nan, "auc_std": np.nan,
                           "auc_min": np.nan, "auc_max": np.nan, "auc_gain": np.nan,
                           "n_splits_ok": 0, "status": "ok"}
    s = X[feature]
    if s.isna().all() or s.nunique(dropna=True) < 2:
        row["status"] = "constant_or_all_nan"
        return row
    try:
        res = cross_validate(X[[feature]], y, cat_features=[feature] if is_cat else [],
                             params=params, folds=folds, return_oof=False,
                             early_stopping_rounds=early_stopping_rounds)
    except Exception as e:  # noqa: BLE001 — ошибка одной фичи не должна валить весь расчёт
        row["status"] = f"error: {type(e).__name__}: {e}"
        return row
    aucs = np.array([a for a in res.fold_scores if not np.isnan(a)])
    if len(aucs) == 0:
        row["status"] = "no_valid_folds"
        return row
    row.update(auc_mean=aucs.mean(), auc_std=aucs.std(), auc_min=aucs.min(), auc_max=aucs.max(),
               auc_gain=abs(aucs.mean() - 0.5), n_splits_ok=len(aucs))
    return row


def univariate_scores(
    X: pd.DataFrame,
    y,
    features: Sequence[str] | None = None,
    *,
    cat_features: Iterable[str] | None = None,
    n_splits: int = 5,
    test_size: float = 0.2,
    folds: Folds | None = None,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Result:
    """AUC маленькой модели LightGBM на каждом признаке отдельно
    (бывший ``evaluate_features_univariate``).

    Parameters
    ----------
    n_splits : int
        Число фолдов; ``<= 1`` — holdout с долей ``test_size``.
    params : dict, optional
        Параметры поверх ``config.LGBM_UNIVARIATE_PARAMS``.
    n_jobs : int
        Параллельность по признакам (процессы joblib).

    Returns
    -------
    Result
        ``table``: ``feature, auc_mean, auc_std, auc_min, auc_max, auc_gain, n_splits_ok,
        status`` по убыванию ``auc_gain``; ``selected`` — признаки со статусом ``ok``.
    """
    features = list(X.columns if features is None else features)
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"Признаки не найдены в X: {missing}")
    cat_set = set(detect_categorical(X[features]) if cat_features is None else cat_features)
    y = np.asarray(y)
    X = X[features].reset_index(drop=True)
    if folds is None:
        folds = make_folds(y, n_splits, test_size=test_size, random_state=random_state)
    params = merge_params(LGBM_UNIVARIATE_PARAMS, params)

    tasks = (delayed(_score_feature)(f, X[[f]], y, f in cat_set, folds, params,
                                     early_stopping_rounds) for f in features)
    with tqdm_joblib(total=len(features), desc="Univariate AUC", disable=not verbose):
        rows = Parallel(n_jobs=n_jobs)(tasks)

    table = (pd.DataFrame(rows).sort_values("auc_gain", ascending=False, na_position="last")
             .reset_index(drop=True))
    selected = table.loc[table["status"] == "ok", "feature"].tolist()
    info = {"n_splits": len(folds), "params": params}
    return Result("univariate_scores", table, selected, info, plotter=_plot_univariate)


def univariate_filter(
    X: pd.DataFrame, y, features: Sequence[str] | None = None, *, min_auc: float = 0.52,
    **kwargs,
) -> Result:
    """Оставляет признаки с однофакторным AUC ``>= min_auc`` (см. :func:`univariate_scores`)."""
    res = univariate_scores(X, y, features, **kwargs)
    table = res.table.copy()
    keep = (table["status"] == "ok") & (table["auc_mean"] >= min_auc)
    table["action"] = np.where(keep, "keep", "drop")
    table["reason"] = np.where(keep, "", np.where(table["status"] == "ok",
                                                  f"auc_mean < {min_auc}", table["status"]))
    selected = table.loc[keep, "feature"].tolist()
    return Result("univariate_filter", table, selected, {**res.info, "min_auc": min_auc},
                  plotter=_plot_univariate)


def _plot_univariate(result: Result, top_k: int = 40) -> go.Figure:
    t = result.table.dropna(subset=["auc_mean"]).head(top_k).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["auc_mean"], y=t["feature"], orientation="h",
        error_x={"type": "data", "array": t["auc_std"]},
        hovertemplate="%{y}: AUC=%{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey")
    lo = max(0.45, float(t["auc_mean"].min()) - 0.02) if len(t) else 0.45
    fig.update_xaxes(range=[lo, float(t["auc_mean"].max()) + 0.02] if len(t) else None)
    return style(fig, f"Однофакторный AUC (top-{min(top_k, len(t))})",
                 height=max(350, 22 * len(t) + 120), xaxis_title="AUC")
