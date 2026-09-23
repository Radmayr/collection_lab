"""Обучение модели на :class:`DataSplit` и сравнение алгоритмов."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from collection_lab.config import EARLY_STOPPING_ROUNDS
from collection_lab.core.cv import Folds, cross_validate, make_folds
from collection_lab.core.models import BaseModel, LGBMModel, make_model
from collection_lab.data.split import DataSplit
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import get_metric


def train_model(
    split: DataSplit,
    features: Sequence[str],
    *,
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = EARLY_STOPPING_ROUNDS,
    auto_scale_pos_weight: bool = False,
    refit_on_train_val: bool = False,
    metric: str = "auc",
    verbose: int = 0,
) -> BaseModel:
    """Обучает модель на ``split.train`` с early stopping по ``split.val``
    (бывший ``train_lgbm``, исправлен ``auto_scale_pos_weight``).

    Parameters
    ----------
    auto_scale_pos_weight : bool
        ``scale_pos_weight = neg / pos`` по train (только LightGBM).
    refit_on_train_val : bool
        После подбора числа деревьев переобучить на train + val с этим числом деревьев
        (финальная модель; test остаётся независимым).
    verbose : int
        Период печати метрик на итерациях (0 — молча).

    Returns
    -------
    BaseModel
        Обученный адаптер; ``model.scores_`` — метрика на каждой части сплита.
    """
    features = list(features)
    X_tr, y_tr = split.xy("train", features)
    X_va, y_va = split.xy("val", features)
    if cat_features is None:
        cat_features = detect_categorical(X_tr)
    cat_features = [c for c in cat_features if c in features]
    params = dict(params or {})
    m = make_model(model, params, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    if auto_scale_pos_weight:
        if not isinstance(m, LGBMModel):
            raise ValueError("auto_scale_pos_weight поддерживается только для LightGBM")
        pos = int(np.sum(y_tr))
        m = m.clone({"scale_pos_weight": (len(y_tr) - pos) / max(pos, 1)})
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features)

    if refit_on_train_val:
        n_iter = m.n_iterations_
        key = "n_estimators" if isinstance(m, LGBMModel) else "iterations"
        final = m.clone({key: n_iter})
        final.early_stopping_rounds = None
        X_all = pd.concat([X_tr, X_va])
        y_all = np.concatenate([np.asarray(y_tr), np.asarray(y_va)])
        final.fit(X_all, y_all, cat_features=cat_features)
        m = final

    metric_obj = get_metric(metric)
    m.scores_ = {name: metric_obj(df[split.target], m.predict(df[features]))
                 for name, df in split.items()}
    if verbose:
        print(", ".join(f"{k}: {metric_obj.name}={v:.4f}" for k, v in m.scores_.items()))
    return m


def compare_models(
    X: pd.DataFrame,
    y,
    models: dict[str, str | BaseModel] | Sequence[str] = ("lgbm", "catboost"),
    *,
    features: Sequence[str] | None = None,
    cat_features: Iterable[str] | None = None,
    params: dict[str, dict[str, Any]] | None = None,
    n_splits: int = 5,
    folds: Folds | None = None,
    metric: str = "auc",
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CV-сравнение моделей на одних и тех же фолдах (бывший ``compare_catboost_vs_lgbm``).

    Parameters
    ----------
    models : dict | list
        ``{"имя": "lgbm" | "catboost" | BaseModel}`` или список имён.
    params : dict, optional
        Параметры по имени модели: ``{"lgbm": {...}, "catboost": {...}}``.

    Returns
    -------
    summary : pd.DataFrame
        ``model, mean_<metric>, std_<metric>, mean_fit_time_sec`` по убыванию качества.
    oof : pd.DataFrame
        ``fold, y_true, <model>_pred, ...`` с индексом ``X``.
    """
    if not isinstance(models, dict):
        models = {name: name for name in models}
    params = params or {}
    y = np.asarray(y)
    folds = folds or make_folds(y, n_splits)
    metric_obj = get_metric(metric)
    fold_id = np.empty(len(y), dtype=int)
    for k, (_, va) in enumerate(folds, start=1):
        fold_id[va] = k
    oof = pd.DataFrame({"fold": fold_id, "y_true": y}, index=X.index)
    rows = []
    for name, spec in models.items():
        t0 = time.perf_counter()
        res = cross_validate(X, y, features=features, cat_features=cat_features, model=spec,
                             params=params.get(name), folds=folds, metric=metric_obj,
                             early_stopping_rounds=early_stopping_rounds)
        elapsed = (time.perf_counter() - t0) / len(folds)
        oof[f"{name}_pred"] = res.oof.to_numpy()
        rows.append({"model": name, f"mean_{metric_obj.name}": res.mean,
                     f"std_{metric_obj.name}": float(np.std(res.fold_scores, ddof=1))
                     if len(res.fold_scores) > 1 else 0.0,
                     "mean_fit_time_sec": elapsed, "best_iteration": res.mean_best_iteration})
    summary = pd.DataFrame(rows).sort_values(
        f"mean_{metric_obj.name}", ascending=not metric_obj.greater_is_better, ignore_index=True)
    return summary, oof
