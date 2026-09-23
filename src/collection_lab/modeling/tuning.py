"""Подбор гиперпараметров Optuna без утечки: выбор по val, test — только для отчёта."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from collection_lab.config import RANDOM_STATE
from collection_lab.core.models import BaseModel, LGBMModel, make_model
from collection_lab.core.results import Result
from collection_lab.data.split import DataSplit
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import get_metric
from collection_lab.plotting.theme import style


def lgbm_search_space(trial) -> dict[str, Any]:
    """Пространство поиска LightGBM из ноутбука RTK (неглубокие деревья, регуляризация)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 4, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
    }


def catboost_search_space(trial) -> dict[str, Any]:
    """Пространство поиска CatBoost."""
    return {
        "depth": trial.suggest_int("depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
    }


def tune_hyperparams(
    split: DataSplit,
    features: Sequence[str],
    *,
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    search_space: Callable[[Any], dict[str, Any]] | None = None,
    fixed_params: dict[str, Any] | None = None,
    n_trials: int = 100,
    metric: str = "auc",
    early_stopping_rounds: int | None = 50,
    max_train_size: int | None = 300_000,
    timeout: float | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Result:
    """Optuna-поиск: каждое испытание обучается на train, оценивается на val.

    Лучшие параметры выбираются **только по val**. Метрика на test записывается в таблицу
    для контроля переобучения, но в выборе не участвует (в ноутбуке RTK лучший набор
    выбирался по test — это утечка).

    Parameters
    ----------
    search_space : callable, optional
        ``trial -> dict`` параметров; по умолчанию :func:`lgbm_search_space` /
        :func:`catboost_search_space`.
    fixed_params : dict, optional
        Параметры, общие для всех испытаний (например, ``{"n_estimators": 1000}``).
    max_train_size : int, optional
        Подвыборка train для ускорения поиска.

    Returns
    -------
    Result
        ``table`` — испытания (``trial, <metric>_train, <metric>_val, <metric>_test,
        best_iteration, params…``); ``info["best_params"]`` — лучшие параметры
        (включая ``fixed_params`` и число деревьев), ``info["study"]`` — объект Optuna.
    """
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError("Нужен optuna: pip install collection_lab[optuna]") from e

    features = list(features)
    metric_obj = get_metric(metric)
    train = split.train
    if max_train_size and len(train) > max_train_size:
        train = train.sample(n=max_train_size, random_state=random_state)
    X_tr, y_tr = train[features], train[split.target]
    X_va, y_va = split.xy("val", features)
    test = split.test
    if cat_features is None:
        cat_features = detect_categorical(X_tr)
    cat_features = [c for c in cat_features if c in features]
    base = make_model(model, fixed_params, early_stopping_rounds=early_stopping_rounds)
    if search_space is None:
        search_space = lgbm_search_space if isinstance(base, LGBMModel) else catboost_search_space
    records: list[dict[str, Any]] = []

    def objective(trial) -> float:
        params = search_space(trial)
        m = base.clone(params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features)
        rec = {"trial": trial.number,
               f"{metric_obj.name}_train": metric_obj(y_tr, m.predict(X_tr)),
               f"{metric_obj.name}_val": metric_obj(y_va, m.predict(X_va)),
               "best_iteration": m.n_iterations_, **params}
        if test is not None:
            rec[f"{metric_obj.name}_test"] = metric_obj(test[split.target],
                                                        m.predict(test[features]))
        records.append(rec)
        return rec[f"{metric_obj.name}_val"]

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize" if metric_obj.greater_is_better else "minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)

    table = pd.DataFrame(records)
    best = table.loc[table["trial"] == study.best_trial.number].iloc[0]
    key = "n_estimators" if isinstance(base, LGBMModel) else "iterations"
    best_params = {**(fixed_params or {}), **study.best_params, key: int(best["best_iteration"])}
    info = {"metric": metric_obj.name, "best_params": best_params,
            "best_trial": int(study.best_trial.number),
            "best_val": float(best[f"{metric_obj.name}_val"]),
            "best_test": float(best.get(f"{metric_obj.name}_test", np.nan)), "study": study}
    if verbose:
        print(f"Лучший val {metric_obj.name}: {info['best_val']:.4f}"
              + (f" (test: {info['best_test']:.4f})" if test is not None else ""))
    return Result("tune_hyperparams", table, None, info, plotter=_plot_tuning)


def _plot_tuning(result: Result) -> go.Figure:
    t, name = result.table, result.info["metric"]
    fig = go.Figure()
    for part, dash in (("train", "dot"), ("val", "solid"), ("test", "dash")):
        col = f"{name}_{part}"
        if col in t.columns:
            fig.add_scatter(x=t["trial"], y=t[col], mode="lines+markers", name=part,
                            line={"dash": dash})
    fig.add_vline(x=result.info["best_trial"], line_dash="dash", line_color="grey",
                  annotation_text="лучший по val")
    return style(fig, f"Optuna: {name} по испытаниям", height=450, xaxis_title="trial",
                 yaxis_title=name)
