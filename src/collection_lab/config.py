"""Параметры моделей по умолчанию — единственное место, где они задаются.

Функции библиотеки берут отсюда копию и обновляют её пользовательскими параметрами
через :func:`merge_params`, поэтому менять дефолты нужно только здесь.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

RANDOM_STATE = 42

# objective / loss_function выставляет адаптер модели по типу задачи (binary / regression).
LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 16,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}

# Маленькая модель для однофакторной оценки признаков.
LGBM_UNIVARIATE_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 50,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,  # параллелим снаружи, по признакам
    "verbosity": -1,
}

CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 5000,
    "learning_rate": 0.03,
    "depth": 4,
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}

EARLY_STOPPING_ROUNDS = 100


def merge_params(defaults: dict[str, Any], params: dict[str, Any] | None) -> dict[str, Any]:
    """Возвращает копию ``defaults``, обновлённую ``params`` (исходные словари не меняются)."""
    merged = deepcopy(defaults)
    if params:
        merged.update(params)
    return merged
