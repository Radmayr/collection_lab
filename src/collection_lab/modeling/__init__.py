"""Обучение, сравнение алгоритмов, подбор гиперпараметров (Optuna), влияние объёма выборки."""

from collection_lab.modeling.sample_size import sample_size_curve
from collection_lab.modeling.train import compare_models, train_model
from collection_lab.modeling.tuning import (
    catboost_search_space,
    lgbm_search_space,
    tune_hyperparams,
)

__all__ = [
    "catboost_search_space",
    "compare_models",
    "lgbm_search_space",
    "sample_size_curve",
    "train_model",
    "tune_hyperparams",
]
