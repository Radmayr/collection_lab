"""Ядро: адаптеры моделей (LightGBM/CatBoost), кросс-валидация с сегментами, объекты результатов."""

from collection_lab.core.cv import CVResult, cross_validate, make_folds
from collection_lab.core.models import (
    MODELS,
    BaseModel,
    CatBoostModel,
    FeaturePreparer,
    LGBMModel,
    make_lgbm_ready,
    make_model,
)

__all__ = [
    "MODELS",
    "BaseModel",
    "CVResult",
    "CatBoostModel",
    "FeaturePreparer",
    "LGBMModel",
    "cross_validate",
    "make_folds",
    "make_lgbm_ready",
    "make_model",
]
