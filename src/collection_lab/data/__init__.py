"""Подготовка данных: типы признаков, приведение dtype, сплиты train/val/test без утечек."""

from collection_lab.data.demo import (
    DEMO_DOMAINS,
    DEMO_PROCESSED_DT,
    make_demo_data,
    make_demo_transactions,
)
from collection_lab.data.split import DataSplit, random_split, time_split
from collection_lab.data.types import (
    cast_types,
    category_strings,
    detect_categorical,
    is_categorical,
    normalize_missing,
    split_feature_types,
    to_numeric_frame,
)

__all__ = [
    "DEMO_DOMAINS",
    "DEMO_PROCESSED_DT",
    "DataSplit",
    "cast_types",
    "category_strings",
    "detect_categorical",
    "is_categorical",
    "make_demo_data",
    "make_demo_transactions",
    "normalize_missing",
    "random_split",
    "split_feature_types",
    "time_split",
    "to_numeric_frame",
]
