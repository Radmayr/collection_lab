"""Определение и приведение типов признаков."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

# Строки, которые выгрузки из БД используют вместо пропуска.
MISSING_TOKENS = ("nan", "NaN", "None", "null", "NULL", "")


def normalize_missing(s: pd.Series) -> pd.Series:
    """Заменяет строковые обозначения пропуска на NaN (только для object/string колонок)."""
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return s
    return s.mask(s.isin(MISSING_TOKENS))


def is_categorical(s: pd.Series) -> bool:
    """Категориальный признак — тот, что нельзя целиком привести к числу.

    Пропуски (в том числе строки ``'nan'``, ``'None'``, ``'null'``, ``''``) не учитываются.
    Колонки с dtype ``category`` считаются категориальными всегда.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return False
    if is_datetime64_any_dtype(s) or is_timedelta64_dtype(s):
        return False
    s = normalize_missing(s)
    notna = s.notna()
    converted = pd.to_numeric(s[notna], errors="coerce")
    return bool(converted.isna().any())


def detect_categorical(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    """Возвращает список категориальных колонок ``df`` (см. :func:`is_categorical`)."""
    exclude = set(exclude or ())
    return [c for c in df.columns if c not in exclude and is_categorical(df[c])]


def split_feature_types(
    df: pd.DataFrame, features: Iterable[str] | None = None
) -> tuple[list[str], list[str]]:
    """Делит признаки на числовые и категориальные: ``(num_cols, cat_cols)``."""
    features = list(df.columns if features is None else features)
    cat_cols = detect_categorical(df[features])
    num_cols = [c for c in features if c not in cat_cols]
    return num_cols, cat_cols


def cast_types(
    df: pd.DataFrame,
    num_cols: Iterable[str],
    cat_cols: Iterable[str],
    target_col: str | None = None,
) -> pd.DataFrame:
    """Приводит типы: числовые → ``float64``, категориальные → ``object`` с NaN вместо
    строковых пропусков, таргет → ``int``. Возвращает копию.
    """
    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(normalize_missing(out[c]), errors="coerce").astype("float64")
    for c in cat_cols:
        s = normalize_missing(out[c])
        out[c] = s.where(s.isna(), s.astype(str)).astype(object)
    if target_col is not None:
        out[target_col] = out[target_col].astype(int)
    return out


def to_numeric_frame(X: pd.DataFrame, cat_cols: Iterable[str] = ()) -> pd.DataFrame:
    """Приводит не категориальные колонки к float: даты и интервалы — в наносекунды,
    bool — в 0/1, прочее — через ``pd.to_numeric``. Категориальные не трогает."""
    cat_cols = set(cat_cols)
    out = X.copy()
    for c in out.columns:
        if c in cat_cols:
            continue
        s = out[c]
        if is_datetime64_any_dtype(s) or is_timedelta64_dtype(s):
            if is_datetime64_any_dtype(s):
                if getattr(s.dt, "tz", None) is not None:
                    s = s.dt.tz_localize(None)
                raw = s.to_numpy(dtype="datetime64[ns]")
            else:
                raw = s.to_numpy(dtype="timedelta64[ns]")
            values = raw.view("int64").astype("float64")
            values[s.isna().to_numpy()] = np.nan
            out[c] = values
        elif is_bool_dtype(s):
            out[c] = s.astype("float64")
        elif is_numeric_dtype(s):
            out[c] = s.astype("float64")
        else:
            out[c] = pd.to_numeric(normalize_missing(s), errors="coerce").astype("float64")
    return out
