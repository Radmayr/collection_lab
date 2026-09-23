"""Обзор таблицы: типы, пропуски, уникальные значения, базовые статистики."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from collection_lab.data.types import is_categorical


def overview(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Сводка по колонкам.

    Returns
    -------
    pd.DataFrame
        Индекс — колонка; ``dtype, kind, n_missing, missing_share, n_unique, zero_share,
        top_value, top_share, min, median, mean, max``. ``kind`` — numeric / categorical /
        datetime / constant / empty.
    """
    columns = list(df.columns if columns is None else columns)
    n = len(df)
    rows = {}
    for c in columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        vc = s.value_counts(dropna=True)
        if n_missing == n:
            kind = "empty"
        elif n_unique <= 1:
            kind = "constant"
        elif is_datetime64_any_dtype(s):
            kind = "datetime"
        elif is_categorical(s):
            kind = "categorical"
        else:
            kind = "numeric"
        row = {
            "dtype": str(s.dtype), "kind": kind, "n_missing": n_missing,
            "missing_share": n_missing / n if n else np.nan, "n_unique": n_unique,
            "zero_share": np.nan,
            "top_value": vc.index[0] if len(vc) else None,
            "top_share": vc.iloc[0] / n if len(vc) and n else np.nan,
            "min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan,
        }
        if kind in ("numeric", "constant") and not is_datetime64_any_dtype(s):
            num = s if is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
            if num.notna().any():
                row.update(zero_share=float((num == 0).mean()), min=num.min(),
                           median=num.median(), mean=num.mean(), max=num.max())
        elif kind == "datetime":
            row.update(min=s.min(), max=s.max())
        rows[c] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "column"
    return out


def target_summary(
    df: pd.DataFrame, target: str, by: str | list[str] | None = None
) -> pd.DataFrame:
    """Размер выборки, число и доля таргета — всего или в разрезе ``by``."""
    if by is None:
        return pd.DataFrame({"n": [len(df)], "n_target": [df[target].sum()],
                             "target_rate": [df[target].mean()]})
    return (df.groupby(by, dropna=False)[target]
            .agg(n="size", n_target="sum", target_rate="mean").reset_index())
