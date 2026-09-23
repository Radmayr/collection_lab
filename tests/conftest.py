"""Общие фикстуры.

Быстрые тесты работают на синтетике. Тесты с маркером ``slow`` используют реальные
выборки; путь к папке с ними задаётся переменной окружения ``COLLECTION_LAB_DATA``
(по умолчанию ``D:/ML_lib``). Если файлов нет, такие тесты пропускаются.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(os.environ.get("COLLECTION_LAB_DATA", "D:/ML_lib"))
FEATURES_CSV = DATA_DIR / "df_agg.csv"
TRANSACTIONS_CSV = DATA_DIR / "RTK_model" / "RTK_model" / "data" / "df_1.csv"


@pytest.fixture(scope="session")
def binary_df() -> pd.DataFrame:
    """Синтетическая бинарная выборка: информативные, шумовые, коррелированные,
    категориальные, константные и пустые признаки, дата и сегмент."""
    rng = np.random.default_rng(42)
    n = 4000

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c", "d"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    cat_effect = pd.Series(cat).map({"a": -0.5, "b": 0.0, "c": 0.5, "d": 1.0}).to_numpy()
    logit = -1.5 + 1.0 * x1 + 0.6 * x2 + cat_effect
    target = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x1_copy": x1 * 2 + rng.normal(scale=0.05, size=n),  # почти дубль x1
        "noise": rng.normal(size=n),
        "cat": cat,
        "const": 1.0,
        "all_nan": np.nan,
        "segment": rng.choice(["s1", "s2"], size=n),
        "report_date": pd.to_datetime("2023-01-01")
        + pd.to_timedelta(rng.integers(0, 540, size=n), unit="D"),
        "target": target,
    })
    df.loc[rng.random(n) < 0.1, "x2"] = np.nan
    return df


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    """Реальная выборка признаков (df_agg.csv)."""
    if not FEATURES_CSV.exists():
        pytest.skip(f"нет файла {FEATURES_CSV}")
    return pd.read_csv(FEATURES_CSV, low_memory=False)


@pytest.fixture(scope="session")
def transactions_df() -> pd.DataFrame:
    """Реальная выборка транзакций (df_1.csv) — для EDA и винтажей."""
    if not TRANSACTIONS_CSV.exists():
        pytest.skip(f"нет файла {TRANSACTIONS_CSV}")
    return pd.read_csv(TRANSACTIONS_CSV, low_memory=False)
