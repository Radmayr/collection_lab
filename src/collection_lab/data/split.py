"""Разбиение выборки на train / val / test без утечек.

Один объект :class:`DataSplit` передаётся через весь пайплайн: отбор признаков и подбор
параметров идут на ``train``/``val``, ``test`` используется только для отчёта.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from collection_lab.config import RANDOM_STATE


@dataclass
class DataSplit:
    """Непересекающиеся части выборки.

    Attributes
    ----------
    train, val : pd.DataFrame
        Обучение и валидация (early stopping, выбор параметров).
    test : pd.DataFrame | None
        Отложенная выборка (обычно out-of-time) — только для итоговой оценки.
    target : str
        Имя колонки таргета.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame | None
    target: str

    def __post_init__(self) -> None:
        parts = [("train", self.train), ("val", self.val)]
        if self.test is not None:
            parts.append(("test", self.test))
        for i, (name_a, a) in enumerate(parts):
            for name_b, b in parts[i + 1:]:
                overlap = a.index.intersection(b.index)
                if len(overlap):
                    raise ValueError(
                        f"{name_a} и {name_b} пересекаются по индексу ({len(overlap)} строк). "
                        "Проверьте, что индекс исходного DataFrame уникален."
                    )

    def items(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Итерирует по непустым частям: ``('train', df), ('val', df), ('test', df)``."""
        yield "train", self.train
        yield "val", self.val
        if self.test is not None:
            yield "test", self.test

    def xy(self, part: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        """Возвращает ``(X, y)`` для части ``'train' | 'val' | 'test'``."""
        df = dict(self.items())[part]
        return df[features], df[self.target]

    def summary(self, date_col: str | None = None) -> pd.DataFrame:
        """Размер, число и доля таргета по частям (+ диапазон дат, если задан ``date_col``)."""
        rows = []
        for name, df in self.items():
            row = {
                "part": name,
                "n": len(df),
                "n_target": int(df[self.target].sum()),
                "target_rate": float(df[self.target].mean()) if len(df) else float("nan"),
            }
            if date_col is not None:
                row["date_min"] = df[date_col].min()
                row["date_max"] = df[date_col].max()
            rows.append(row)
        return pd.DataFrame(rows).set_index("part")

    def labeled(self, col: str = "sample") -> pd.DataFrame:
        """Склеивает части в один DataFrame с колонкой-меткой части."""
        return pd.concat([df.assign(**{col: name}) for name, df in self.items()])


def _check_index(df: pd.DataFrame) -> None:
    if not df.index.is_unique:
        raise ValueError("Индекс df не уникален — сделайте df.reset_index(drop=True).")


def random_split(
    df: pd.DataFrame,
    target: str,
    *,
    val_size: float = 0.2,
    test_size: float = 0.0,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """Случайное разбиение на train / val / (test), по умолчанию стратифицированное."""
    _check_index(df)
    test = None
    rest = df
    if test_size > 0:
        rest, test = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df[target] if stratify else None,
        )
    train, val = train_test_split(
        rest, test_size=val_size / (1 - test_size), random_state=random_state,
        stratify=rest[target] if stratify else None,
    )
    return DataSplit(train=train, val=val, test=test, target=target)


def time_split(
    df: pd.DataFrame,
    target: str,
    date_col: str,
    oot_from,
    *,
    oot_to=None,
    val_size: float = 0.2,
    val_mode: str = "random",
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """Out-of-time разбиение.

    ``test`` — строки с ``date_col >= oot_from`` (и ``< oot_to``, если задано);
    остальное делится на train / val.

    Parameters
    ----------
    val_mode : {"random", "time"}
        ``"random"`` — случайная доля ``val_size`` из периода до ``oot_from``;
        ``"time"`` — последние по дате ``val_size`` строк этого периода.
    """
    _check_index(df)
    if val_mode not in ("random", "time"):
        raise ValueError("val_mode должен быть 'random' или 'time'")

    dates = pd.to_datetime(df[date_col])
    oot_from = pd.Timestamp(oot_from)
    is_test = dates >= oot_from
    if oot_to is not None:
        is_test &= dates < pd.Timestamp(oot_to)
    test = df[is_test]
    dev = df[dates < oot_from]
    if dev.empty:
        raise ValueError(f"Нет строк раньше {oot_from.date()} — нечего отдать в train/val.")

    if val_mode == "random":
        train, val = train_test_split(
            dev, test_size=val_size, random_state=random_state,
            stratify=dev[target] if stratify else None,
        )
    else:
        order = pd.to_datetime(dev[date_col]).sort_values(kind="stable").index
        n_val = int(round(len(dev) * val_size))
        train, val = dev.loc[order[:-n_val]], dev.loc[order[-n_val:]]

    return DataSplit(train=train, val=val, test=test if len(test) else None, target=target)
