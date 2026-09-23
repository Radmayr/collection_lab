"""Метрики бинарной классификации и их расчёт по сегментам."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import log_loss, roc_auc_score


def roc_auc(y_true, y_pred) -> float:
    """ROC AUC; ``NaN``, если в ``y_true`` один класс (вместо исключения)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def gini(y_true, y_pred) -> float:
    """Коэффициент Джини: ``2 * AUC - 1``."""
    return 2 * roc_auc(y_true, y_pred) - 1


def ks(y_true, y_pred) -> float:
    """Статистика Колмогорова–Смирнова между скорами классов 1 и 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos, neg = y_pred[y_true == 1], y_pred[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(ks_2samp(pos, neg).statistic)


def logloss(y_true, y_pred) -> float:
    """Log loss (вход — вероятности класса 1)."""
    return float(log_loss(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15), labels=[0, 1]))


@dataclass(frozen=True)
class Metric:
    """Метрика и направление оптимизации."""

    name: str
    func: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool = True

    def __call__(self, y_true, y_pred) -> float:
        return self.func(y_true, y_pred)

    def delta(self, new: float, old: float) -> float:
        """Улучшение ``new`` относительно ``old`` (> 0 — стало лучше) для любого направления."""
        return (new - old) if self.greater_is_better else (old - new)


METRICS: dict[str, Metric] = {
    "auc": Metric("auc", roc_auc),
    "gini": Metric("gini", gini),
    "ks": Metric("ks", ks),
    "logloss": Metric("logloss", logloss, greater_is_better=False),
}


def get_metric(metric: str | Metric | Callable) -> Metric:
    """Возвращает :class:`Metric` по имени (``'auc'``, ``'gini'``, ``'ks'``, ``'logloss'``),
    объекту Metric или произвольной функции ``f(y_true, y_pred)`` (считается «больше — лучше»)."""
    if isinstance(metric, Metric):
        return metric
    if isinstance(metric, str):
        try:
            return METRICS[metric]
        except KeyError:
            raise ValueError(f"Неизвестная метрика {metric!r}. Доступны: {list(METRICS)}") from None
    if callable(metric):
        return Metric(getattr(metric, "__name__", "custom"), metric)
    raise TypeError(f"metric должен быть str, Metric или callable, получено {type(metric)}")


def metrics_by_segment(
    y_true,
    y_pred,
    segments,
    *,
    metrics: Sequence[str | Metric | Callable] = ("auc", "gini"),
    min_size: int = 0,
    add_total: bool = True,
) -> pd.DataFrame:
    """Метрики в разрезе сегментов.

    Parameters
    ----------
    segments : array-like
        Метка сегмента для каждого наблюдения (пропуски — в сегмент ``'NaN'``).
    min_size : int
        Сегменты меньше этого размера не выводятся.
    add_total : bool
        Добавить строку ``'ALL'`` по всей выборке.

    Returns
    -------
    pd.DataFrame
        Индекс — сегмент; колонки ``n, n_target, target_rate`` и по одной на метрику.
    """
    metric_objs = [get_metric(m) for m in metrics]
    df = pd.DataFrame({
        "y": np.asarray(y_true),
        "p": np.asarray(y_pred),
        "seg": pd.Series(np.asarray(segments, dtype=object)).fillna("NaN").to_numpy(),
    })

    def _row(part: pd.DataFrame) -> dict:
        row = {"n": len(part), "n_target": int(part["y"].sum()), "target_rate": part["y"].mean()}
        for m in metric_objs:
            row[m.name] = m(part["y"].to_numpy(), part["p"].to_numpy())
        return row

    rows = {seg: _row(part) for seg, part in df.groupby("seg", sort=True) if len(part) >= min_size}
    out = pd.DataFrame.from_dict(rows, orient="index")
    if add_total:
        out = pd.concat([pd.DataFrame({"ALL": _row(df)}).T, out])
    out.index.name = "segment"
    return out
