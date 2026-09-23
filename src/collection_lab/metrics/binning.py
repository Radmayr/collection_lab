"""Разбиение признака на бакеты, WoE и Information Value."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from collection_lab.metrics.classification import roc_auc

WOE_EPS = 1e-5  # замена нулевого числа «плохих»/«хороших» в бакете (avoid_null_mode)


def quantile_buckets(values, n_buckets: int) -> np.ndarray:
    """Квантильное разбиение, не разрывающее одинаковые значения.

    Жадно идёт по отсортированным уникальным значениям: бакет закрывается, как только в
    нём набирается ``остаток / число_оставшихся_бакетов`` наблюдений. Значение, которого
    больше этой доли, образует отдельный бакет — так у признака с 86% нулей получается
    ``[0]``, ``[1]``, ``[2]``, ... вместо схлопывания квантилей.

    Returns
    -------
    np.ndarray
        Номер бакета (0, 1, ...; по возрастанию значений) для каждого наблюдения; NaN → -1.
    """
    x = pd.Series(np.asarray(values, dtype="float64"))
    out = np.full(len(x), -1, dtype=int)
    counts = x.value_counts().sort_index()
    uniq, cnt = counts.index.to_numpy(), counts.to_numpy()
    mapping: dict[float, int] = {}
    start, bucket, remaining, k = 0, 0, cnt.sum(), n_buckets
    while start < len(uniq):
        if k <= 1:
            end = len(uniq)
        else:
            target = remaining / k
            cum = np.cumsum(cnt[start:])
            end = start + int(np.searchsorted(cum, target)) + 1
        for v in uniq[start:end]:
            mapping[v] = bucket
        remaining -= cnt[start:end].sum()
        start, bucket, k = end, bucket + 1, k - 1
    notna = x.notna().to_numpy()
    out[notna] = x[notna].map(mapping).to_numpy()
    return out


def uniform_buckets(values, n_buckets: int) -> np.ndarray:
    """Равные по ширине интервалы между min и max (``bins_method="dense"``); NaN → -1."""
    x = np.asarray(values, dtype="float64")
    out = np.full(len(x), -1, dtype=int)
    mask = ~np.isnan(x)
    if mask.any():
        lo, hi = x[mask].min(), x[mask].max()
        if hi == lo:
            out[mask] = 0
        else:
            edges = np.linspace(lo, hi, n_buckets + 1)
            out[mask] = np.clip(np.searchsorted(edges, x[mask], side="right") - 1, 0,
                                n_buckets - 1)
            # перенумеровать без пустых бакетов
            _, out[mask] = np.unique(out[mask], return_inverse=True)
    return out


def bucket_labels(values, buckets: np.ndarray, name: str = "var") -> dict[int, str]:
    """Подписи бакетов вида ``'var in [min, max]'`` по фактическим значениям."""
    s = pd.Series(np.asarray(values))
    labels = {}
    for b in sorted(set(buckets[buckets >= 0].tolist())):
        part = s[buckets == b]
        lo, hi = part.min(), part.max()
        labels[b] = f"{name} in [{_fmt(lo)}, {_fmt(hi)}]"
    if (buckets < 0).any():
        labels[-1] = f"{name} is null"
    return labels


def _fmt(v) -> str:
    v = v.item() if hasattr(v, "item") else v
    return str(round(v, 6)) if isinstance(v, float) else str(v)


def woe_iv_table(values, target, buckets: np.ndarray, eps: float = WOE_EPS) -> pd.DataFrame:
    """WoE и вклад в IV по бакетам.

    ``WoE = ln(доля «плохих» бакета / доля «хороших» бакета)`` — положительный WoE у бакетов
    с повышенной долей таргета. Нулевые счётчики заменяются на ``eps``.

    Returns
    -------
    pd.DataFrame
        ``bucket, n, bad, good, bad_rate, bad_share, good_share, woe, iv``.
    """
    df = pd.DataFrame({"b": buckets, "y": np.asarray(target, dtype=float)})
    g = df.groupby("b")["y"].agg(n="size", bad="sum")
    g["good"] = g["n"] - g["bad"]
    g["bad_rate"] = g["bad"] / g["n"]
    bad = g["bad"].where(g["bad"] > 0, eps)
    good = g["good"].where(g["good"] > 0, eps)
    g["bad_share"] = bad / g["bad"].sum() if g["bad"].sum() > 0 else np.nan
    g["good_share"] = good / g["good"].sum() if g["good"].sum() > 0 else np.nan
    g["woe"] = np.log(g["bad_share"] / g["good_share"])
    g["iv"] = (g["bad_share"] - g["good_share"]) * g["woe"]
    return g.reset_index().rename(columns={"b": "bucket"})


def woe_ci(bad, good, z: float = norm.ppf(0.995)) -> np.ndarray:
    """Полуширина доверительного интервала WoE (дельта-метод): ``z·sqrt(1/bad + 1/good)``."""
    bad = np.maximum(np.asarray(bad, dtype=float), WOE_EPS)
    good = np.maximum(np.asarray(good, dtype=float), WOE_EPS)
    return z * np.sqrt(1 / bad + 1 / good)


def iv_from_auc(auc: float) -> float:
    """IV через ROC AUC (бинормальная модель с равными дисперсиями):
    ``IV = 2 · Φ⁻¹(AUC)²``."""
    if np.isnan(auc):
        return np.nan
    auc = min(max(auc, 1 - auc), 1 - 1e-12)
    return float(2 * norm.ppf(auc) ** 2)


def information_value(
    values, target, n_buckets: int = 10, *, method: str = "IV_empirical",
    bins_method: str = "quantile",
) -> float:
    """Information Value признака (NaN — отдельный бакет)."""
    values = np.asarray(values, dtype="float64")
    target = np.asarray(target)
    if method == "IV_auc":
        mask = ~np.isnan(values)
        return iv_from_auc(roc_auc(target[mask], values[mask]))
    buckets = (quantile_buckets if bins_method == "quantile" else uniform_buckets)(
        values, n_buckets)
    return float(woe_iv_table(values, target, buckets)["iv"].sum())


def iv_table(
    df: pd.DataFrame, target: str, features: Iterable[str] | None = None, *,
    n_buckets: int = 10, method: str = "IV_empirical",
) -> pd.DataFrame:
    """IV числовых признаков по убыванию: ``feature, iv, strength``.

    Шкала силы: < 0.02 — нет, 0.02–0.1 — слабая, 0.1–0.3 — средняя, > 0.3 — сильная.
    """
    features = [c for c in (df.columns if features is None else features) if c != target]
    rows = []
    for f in features:
        x = pd.to_numeric(df[f], errors="coerce")
        value = information_value(x, df[target], n_buckets, method=method)
        strength = ("none" if value < 0.02 else "weak" if value < 0.1
                    else "medium" if value < 0.3 else "strong")
        rows.append({"feature": f, "iv": value, "strength": strength})
    return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)
