"""Индекс стабильности популяции (PSI) и производные: CSI по признакам, PSI по периодам.

Формула: ``PSI = Σ (a_i - e_i) · ln(a_i / e_i)``, где ``e_i``, ``a_i`` — доли бина в
ожидаемой (обычно train) и фактической выборках. Бин, которого нет в одной из выборок,
получает в ней count = 1 (как в исходной версии ``scr/metric_tools.psi``), чтобы логарифм
оставался конечным. Пропуски образуют отдельный бин ``'NaN'``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PSI_THRESHOLDS = (0.1, 0.25)


def psi_label(value: float) -> str:
    """Интерпретация PSI: ``< 0.1`` — stable, ``0.1–0.25`` — moderate, ``> 0.25`` — significant."""
    if np.isnan(value):
        return "n/a"
    if value < PSI_THRESHOLDS[0]:
        return "stable"
    if value < PSI_THRESHOLDS[1]:
        return "moderate"
    return "significant"


def quantile_edges(x, bins: int = 10) -> np.ndarray:
    """Границы квантильных бинов по ``x``: ``[-inf, q_1, ..., q_{bins-1}, inf]`` без дублей,
    отсортированные по возрастанию."""
    x = pd.Series(x, dtype="float64").dropna()
    inner = x.quantile([i / bins for i in range(1, bins)]).to_numpy() if len(x) else []
    return np.unique(np.concatenate([[-np.inf], inner, [np.inf]]))


def _format_edge(v: float) -> str:
    if v == -np.inf:
        return "-inf"
    if v == np.inf:
        return "inf"
    return f"{v:.4g}"


def bin_labels(edges: np.ndarray) -> list[str]:
    """Подписи бинов вида ``'01: (-inf, 3.5]'`` в порядке возрастания границ."""
    return [
        f"{i + 1:02d}: ({_format_edge(lo)}, {_format_edge(hi)}]"
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]


def assign_bins(x, edges: np.ndarray) -> pd.Series:
    """Бин для каждого значения числового ``x`` по готовым границам; NaN → ``'NaN'``."""
    x = pd.Series(x, dtype="float64").reset_index(drop=True)
    labels = bin_labels(edges)
    binned = pd.cut(x, bins=edges, labels=labels)
    return binned.astype(object).where(x.notna(), "NaN")


def _as_categories(x) -> pd.Series:
    s = pd.Series(x).reset_index(drop=True)
    return s.astype(object).where(s.notna(), "NaN").astype(str)


def _is_categorical(expected, actual, categorical: bool | None) -> bool:
    if categorical is not None:
        return categorical
    return not (is_numeric_dtype(pd.Series(expected)) and is_numeric_dtype(pd.Series(actual)))


def psi_table(expected, actual, bins: int = 10, categorical: bool | None = None) -> pd.DataFrame:
    """Разбивка PSI по бинам.

    Parameters
    ----------
    expected, actual : array-like
        Значения признака (или скора) в базовой и сравниваемой выборках.
    bins : int
        Число квантильных бинов для числового признака (границы — по ``expected``).
    categorical : bool, optional
        ``True`` — каждое значение как отдельный бин; по умолчанию определяется по dtype.

    Returns
    -------
    pd.DataFrame
        ``bin, expected_count, actual_count, expected_share, actual_share, psi``.
    """
    if _is_categorical(expected, actual, categorical):
        e_bins, a_bins = _as_categories(expected), _as_categories(actual)
    else:
        edges = quantile_edges(expected, bins)
        e_bins, a_bins = assign_bins(expected, edges), assign_bins(actual, edges)

    e_counts = e_bins.value_counts()
    a_counts = a_bins.value_counts()
    table = pd.concat([e_counts.rename("expected_count"), a_counts.rename("actual_count")], axis=1)
    table = table.fillna(1).sort_index()
    table["expected_share"] = table["expected_count"] / e_counts.sum()
    table["actual_share"] = table["actual_count"] / a_counts.sum()
    table["psi"] = (table["actual_share"] - table["expected_share"]) * np.log(
        table["actual_share"] / table["expected_share"]
    )
    table.index.name = "bin"
    return table.reset_index()


def psi(expected, actual, bins: int = 10, categorical: bool | None = None) -> float:
    """PSI между двумя выборками (см. :func:`psi_table`)."""
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    return float(psi_table(expected, actual, bins=bins, categorical=categorical)["psi"].sum())


def feature_psi(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    features: Iterable[str] | None = None,
    *,
    bins: int = 10,
    cat_features: Iterable[str] = (),
) -> pd.DataFrame:
    """PSI каждого признака (CSI): ``feature, psi, status``, по убыванию PSI."""
    features = list(expected.columns if features is None else features)
    cat_features = set(cat_features)
    rows = []
    for f in features:
        value = psi(expected[f], actual[f], bins=bins,
                    categorical=True if f in cat_features else None)
        rows.append({"feature": f, "psi": value, "status": psi_label(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def psi_by_period(
    df: pd.DataFrame,
    column: str,
    date_col: str,
    *,
    freq: str = "M",
    mode: str = "reference",
    reference: pd.Series | None = None,
    bins: int = 10,
    categorical: bool | None = None,
) -> pd.DataFrame:
    """PSI признака по периодам.

    Parameters
    ----------
    mode : {"reference", "adjacent"}
        ``"reference"`` — каждый период против базы (``reference`` или первого периода);
        ``"adjacent"`` — каждый период против предыдущего.
    reference : pd.Series, optional
        Значения базовой выборки (например, признак на train). По умолчанию — первый период.

    Returns
    -------
    pd.DataFrame
        ``period, n, psi, status``.
    """
    if mode not in ("reference", "adjacent"):
        raise ValueError("mode должен быть 'reference' или 'adjacent'")
    periods = pd.to_datetime(df[date_col]).dt.to_period(freq)
    groups = [(p, df.loc[periods == p, column]) for p in sorted(periods.dropna().unique())]
    rows = []
    for i, (period, values) in enumerate(groups):
        if mode == "adjacent":
            base = groups[i - 1][1] if i > 0 else None
        else:
            base = reference if reference is not None else groups[0][1]
        value = np.nan if base is None else psi(base, values, bins=bins, categorical=categorical)
        rows.append({"period": period.to_timestamp(), "n": len(values), "psi": value,
                     "status": psi_label(value)})
    return pd.DataFrame(rows)
