"""Фильтры качества признаков: пропуски, нули, константы, кардинальность."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go

from collection_lab.core.results import Result
from collection_lab.data.types import detect_categorical
from collection_lab.plotting.theme import style


def quality_filter(
    df: pd.DataFrame,
    features: Iterable[str] | None = None,
    *,
    cat_features: Iterable[str] | None = None,
    num_missing_threshold: float = 0.95,
    num_zero_threshold: float = 0.95,
    num_zero_nan_threshold: float = 0.95,
    cat_unique_min: int = 1,
    cat_unique_max: int = 100,
    cat_missing_threshold: float = 0.95,
) -> Result:
    """Удаляет неинформативные признаки (бывший ``missing_exclusion``).

    Числовой признак удаляется, если доля пропусков, нулей или «нулей + пропусков»
    **больше** порога либо std = 0. Категориальный — если уникальных значений
    ``<= cat_unique_min`` или ``> cat_unique_max`` либо пропусков больше порога.
    В лог пишется первая сработавшая причина.

    Parameters
    ----------
    features : list[str], optional
        Проверяемые признаки (по умолчанию все колонки ``df``).
    cat_features : list[str], optional
        Категориальные признаки (по умолчанию определяются автоматически).

    Returns
    -------
    Result
        ``selected`` — оставшиеся признаки; ``table`` — по строке на признак:
        ``feature, type, action, reason, missing_share, zero_share, n_unique``.
    """
    features = list(df.columns if features is None else features)
    cat_set = set(detect_categorical(df[features]) if cat_features is None else cat_features)

    rows = []
    for col in features:
        s = df[col]
        missing = float(s.isna().mean())
        n_unique = int(s.nunique(dropna=True))
        reason = None
        if col in cat_set:
            kind, zero = "categorical", float("nan")
            if n_unique <= cat_unique_min:
                reason = f"Уникальных значений <= {cat_unique_min}"
            elif n_unique > cat_unique_max:
                reason = f"Уникальных значений > {cat_unique_max}"
            elif missing > cat_missing_threshold:
                reason = f"Пропущено > {cat_missing_threshold:.0%}"
        else:
            kind = "numeric"
            s = pd.to_numeric(s, errors="coerce")
            zero = float((s == 0).mean())
            if missing > num_missing_threshold:
                reason = f"Пропущено > {num_missing_threshold:.0%}"
            elif zero > num_zero_threshold:
                reason = f"Нулей > {num_zero_threshold:.0%}"
            elif ((s == 0) | s.isna()).mean() > num_zero_nan_threshold:
                reason = f"Нули+пропуски > {num_zero_nan_threshold:.0%}"
            elif not (s.std() > 0):  # std == 0 или NaN (все значения пропущены/одинаковы)
                reason = "Стандартное отклонение = 0"
        rows.append({
            "feature": col, "type": kind, "action": "drop" if reason else "keep",
            "reason": reason or "", "missing_share": missing, "zero_share": zero,
            "n_unique": n_unique,
        })

    table = pd.DataFrame(rows)
    selected = table.loc[table["action"] == "keep", "feature"].tolist()
    info = {"n_input": len(features), "n_selected": len(selected),
            "n_dropped": len(features) - len(selected)}
    return Result("quality_filter", table, selected, info, plotter=_plot_reasons)


def _plot_reasons(result: Result) -> go.Figure:
    dropped = result.table[result.table["action"] == "drop"]
    counts = dropped.groupby("reason").size().sort_values()
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h"))
    return style(fig, f"Удалено признаков: {len(dropped)} из {len(result.table)}",
                 height=max(300, 60 * len(counts) + 120), xaxis_title="число признаков")
