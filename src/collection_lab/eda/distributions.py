"""Распределения признаков."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from collection_lab.plotting.theme import NEGATIVE, style


def plot_distribution(
    data: pd.DataFrame | pd.Series,
    col: str | None = None,
    *,
    clip_low: float = 0.01,
    clip_high: float = 0.99,
    force_log: bool | None = None,
    bins: int = 100,
    title: str | None = None,
) -> go.Figure:
    """Гистограмма с обрезкой выбросов по квантилям и авто-лог-шкалой
    (бывший ``plot_distribution_with_clip``).

    Parameters
    ----------
    data : DataFrame или Series
        Если DataFrame — нужен ``col``.
    clip_low, clip_high : float
        Квантили обрезки.
    force_log : bool, optional
        ``None`` — лог-шкала, если skew > 2 и все значения > 0.

    Returns
    -------
    go.Figure
        В заголовке — mean / median / std / skew обрезанных данных.
    """
    s = data[col] if isinstance(data, pd.DataFrame) else data
    name = col or s.name or "value"
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return style(go.Figure(), f"Нет данных в {name!r}")
    if force_log is None:
        force_log = bool(s.skew() > 2.0 and (s > 0).all())
    if force_log:
        positive = s[s > 0]
        if positive.empty:
            force_log = False
        else:
            s = np.log10(positive)
    lo, hi = s.quantile(clip_low), s.quantile(clip_high)
    clipped = s.clip(lo, hi)
    stats = (f"μ={clipped.mean():.3f} | медиана={clipped.median():.3f} | "
             f"σ={clipped.std():.3f} | skew={clipped.skew():.2f}")
    fig = go.Figure(go.Histogram(x=clipped, nbinsx=bins, marker_line={"width": 0.5,
                                                                       "color": "white"}))
    if not force_log:
        for v in (lo, hi):
            fig.add_vline(x=v, line_dash="dot", line_color=NEGATIVE)
    x_title = f"log10({name})" if force_log else name
    title = title or (f"Распределение {name} (clip {clip_low * 100:g}–{clip_high * 100:g}%)"
                      f"<br><sup>{stats}</sup>")
    fig.update_layout(bargap=0.02)
    return style(fig, title, height=450, xaxis_title=x_title, yaxis_title="частота")


def plot_target_rate_by_bins(
    df: pd.DataFrame, col: str, target: str, *, bins: int = 10, title: str | None = None
) -> go.Figure:
    """Доля таргета по квантильным бинам признака (+ число наблюдений в бине)."""
    x = pd.to_numeric(df[col], errors="coerce")
    binned = pd.qcut(x, bins, duplicates="drop").astype(str).where(x.notna(), "NaN")
    g = df.assign(_bin=binned).groupby("_bin", sort=False)[target].agg(["mean", "size"])
    order = sorted((b for b in g.index if b != "NaN"),
                   key=lambda b: float(b.split(",")[0].strip("(["))) + (
        ["NaN"] if "NaN" in g.index else [])
    g = g.loc[order]
    fig = go.Figure()
    fig.add_bar(x=g.index, y=g["size"], name="n", yaxis="y2", opacity=0.3)
    fig.add_scatter(x=g.index, y=g["mean"], name="target rate", mode="lines+markers")
    fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "showgrid": False})
    return style(fig, title or f"Доля таргета по бинам {col}", height=420,
                 yaxis_title="target rate")
