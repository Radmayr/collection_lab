"""Общий стиль plotly-графиков и компоновка нескольких графиков в сетку."""

from __future__ import annotations

from collections.abc import Sequence

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_white"
PALETTE = px.colors.qualitative.Plotly
POSITIVE = "#2ca02c"
NEGATIVE = "#d62728"
NEUTRAL = "#7f7f7f"


def color(i: int) -> str:
    """i-й цвет палитры (по кругу)."""
    return PALETTE[i % len(PALETTE)]


def style(
    fig: go.Figure,
    title: str | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Применяет общий стиль библиотеки и возвращает ту же фигуру."""
    fig.update_layout(
        template=TEMPLATE,
        title={"text": title, "x": 0.5} if title else None,
        height=height,
        width=width,
        margin={"l": 60, "r": 30, "t": 70 if title else 40, "b": 50},
        hovermode="x unified",
        legend={"orientation": "v"},
    )
    if xaxis_title is not None:
        fig.update_xaxes(title_text=xaxis_title)
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title)
    return fig


def combine(
    figures: Sequence[go.Figure],
    titles: Sequence[str] | None = None,
    *,
    n_cols: int = 2,
    size: int = 450,
    showlegend: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Собирает несколько фигур в одну сетку (замена ``plot_in_line``).

    Parameters
    ----------
    figures : list[go.Figure]
    titles : list[str], optional
        Заголовки подграфиков; по умолчанию берутся заголовки исходных фигур.
    n_cols : int
        Число графиков в строке.
    size : int
        Ширина и высота одного графика в пикселях.
    """
    figures = list(figures)
    if titles is None:
        titles = [(f.layout.title.text or "") for f in figures]
    n_rows = max(1, -(-len(figures) // n_cols))
    out = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(titles),
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for i, fig in enumerate(figures):
        row, col = i // n_cols + 1, i % n_cols + 1
        for trace in fig.data:
            out.add_trace(trace, row=row, col=col)
    style(out, title, height=size * n_rows, width=size * n_cols)
    out.update_layout(showlegend=showlegend, hovermode="closest")
    return out


def bar_colors(values) -> list[str]:
    """Зелёный для неотрицательных, красный для отрицательных значений."""
    return [NEGATIVE if (v is not None and v < 0) else POSITIVE for v in values]
