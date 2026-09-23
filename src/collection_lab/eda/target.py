"""Динамика таргета по времени в разрезе сегментов."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from collection_lab.core.results import Result
from collection_lab.plotting.theme import color, style


def wilson_ci(k, n, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Доверительный интервал Уилсона для доли ``k / n``."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(center - margin, 0, 1), np.clip(center + margin, 0, 1)


def target_dynamics(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    segment_col: str | None = None,
    *,
    freq: str = "M",
    alpha: float = 0.05,
    segment_order: list | None = None,
    verbose: bool = True,
) -> Result:
    """Доля таргета по периодам и сегментам с доверительным интервалом Уилсона.

    Parameters
    ----------
    segment_col : str, optional
        Разрез (например, ``'sample'`` со значениями train/val/test). Без него — одна линия.
    freq : str
        Период агрегации: ``"D"``, ``"W"``, ``"M"``, ``"Q"``.

    Returns
    -------
    Result
        ``table``: ``period, segment, n_total, n_positive, target_rate, ci_low, ci_high``;
        ``plot()`` — линии с CI + объём по периодам.
    """
    cols = [date_col, target_col] + ([segment_col] if segment_col else [])
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Колонка {c!r} не найдена в df.")
    data = df[cols].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", format="mixed")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    n_bad = int(data[[date_col, target_col]].isna().any(axis=1).sum())
    if n_bad and verbose:
        print(f"[warn] {n_bad} строк с пустой датой или таргетом — исключены.")
    data = data.dropna(subset=[date_col, target_col])
    data["segment"] = data[segment_col].astype(object).fillna("NaN") if segment_col else "all"
    data["_period"] = data[date_col].dt.to_period(freq)

    agg = (data.groupby(["_period", "segment"])[target_col]
           .agg(n_total="count", n_positive="sum").reset_index())
    agg["target_rate"] = agg["n_positive"] / agg["n_total"]
    agg["ci_low"], agg["ci_high"] = wilson_ci(agg["n_positive"], agg["n_total"], alpha)
    agg["period"] = agg["_period"].dt.to_timestamp()
    agg = (agg[["period", "segment", "n_total", "n_positive", "target_rate", "ci_low", "ci_high"]]
           .sort_values(["segment", "period"]).reset_index(drop=True))
    info = {"freq": freq, "alpha": alpha, "segment_order": segment_order,
            "target": target_col}
    return Result("target_dynamics", agg, None, info, plotter=_plot_target_dynamics)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _plot_target_dynamics(result: Result) -> go.Figure:
    agg = result.table
    order = result.info.get("segment_order")
    present = list(agg["segment"].unique())
    segments = [s for s in (order or sorted(present, key=str)) if s in present]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.05)
    for i, seg in enumerate(segments):
        sub = agg[agg["segment"] == seg]
        c = color(i)
        fig.add_scatter(x=pd.concat([sub["period"], sub["period"][::-1]]),
                        y=pd.concat([sub["ci_high"], sub["ci_low"][::-1]]), fill="toself",
                        fillcolor=_hex_to_rgba(c, 0.15), line={"width": 0}, hoverinfo="skip",
                        showlegend=False, legendgroup=str(seg), row=1, col=1)
        fig.add_scatter(x=sub["period"], y=sub["target_rate"], mode="lines+markers",
                        name=str(seg), line={"color": c}, legendgroup=str(seg),
                        customdata=np.stack([sub["n_total"], sub["ci_low"], sub["ci_high"]], -1),
                        hovertemplate=("%{y:.4f} [%{customdata[1]:.4f}; %{customdata[2]:.4f}]"
                                       ", n=%{customdata[0]}"), row=1, col=1)
        fig.add_bar(x=sub["period"], y=sub["n_total"], name=str(seg), marker_color=c,
                    opacity=0.5, showlegend=False, legendgroup=str(seg), row=2, col=1)
    fig.update_yaxes(title_text="target rate", row=1, col=1)
    fig.update_yaxes(title_text="n", row=2, col=1)
    fig.update_layout(barmode="group")
    return style(fig, f"Динамика таргета ({result.info['freq']})", height=600)
