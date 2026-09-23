"""Метрики модели и признаков в динамике по периодам и сегментам."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collection_lab.config import LGBM_UNIVARIATE_PARAMS, merge_params
from collection_lab.core.models import make_model
from collection_lab.core.results import Result
from collection_lab.data.types import is_categorical
from collection_lab.metrics.classification import get_metric
from collection_lab.plotting.theme import color, style


def _periods(dates: pd.Series, freq: str) -> pd.Series:
    return pd.to_datetime(dates, format="mixed").dt.to_period(freq).dt.to_timestamp()


def metric_dynamics(
    df: pd.DataFrame,
    target: str,
    score: str,
    date_col: str,
    segment: str | None = None,
    *,
    freq: str = "M",
    metric: str = "auc",
    min_size: int = 30,
) -> Result:
    """Метрика скора по периодам (и сегментам) — бывший ``dynamic_auc``.

    Returns
    -------
    Result
        ``table``: ``period, segment, n, target_rate, <metric>``; ``plot()`` — линии по
        сегментам. Периоды с одним классом или размером < ``min_size`` дают NaN.
    """
    m = get_metric(metric)
    data = df[[target, score, date_col] + ([segment] if segment else [])].copy()
    data["period"] = _periods(data[date_col], freq)
    data["segment"] = data[segment].astype(object).fillna("NaN") if segment else "all"
    rows = []
    for (p, s), g in data.groupby(["period", "segment"], sort=True):
        value = m(g[target], g[score]) if len(g) >= min_size else np.nan
        rows.append({"period": p, "segment": s, "n": len(g), "target_rate": g[target].mean(),
                     m.name: value})
    info = {"metric": m.name, "freq": freq, "score": score}
    return Result("metric_dynamics", pd.DataFrame(rows), None, info,
                  plotter=_plot_metric_dynamics)


def _plot_metric_dynamics(result: Result, title: str | None = None) -> go.Figure:
    t, name = result.table, result.info["metric"]
    fig = go.Figure()
    for i, (seg, g) in enumerate(t.groupby("segment", sort=True)):
        fig.add_scatter(x=g["period"], y=g[name], mode="lines+markers", name=str(seg),
                        line={"color": color(i)}, customdata=g["n"],
                        hovertemplate="%{y:.4f} (n=%{customdata})")
    return style(fig, title or f"{name.upper()} по времени ({result.info['score']})",
                 height=450, yaxis_title=name)


def feature_metric_dynamics(
    train: pd.DataFrame,
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    date_col: str,
    *,
    freq: str = "M",
    metric: str = "auc",
    params: dict[str, Any] | None = None,
    min_size: int = 30,
) -> Result:
    """Однофакторная метрика признаков по периодам: для каждого признака маленькая модель
    обучается на ``train`` и оценивается в каждом периоде ``data`` (из ноутбука RTK).

    Returns
    -------
    Result
        ``table``: ``feature, period, n, <metric>``; ``plot()`` — сетка графиков.
    """
    m = get_metric(metric)
    params = merge_params(LGBM_UNIVARIATE_PARAMS, {"n_estimators": 150, **(params or {})})
    periods = _periods(data[date_col], freq)
    rows = []
    for f in features:
        model = make_model("lgbm", params, early_stopping_rounds=None)
        model.fit(train[[f]], train[target], cat_features=[f] if is_categorical(train[f]) else [])
        pred = pd.Series(model.predict(data[[f]]), index=data.index)
        for p in sorted(periods.dropna().unique()):
            mask = periods == p
            y = data.loc[mask, target]
            value = m(y, pred[mask]) if mask.sum() >= min_size else np.nan
            rows.append({"feature": f, "period": p, "n": int(mask.sum()), m.name: value})
    return Result("feature_metric_dynamics", pd.DataFrame(rows), None,
                  {"metric": m.name, "freq": freq}, plotter=_plot_feature_dynamics)


def _plot_feature_dynamics(result: Result, n_cols: int = 3) -> go.Figure:
    t, name = result.table, result.info["metric"]
    feats = list(dict.fromkeys(t["feature"]))
    n_rows = max(1, -(-len(feats) // n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=feats,
                        vertical_spacing=min(0.12, 0.6 / n_rows))
    for i, f in enumerate(feats):
        g = t[t["feature"] == f]
        fig.add_scatter(x=g["period"], y=g[name], mode="lines+markers", name=f, showlegend=False,
                        row=i // n_cols + 1, col=i % n_cols + 1)
    return style(fig, f"{name.upper()} по периодам (однопризнаковые модели)",
                 height=300 * n_rows + 80)


def learning_curve(model, title: str = "Метрика по итерациям обучения") -> go.Figure:
    """Кривые метрики по итерациям обучения LightGBM (по ``evals_result_``)."""
    est = getattr(model, "estimator_", model)
    evals = getattr(est, "evals_result_", None)
    if not evals:
        raise ValueError("У модели нет evals_result_ — обучите её с eval_set.")
    fig = go.Figure()
    for i, (name, metrics) in enumerate(evals.items()):
        for metric_name, values in metrics.items():
            fig.add_scatter(y=values, mode="lines", name=f"{name} {metric_name}",
                            line={"color": color(i)})
    return style(fig, title, height=450, xaxis_title="итерация")
