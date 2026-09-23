"""Эксперимент: как качество зависит от объёма обучающей выборки."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from collection_lab.core.models import BaseModel, make_model
from collection_lab.core.results import Result
from collection_lab.data.split import DataSplit
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import get_metric
from collection_lab.plotting.theme import style


def sample_size_curve(
    split: DataSplit,
    features: Sequence[str],
    sizes: Iterable[int],
    *,
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    n_repeats: int = 5,
    early_stopping_rounds: int = 20,
    metric: str = "auc",
) -> Result:
    """Для каждого размера ``n`` обучает модель на случайной подвыборке train ``n_repeats``
    раз (early stopping по val) и считает метрику на val и test.

    Returns
    -------
    Result
        ``table``: ``sample_size, repeat, <metric>_val, <metric>_test, best_iteration``;
        ``plot()`` — среднее ± std по размерам.
    """
    features = list(features)
    metric_obj = get_metric(metric)
    train = split.train
    X_va, y_va = split.xy("val", features)
    if cat_features is None:
        cat_features = detect_categorical(train[features])
    cat_features = [c for c in cat_features if c in features]
    base = make_model(model, params, early_stopping_rounds=early_stopping_rounds)
    rows = []
    for size in sorted({min(int(s), len(train)) for s in sizes}):
        for rep in range(n_repeats):
            sub = train.sample(n=size, random_state=rep)
            m = base.clone().fit(sub[features], sub[split.target], eval_set=(X_va, y_va),
                                 cat_features=cat_features)
            row = {"sample_size": size, "repeat": rep,
                   f"{metric_obj.name}_val": metric_obj(y_va, m.predict(X_va)),
                   "best_iteration": m.n_iterations_}
            if split.test is not None:
                row[f"{metric_obj.name}_test"] = metric_obj(split.test[split.target],
                                                            m.predict(split.test[features]))
            rows.append(row)
    return Result("sample_size_curve", pd.DataFrame(rows), None, {"metric": metric_obj.name},
                  plotter=_plot_sample_size)


def _plot_sample_size(result: Result) -> go.Figure:
    t, name = result.table, result.info["metric"]
    fig = go.Figure()
    for part in ("val", "test"):
        col = f"{name}_{part}"
        if col in t.columns:
            g = t.groupby("sample_size")[col].agg(["mean", "std"]).reset_index()
            fig.add_scatter(x=g["sample_size"], y=g["mean"], name=part, mode="lines+markers",
                            error_y={"type": "data", "array": g["std"].fillna(0)})
    fig.update_xaxes(type="log")
    return style(fig, f"{name} vs объём обучающей выборки", height=450,
                 xaxis_title="размер train", yaxis_title=name)
