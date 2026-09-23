"""Отбор и оценка признаков по важности: кумулятивная, permutation, drop-column."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from collection_lab.config import RANDOM_STATE
from collection_lab.core.models import BaseModel, make_model
from collection_lab.core.results import Result
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import Metric, get_metric
from collection_lab.plotting.theme import bar_colors, style


def permutation_importance(
    model: BaseModel,
    X: pd.DataFrame,
    y,
    *,
    features: Sequence[str] | None = None,
    metric: str | Metric = "auc",
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Падение метрики при случайном перемешивании признака (по обученной модели).

    Returns
    -------
    pd.DataFrame
        ``feature, importance_mean, importance_std`` по убыванию; importance > 0 — признак полезен.
    """
    metric = get_metric(metric)
    y = np.asarray(y)
    features = list(model.features_ if features is None else features)
    base = metric(y, model.predict(X))
    rng = np.random.default_rng(random_state)
    rows = []
    for f in features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = Xp[f].to_numpy()[rng.permutation(len(Xp))]
            drops.append(metric.delta(base, metric(y, model.predict(Xp))))
        rows.append({"feature": f, "importance_mean": float(np.mean(drops)),
                     "importance_std": float(np.std(drops))})
    return (pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def cumulative_importance_selection(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str] | None = None,
    *,
    cat_features: Iterable[str] | None = None,
    threshold: float = 0.90,
    importance: str = "gain",
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    test_size: float = 0.2,
    stratify: bool = True,
    balance_weights: bool = False,
    early_stopping_rounds: int = 100,
    metric: str | Metric = "auc",
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> Result:
    """Отбор по кумулятивной важности (бывший ``cum_importance_select_and_compare``).

    1) train/valid split; 2) модель на всех признаках; 3) важности → признаки, дающие
    ``threshold`` суммарной важности; 4) модель на отобранных; 5) сравнение метрики на valid.

    Parameters
    ----------
    importance : {"gain", "split", "permutation"}
        ``"permutation"`` — перестановочная важность на train (прежний
        ``importance_type="PredictionValuesChange"``).
    balance_weights : bool
        Добавить ``scale_pos_weight = neg / pos`` (для LightGBM).

    Returns
    -------
    Result
        ``table``: ``feature, importance, importance_norm, importance_cum, selected``;
        ``info``: метрика full/selected, дельта, число признаков, best_iteration.
    """
    features = [c for c in (df.columns if features is None else features) if c != target]
    X, y = df[features], df[target].astype(int)
    if cat_features is None:
        cat_features = detect_categorical(X)
    cat_features = list(cat_features)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )
    params = dict(params or {})
    if balance_weights:
        pos = int(y_tr.sum())
        params["scale_pos_weight"] = (len(y_tr) - pos) / max(pos, 1)
    metric = get_metric(metric)

    full = make_model(model, params, early_stopping_rounds=early_stopping_rounds)
    full.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features)
    score_full = metric(y_va, full.predict(X_va))

    if importance == "permutation":
        perm = permutation_importance(full, X_tr, y_tr, metric=metric, n_repeats=n_repeats,
                                      random_state=random_state)
        imp = perm.set_index("feature")["importance_mean"].clip(lower=0)
    elif importance in ("gain", "split"):
        imp = full.feature_importance(importance)
    else:
        raise ValueError("importance: 'gain', 'split' или 'permutation'")

    table = imp.sort_values(ascending=False).rename("importance").rename_axis("feature")
    table = table.reset_index()
    total = float(table["importance"].sum())
    if not np.isfinite(total) or total <= 0:
        table["importance_norm"] = 0.0
        table["importance_cum"] = 0.0
        k = 1
    else:
        table["importance_norm"] = table["importance"] / total
        table["importance_cum"] = table["importance_norm"].cumsum()
        k = max(1, int(np.searchsorted(table["importance_cum"].to_numpy(), threshold) + 1))
    table["selected"] = False
    table.loc[: k - 1, "selected"] = True
    selected = table.loc[table["selected"], "feature"].tolist()

    sel = full.clone()
    sel.fit(X_tr[selected], y_tr, eval_set=(X_va[selected], y_va),
            cat_features=[c for c in cat_features if c in selected])
    score_sel = metric(y_va, sel.predict(X_va[selected]))

    info = {
        f"{metric.name}_full": score_full,
        f"{metric.name}_selected": score_sel,
        "delta_selected_minus_full": score_sel - score_full,
        "n_features_full": len(features),
        "n_features_selected": len(selected),
        "threshold": threshold,
        "importance": importance,
        "best_iter_full": full.best_iteration_,
        "best_iter_selected": sel.best_iteration_,
        "models": {"full": full, "selected": sel},
    }
    return Result("cumulative_importance", table, selected, info, plotter=_plot_cumulative)


def _plot_cumulative(result: Result, top_k: int = 60) -> go.Figure:
    t = result.table.head(top_k)
    fig = go.Figure()
    fig.add_bar(x=t["feature"], y=t["importance_norm"], name="доля важности",
                marker_color=np.where(t["selected"], "#636EFA", "#c7c7c7"))
    fig.add_scatter(x=t["feature"], y=t["importance_cum"], name="накопленная", yaxis="y2",
                    mode="lines+markers")
    fig.add_hline(y=result.info["threshold"], line_dash="dash", line_color="red", yref="y2")
    fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 1.05]})
    fig.update_xaxes(tickangle=45)
    return style(fig, f"Кумулятивная важность: отобрано {len(result.selected)} признаков",
                 height=500)


def drop_column_importance(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    test: pd.DataFrame | None = None,
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 100,
    metric: str | Metric = "auc",
    verbose: bool = True,
) -> Result:
    """Важность признака = падение метрики при его удалении и переобучении
    (бывший ``evaluate_feature_importance_by_removal``).

    Полная модель обучается внутри, поэтому ``auc_full`` передавать не нужно.

    Returns
    -------
    Result
        ``table``: ``feature, score_full_val, score_without_val, delta_val`` (+ те же для
        test); ``delta > 0`` — без признака хуже, т.е. признак полезен.
    """
    features = list(features)
    metric = get_metric(metric)
    if cat_features is None:
        cat_features = detect_categorical(train[features])
    cat_features = list(cat_features)
    parts = {"val": val} if test is None else {"val": val, "test": test}

    def fit_score(feats: list[str]) -> dict[str, float]:
        m = make_model(model, params, early_stopping_rounds=early_stopping_rounds)
        m.fit(train[feats], train[target], eval_set=(val[feats], val[target]),
              cat_features=[c for c in cat_features if c in feats])
        return {name: metric(part[target], m.predict(part[feats]))
                for name, part in parts.items()}

    full = fit_score(features)
    rows = []
    for f in features:
        if verbose:
            print(f"Удаление признака: {f}")
        reduced = fit_score([c for c in features if c != f])
        row = {"feature": f}
        for name in parts:
            row[f"score_full_{name}"] = full[name]
            row[f"score_without_{name}"] = reduced[name]
            row[f"delta_{name}"] = metric.delta(full[name], reduced[name])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("delta_val", ascending=False, ignore_index=True)
    info = {"metric": metric.name, **{f"score_full_{k}": v for k, v in full.items()}}
    return Result("drop_column_importance", table, None, info, plotter=_plot_drop_column)


def _plot_drop_column(result: Result) -> go.Figure:
    t = result.table.iloc[::-1]
    fig = go.Figure()
    for col, name in [("delta_val", "val"), ("delta_test", "test")]:
        if col in t.columns:
            fig.add_bar(x=t[col], y=t["feature"], orientation="h", name=name,
                        marker_color=bar_colors(t[col]) if name == "val" else "#636EFA",
                        opacity=1.0 if name == "val" else 0.6)
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(barmode="group")
    return style(fig, "Потеря метрики при удалении признака (> 0 — признак полезен)",
                 height=max(350, 26 * len(t) + 120), xaxis_title="Δ метрики")


def importance_plot(importance: pd.Series, top_k: int = 20, title: str | None = None):
    """Горизонтальный bar-chart важностей (замена ``plot_feature_importance``)."""
    s = importance.sort_values(ascending=False).head(top_k).iloc[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h"))
    return style(fig, title or f"Top-{len(s)} feature importance ({importance.name or ''})",
                 height=max(350, 24 * len(s) + 120))
