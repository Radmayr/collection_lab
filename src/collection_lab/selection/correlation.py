"""Отбор среди сильно коррелирующих признаков: в каждой группе остаётся один лучший."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from collection_lab.config import LGBM_UNIVARIATE_PARAMS, RANDOM_STATE, merge_params
from collection_lab.core.cv import cross_validate, make_folds
from collection_lab.core.results import Result
from collection_lab.plotting.theme import style
from collection_lab.selection.univariate import direct_auc

STRATEGIES = ("direct", "model", "hybrid")


def correlation_groups(
    X: pd.DataFrame, threshold: float = 0.8, method: str = "pearson"
) -> tuple[list[list[str]], pd.DataFrame]:
    """Группы признаков, связанных цепочками ``|corr| > threshold`` (компоненты связности).

    Returns
    -------
    groups : list[list[str]]
        Только группы из 2+ признаков.
    corr : pd.DataFrame
        Матрица ``|corr|``.
    """
    corr = X.corr(method=method).abs()
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    rows, cols = np.where((corr.to_numpy() > threshold) & upper)
    graph: dict[str, list[str]] = defaultdict(list)
    for i, j in zip(rows, cols, strict=True):
        a, b = corr.columns[i], corr.columns[j]
        graph[a].append(b)
        graph[b].append(a)

    visited: set[str] = set()
    groups = []
    for node in graph:
        if node in visited:
            continue
        group, stack = [], [node]
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                group.append(n)
                stack.extend(graph[n])
        if len(group) > 1:
            groups.append(sorted(group))
    return groups, corr


def correlation_filter(
    df: pd.DataFrame,
    y,
    features: Sequence[str] | None = None,
    *,
    threshold: float = 0.8,
    method: str = "pearson",
    strategy: str = "hybrid",
    n_splits: int = 5,
    auc_gap_threshold: float = 0.01,
    top_k: int = 3,
    model: str = "lgbm",
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Result:
    """В каждой группе коррелирующих числовых признаков оставляет один — с лучшим AUC.

    Объединяет прежние ``select_features_by_auc_cv_hybrid`` (``strategy="hybrid"``) и
    ``select_features_by_auc_cv`` (``strategy="model"``, ``model="catboost"``).

    Parameters
    ----------
    features : list[str], optional
        Проверяемые признаки; учитываются только числовые, остальные проходят без изменений.
    strategy : {"direct", "model", "hybrid"}
        ``"direct"`` — AUC признака как скора (быстро);
        ``"model"`` — CV AUC однофакторной модели для всех признаков групп;
        ``"hybrid"`` — ``"direct"``, а если разрыв между 1-м и 2-м в группе меньше
        ``auc_gap_threshold`` — модельный AUC для ``top_k`` лучших.
    model, params
        Модель для ``"model"``/``"hybrid"`` (по умолчанию маленький LightGBM).

    Returns
    -------
    Result
        ``table``: ``feature, action, reason, group, kept, corr_with_kept, direct_auc,
        model_auc``; ``selected`` — все ``features`` без удалённых.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy должен быть одним из {STRATEGIES}")
    features = list(df.columns if features is None else features)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric].reset_index(drop=True)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("Количество строк в df и y должно совпадать.")

    groups, corr = correlation_groups(X, threshold, method)
    info: dict[str, Any] = {"threshold": threshold, "method": method, "strategy": strategy,
                            "n_groups": len(groups)}
    columns = ["feature", "action", "reason", "group", "kept", "corr_with_kept",
               "direct_auc", "model_auc"]
    if not groups:
        if verbose:
            print("Нет сильно коррелирующих признаков.")
        return Result("correlation_filter", pd.DataFrame(columns=columns), features, info,
                      plotter=_plot_groups)

    folds = make_folds(y, n_splits, random_state=random_state)
    in_groups = sorted({f for g in groups for f in g})
    direct = {f: direct_auc(X[f], y, folds) for f in in_groups}
    model_params = merge_params(LGBM_UNIVARIATE_PARAMS, params) if model == "lgbm" else params
    model_cache: dict[str, float] = {}

    def model_auc(f: str) -> float:
        if f not in model_cache:
            res = cross_validate(X[[f]], y, model=model, params=model_params, folds=folds,
                                 cat_features=[], return_oof=False,
                                 early_stopping_rounds=early_stopping_rounds)
            model_cache[f] = res.mean
        return model_cache[f]

    rows = []
    for gid, group in enumerate(groups, start=1):
        ranked = sorted(group, key=lambda f: direct[f], reverse=True)
        chosen, how = ranked[0], "direct"
        if strategy == "model":
            chosen = max(group, key=model_auc)
            how = "model"
        elif strategy == "hybrid" and direct[ranked[0]] - direct[ranked[1]] < auc_gap_threshold:
            candidates = ranked[:top_k]
            chosen = max(candidates, key=model_auc)
            how = f"model(top{len(candidates)})"
        for f in group:
            keep = f == chosen
            c = float(corr.loc[f, chosen])
            rows.append({
                "feature": f, "action": "keep" if keep else "drop",
                "reason": "" if keep else (
                    f"Удалён из группы с {chosen} (method={how}, |corr|={c:.3f}, "
                    f"auc={direct[f]:.4f} vs {direct[chosen]:.4f})"),
                "group": gid, "kept": chosen, "corr_with_kept": c,
                "direct_auc": direct[f], "model_auc": model_cache.get(f, np.nan),
            })

    table = pd.DataFrame(rows, columns=columns)
    dropped = set(table.loc[table["action"] == "drop", "feature"])
    selected = [f for f in features if f not in dropped]
    info.update(n_in_groups=len(in_groups), n_dropped=len(dropped),
                n_model_evaluated=len(model_cache))
    if verbose:
        print(f"Групп: {len(groups)}; признаков в группах: {len(in_groups)}; "
              f"удалено: {len(dropped)}; модель обучалась для {len(model_cache)} признаков.")
    return Result("correlation_filter", table, selected, info, plotter=_plot_groups)


def _plot_groups(result: Result) -> go.Figure:
    t = result.table
    if t.empty:
        return style(go.Figure(), "Коррелирующих групп нет")
    sizes = t.groupby(["group", "kept"]).size().reset_index(name="n").sort_values("n")
    fig = go.Figure(go.Bar(x=sizes["n"], y=[f"#{g}: {k}" for g, k in
                                            zip(sizes["group"], sizes["kept"], strict=True)],
                           orientation="h"))
    return style(fig, "Группы коррелирующих признаков (оставленный признак и размер группы)",
                 height=max(350, 22 * len(sizes) + 120), xaxis_title="признаков в группе")
