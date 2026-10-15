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
GROUPINGS = ("components", "pairwise")


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
    grouping: str = "components",
    n_splits: int = 5,
    auc_gap_threshold: float = 0.01,
    top_k: int = 3,
    model: str = "lgbm",
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Result:
    """Удаляет сильно коррелирующие числовые признаки, оставляя более сильные по AUC.

    Объединяет прежние ``select_features_by_auc_cv_hybrid`` (``strategy="hybrid"``) и
    ``select_features_by_auc_cv`` (``strategy="model"``, ``model="catboost"``).

    Два способа решать, кого удалять (``grouping``):

    - ``"components"`` — признаки, связанные цепочками ``|corr| > threshold``, собираются в
      группы (компоненты связности), и в каждой группе остаётся **один** лучший. Цепочка
      A ~ B ~ C даёт одну группу, даже если A и C между собой не связаны — останется один из трёх;
    - ``"pairwise"`` — признаки идут от сильного к слабому, и признак остаётся, если он не
      коррелирует сильнее порога ни с одним **уже оставленным**. Из каждой сильно
      коррелирующей пары выпадает более слабый; в цепочке A ~ B ~ C остаются A и C.

    Parameters
    ----------
    features : list[str], optional
        Проверяемые признаки; учитываются только числовые, остальные проходят без изменений.
    strategy : {"direct", "model", "hybrid"}
        Как сравнивать признаки по силе.
        ``"direct"`` — AUC признака как скора (быстро);
        ``"model"`` — CV AUC однофакторной модели для всех признаков групп;
        ``"hybrid"`` — ``"direct"``, а если разрыв между 1-м и 2-м в группе меньше
        ``auc_gap_threshold`` — модельный AUC для ``top_k`` лучших. При
        ``grouping="pairwise"`` — модельный AUC решает пару, если разрыв по ``"direct"``
        меньше ``auc_gap_threshold`` (``top_k`` не используется).
    grouping : {"components", "pairwise"}
        Группы целиком или попарное сравнение (см. выше).
    model, params
        Модель для ``"model"``/``"hybrid"`` (по умолчанию маленький LightGBM).

    Returns
    -------
    Result
        ``table``: ``feature, action, reason, group, kept, corr_with_kept, direct_auc,
        model_auc`` — по строке на каждый признак, у которого есть сильная корреляция;
        ``kept`` — признак, из-за которого удалён (для оставленных — он сам);
        ``selected`` — все ``features`` без удалённых.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy должен быть одним из {STRATEGIES}")
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping должен быть одним из {GROUPINGS}")
    features = list(df.columns if features is None else features)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric].reset_index(drop=True)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("Количество строк в df и y должно совпадать.")

    groups, corr = correlation_groups(X, threshold, method)
    info: dict[str, Any] = {"threshold": threshold, "method": method, "strategy": strategy,
                            "grouping": grouping, "n_groups": len(groups)}
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

    if grouping == "pairwise":
        rows = _pairwise_rows(groups, corr, threshold, strategy, direct, model_auc,
                              auc_gap_threshold)
    else:
        rows = _component_rows(groups, corr, strategy, direct, model_auc, model_cache,
                               auc_gap_threshold, top_k)
    for row in rows:
        row["model_auc"] = model_cache.get(row["feature"], np.nan)

    table = pd.DataFrame(rows, columns=columns)
    dropped = set(table.loc[table["action"] == "drop", "feature"])
    selected = [f for f in features if f not in dropped]
    info.update(n_in_groups=len(in_groups), n_dropped=len(dropped),
                n_model_evaluated=len(model_cache))
    if verbose:
        print(f"Групп: {len(groups)}; признаков в группах: {len(in_groups)}; "
              f"удалено: {len(dropped)}; модель обучалась для {len(model_cache)} признаков.")
    return Result("correlation_filter", table, selected, info, plotter=_plot_groups)


def _component_rows(groups, corr, strategy, direct, model_auc, model_cache, auc_gap_threshold,
                    top_k) -> list[dict[str, Any]]:
    """В каждой группе остаётся один признак."""
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
                "direct_auc": direct[f],
            })
    return rows


def _pairwise_rows(groups, corr, threshold, strategy, direct, model_auc,
                   auc_gap_threshold) -> list[dict[str, Any]]:
    """Попарно: признак остаётся, если не коррелирует сильнее порога ни с одним оставленным.

    Признаки идут по убыванию силы (``"model"`` — модельный AUC, иначе ``"direct"``). При
    ``"hybrid"`` близкую по ``"direct"`` пару решает модельный AUC: если более поздний признак
    по модели сильнее всех оставленных, с которыми конфликтует, он занимает их место.
    """
    group_of = {f: gid for gid, group in enumerate(groups, start=1) for f in group}
    score = model_auc if strategy == "model" else direct.__getitem__
    order = sorted(group_of, key=lambda f: (-score(f), f))
    kept: list[str] = []
    decision: dict[str, tuple[str, str, str]] = {}  # признак -> (оставлен ли, из-за кого, как)

    for f in order:
        conflicts = [k for k in kept if corr.loc[f, k] > threshold]
        if not conflicts:
            kept.append(f)
            decision[f] = ("keep", f, "")
            continue
        if (strategy == "hybrid"
                and all(direct[k] - direct[f] < auc_gap_threshold for k in conflicts)
                and all(model_auc(f) > model_auc(k) for k in conflicts)):
            for k in conflicts:  # f сильнее по модели — занимает место оставленных
                kept.remove(k)
                decision[k] = ("drop", f, "model")
            kept.append(f)
            decision[f] = ("keep", f, "")
            continue
        winner = max(conflicts, key=lambda k: corr.loc[f, k])
        how = "model" if strategy == "model" else (
            "model" if strategy == "hybrid" and direct[winner] - direct[f] < auc_gap_threshold
            else "direct")
        decision[f] = ("drop", winner, how)

    rows = []
    for f in order:
        action, by, how = decision[f]
        c = float(corr.loc[f, by])
        if action == "keep":
            reason = ""
        else:
            values = (f"model_auc={model_auc(f):.4f} vs {model_auc(by):.4f}" if how == "model"
                      else f"auc={direct[f]:.4f} vs {direct[by]:.4f}")
            reason = (f"Удалён в паре с {by} (method={how}, |corr|={c:.3f}, {values})")
        rows.append({"feature": f, "action": action, "reason": reason, "group": group_of[f],
                     "kept": by, "corr_with_kept": c, "direct_auc": direct[f]})
    return rows


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
