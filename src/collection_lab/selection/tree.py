"""Однофакторное дерево решений: качество каждого признака и его разбиение на интервалы."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed

from collection_lab.config import RANDOM_STATE, merge_params
from collection_lab.core.cv import Folds, cross_validate, make_folds
from collection_lab.core.models import make_model
from collection_lab.core.results import Result
from collection_lab.data.types import category_strings, detect_categorical
from collection_lab.plotting.theme import color, style
from collection_lab.utils.parallel import tqdm_joblib

# Одно дерево: бустинг LightGBM из одного дерева с шагом 1 — без подвыборок строк и колонок.
TREE_PARAMS: dict[str, Any] = {
    "n_estimators": 1,
    "learning_rate": 1.0,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1e-3,
    "n_jobs": 1,  # параллелим снаружи, по признакам
    "verbosity": -1,
}
MAX_CATEGORIES_IN_RULE = 5


def _rule(values: pd.Series, is_cat: bool) -> str:
    """Подпись листа по фактическим значениям признака в нём."""
    has_nan = values.isna().any()
    known = values.dropna()
    if known.empty:
        return "пропуски"
    if is_cat:
        cats = sorted(category_strings(known).unique())
        text = ", ".join(cats[:MAX_CATEGORIES_IN_RULE])
        if len(cats) > MAX_CATEGORIES_IN_RULE:
            text += f" и ещё {len(cats) - MAX_CATEGORIES_IN_RULE}"
    else:
        x = pd.to_numeric(known, errors="coerce")
        text = f"[{x.min():.4g}; {x.max():.4g}]"
    return text + (" + пропуски" if has_nan else "")


def _leaves(model, X: pd.DataFrame, y: np.ndarray, feature: str, is_cat: bool) -> pd.DataFrame:
    """Листья дерева, обученного на всех данных: правило, размер, доля таргета."""
    leaf = model.estimator_.predict(model.preparer_.transform(X), pred_leaf=True)
    leaf = np.asarray(leaf).reshape(len(X), -1)[:, 0]
    data = pd.DataFrame({"leaf": leaf, "x": X[feature].to_numpy(), "y": y})
    rows = []
    for _, part in data.groupby("leaf"):
        numeric = pd.to_numeric(part["x"], errors="coerce") if not is_cat else None
        rows.append({
            "rule": _rule(part["x"], is_cat),
            "n": len(part),
            "share": len(part) / len(data),
            "n_target": int(part["y"].sum()),
            "target_rate": float(part["y"].mean()),
            "_order": (numeric.min() if numeric is not None and numeric.notna().any()
                       else np.inf),
        })
    table = pd.DataFrame(rows)
    # числовой признак — листья по возрастанию значений (лист только с пропусками — в конце),
    # категориальный — по доле таргета
    table = table.sort_values("target_rate" if is_cat else "_order", kind="stable")
    table = table.drop(columns="_order").reset_index(drop=True)
    table.insert(0, "leaf", np.arange(1, len(table) + 1))
    table.insert(0, "feature", feature)
    return table


def _monotonic(leaves: pd.DataFrame, is_cat: bool) -> float | bool:
    """Монотонна ли доля таргета по листьям, упорядоченным по значению (для числовых)."""
    if is_cat:
        return np.nan
    rates = leaves.loc[leaves["rule"] != "пропуски", "target_rate"].to_numpy()
    if len(rates) < 2:
        return np.nan
    diffs = np.diff(rates)
    return bool((diffs >= 0).all() or (diffs <= 0).all())


def _tree_feature(feature: str, X: pd.DataFrame, y: np.ndarray, is_cat: bool, folds: Folds,
                  params: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame | None]:
    row: dict[str, Any] = {"feature": feature, "auc_mean": np.nan, "auc_std": np.nan,
                           "auc_min": np.nan, "auc_max": np.nan, "n_leaves": 0,
                           "monotonic": np.nan, "n_splits_ok": 0, "status": "ok"}
    s = X[feature]
    if s.nunique(dropna=False) < 2:  # «заполнен / пропуск» — тоже разбиение
        row["status"] = "constant_or_all_nan"
        return row, None
    cats = [feature] if is_cat else []
    try:
        res = cross_validate(X, y, cat_features=cats, params=params, folds=folds,
                             return_oof=False, early_stopping_rounds=None)
        model = make_model("lgbm", params, early_stopping_rounds=None).fit(X, y, cat_features=cats)
        leaves = _leaves(model, X, y, feature, is_cat)
    except Exception as e:  # noqa: BLE001 — ошибка одной фичи не должна валить весь расчёт
        row["status"] = f"error: {type(e).__name__}: {e}"
        return row, None
    aucs = np.array([a for a in res.fold_scores if not np.isnan(a)])
    if len(aucs):
        row.update(auc_mean=aucs.mean(), auc_std=aucs.std(), auc_min=aucs.min(),
                   auc_max=aucs.max(), n_splits_ok=len(aucs))
    else:
        row["status"] = "no_valid_folds"
    row["n_leaves"] = len(leaves)
    row["monotonic"] = _monotonic(leaves, is_cat)
    return row, leaves


def univariate_tree_scores(
    X: pd.DataFrame,
    y,
    features: Sequence[str] | None = None,
    *,
    cat_features: Iterable[str] | None = None,
    max_depth: int = 3,
    min_leaf_share: float = 0.05,
    n_splits: int = 5,
    folds: Folds | None = None,
    params: dict[str, Any] | None = None,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Result:
    """Дерево решений на каждом признаке отдельно: ранжирование признаков по AUC дерева на
    кросс-валидации и разбиение каждого признака на интервалы (листья дерева).

    В отличие от :func:`univariate_scores` (маленький бустинг из сотен деревьев) здесь ровно
    одно дерево — его разбиение можно прочитать: на какие интервалы оно делит признак и какая
    доля таргета в каждом. Пропуски и категориальные признаки обрабатываются так же, как в
    моделях библиотеки.

    Parameters
    ----------
    X : pd.DataFrame
        Признаки (могут быть лишние колонки — берутся ``features``).
    y : array-like
        Бинарный таргет.
    features : list[str], optional
        Признаки; по умолчанию все колонки ``X``.
    cat_features : list[str], optional
        Категориальные признаки; по умолчанию определяются автоматически.
    max_depth : int
        Глубина дерева: до ``2 ** max_depth`` листьев.
    min_leaf_share : float
        Минимальная доля наблюдений в листе (от обучающей части фолда) — защита от мелких
        случайных интервалов.
    n_splits : int
        Число фолдов кросс-валидации; ``<= 1`` — holdout 20%.
    folds : list, optional
        Готовые фолды из :func:`~collection_lab.core.make_folds`.
    params : dict, optional
        Параметры LightGBM поверх одного дерева (например, ``{"min_split_gain": 1}``).
    n_jobs : int
        Параллельность по признакам (процессы joblib).

    Returns
    -------
    Result
        ``table`` — по строке на признак, по убыванию ``auc_mean``: ``feature, auc_mean,
        auc_std, auc_min, auc_max, n_leaves, monotonic, n_splits_ok, status``.
        ``monotonic`` — растёт или падает доля таргета по интервалам числового признака.
        ``info["leaves"]`` — листья всех признаков (дерево на всех данных): ``feature, leaf,
        rule, n, share, n_target, target_rate``. ``selected`` — признаки со статусом ``ok``.
        ``plot()`` — AUC по признакам; ``plot(feature="имя")`` — доля таргета по листьям.
    """
    features = list(X.columns if features is None else features)
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"Признаки не найдены в X: {missing}")
    if not 0 < min_leaf_share < 0.5:
        raise ValueError("min_leaf_share должен быть в интервале (0, 0.5).")
    cat_set = set(detect_categorical(X[features]) if cat_features is None else cat_features)
    y = np.asarray(y)
    X = X[features].reset_index(drop=True)
    if folds is None:
        folds = make_folds(y, n_splits, random_state=random_state)
    min_leaf = max(1, int(min_leaf_share * min(len(tr) for tr, _ in folds)))
    params = merge_params({**TREE_PARAMS, "max_depth": max_depth, "num_leaves": 2 ** max_depth,
                           "min_child_samples": min_leaf, "min_data_per_group": min_leaf,
                           "random_state": random_state}, params)

    tasks = (delayed(_tree_feature)(f, X[[f]], y, f in cat_set, folds, params)
             for f in features)
    with tqdm_joblib(total=len(features), desc="Univariate tree", disable=not verbose):
        outputs = Parallel(n_jobs=n_jobs)(tasks)

    table = (pd.DataFrame([row for row, _ in outputs])
             .sort_values("auc_mean", ascending=False, na_position="last")
             .reset_index(drop=True))
    leaves = [lv for _, lv in outputs if lv is not None]
    leaves_df = (pd.concat(leaves, ignore_index=True) if leaves
                 else pd.DataFrame(columns=["feature", "leaf", "rule", "n", "share",
                                            "n_target", "target_rate"]))
    selected = table.loc[table["status"] == "ok", "feature"].tolist()
    info = {"n_splits": len(folds), "max_depth": max_depth, "min_leaf_share": min_leaf_share,
            "min_leaf": min_leaf, "target_rate": float(y.mean()), "params": params,
            "leaves": leaves_df}
    return Result("univariate_tree_scores", table, selected, info, plotter=_plot_tree)


def _plot_tree(result: Result, feature: str | None = None, top_k: int = 40) -> go.Figure:
    if feature is not None:
        leaves = result.info["leaves"]
        t = leaves[leaves["feature"] == feature]
        if t.empty:
            raise ValueError(f"Нет листьев для признака {feature!r}.")
        fig = go.Figure(go.Bar(
            x=t["rule"], y=t["target_rate"], marker_color=color(0), customdata=t[["n", "share"]],
            text=[f"{v:.1%}" for v in t["target_rate"]], textposition="outside",
            hovertemplate="%{x}<br>доля таргета %{y:.2%}<br>n = %{customdata[0]} "
                          "(%{customdata[1]:.1%})<extra></extra>"))
        fig.add_hline(y=result.info["target_rate"], line_dash="dash", line_color="grey",
                      annotation_text="в среднем")
        auc = result.table.set_index("feature").loc[feature, "auc_mean"]
        fig = style(fig, f"{feature}: доля таргета по листьям дерева (AUC {auc:.3f})",
                    xaxis_title="интервал значений", yaxis_title="доля таргета")
        return fig.update_layout(hovermode="closest")

    t = result.table.dropna(subset=["auc_mean"]).head(top_k).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["auc_mean"], y=t["feature"], orientation="h", marker_color=color(0),
        error_x={"type": "data", "array": t["auc_std"]}, customdata=t["n_leaves"],
        hovertemplate="%{y}: AUC=%{x:.4f}, листьев %{customdata}<extra></extra>"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey")
    lo = max(0.45, float(t["auc_mean"].min()) - 0.02) if len(t) else 0.45
    fig.update_xaxes(range=[lo, float(t["auc_mean"].max()) + 0.02] if len(t) else None)
    return style(fig, f"AUC дерева решений на одном признаке (top-{min(top_k, len(t))})",
                 height=max(350, 22 * len(t) + 120), xaxis_title="AUC")
