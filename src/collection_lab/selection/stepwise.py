"""Пошаговые методы: backward elimination, forward addition, инкрементальная оценка набора."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from collection_lab.config import LGBM_INCREMENTAL_PARAMS, RANDOM_STATE, merge_params
from collection_lab.core.cv import cross_validate, make_folds
from collection_lab.core.models import BaseModel, make_model
from collection_lab.core.results import Result
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import Metric, get_metric
from collection_lab.plotting.theme import bar_colors, color, style


# ----------------------------------------------------------------------------------------
# Общий помощник для holdout-оценки (train / valid + сегменты)
# ----------------------------------------------------------------------------------------
class _HoldoutEvaluator:
    def __init__(self, X_train, y_train, X_valid, y_valid, *, model, params, cat_features,
                 segments, min_segment_size, metric, early_stopping_rounds):
        self.X_train, self.y_train = X_train, np.asarray(y_train)
        self.X_valid, self.y_valid = X_valid, np.asarray(y_valid)
        self.metric: Metric = get_metric(metric)
        self.model = make_model(model, params, early_stopping_rounds=early_stopping_rounds)
        self.cat_features = list(cat_features)
        if segments is None:
            self.segments, self.valid_segments = None, []
        else:
            seg = X_valid[segments] if isinstance(segments, str) else pd.Series(
                np.asarray(segments), index=X_valid.index)
            self.segments = seg.astype(object).fillna("NaN").to_numpy()
            counts = pd.Series(self.segments).value_counts()
            self.valid_segments = counts[counts >= min_segment_size].index.tolist()

    def __call__(self, feats: list[str]) -> tuple[dict[str, float], BaseModel]:
        m = self.model.clone()
        m.fit(self.X_train[feats], self.y_train, eval_set=(self.X_valid[feats], self.y_valid),
              cat_features=[c for c in self.cat_features if c in feats])
        p_valid = m.predict(self.X_valid[feats])
        name = self.metric.name
        out = {f"{name}_train": self.metric(self.y_train, m.predict(self.X_train[feats])),
               f"{name}_valid": self.metric(self.y_valid, p_valid)}
        for seg in self.valid_segments:
            mask = self.segments == seg
            out[f"{name}_seg={seg}"] = self.metric(self.y_valid[mask], p_valid[mask])
        return out, m


# ----------------------------------------------------------------------------------------
# Backward elimination
# ----------------------------------------------------------------------------------------
def backward_elimination(
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    y_valid,
    features: Sequence[str] | None = None,
    *,
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    segments=None,
    min_segment_size: int = 100,
    min_features: int = 1,
    importance: str = "gain",
    metric: str = "auc",
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> Result:
    """Удаляет по одному наименее важный признак до ``min_features`` и на каждом шаге
    считает метрику на train/valid и в сегментах (бывший ``backward_feature_elimination``).

    Parameters
    ----------
    segments : str | array-like, optional
        Колонка ``X_valid`` или массив меток сегментов для строк ``X_valid``.

    Returns
    -------
    Result
        ``table`` — шаги: ``step, n_features, <metric>_train, <metric>_valid,
        <metric>_seg=..., removed_feature, removed_importance, delta_after_remove_*``.
        ``plot()`` — 2×2 панели; ``plot(kind="per_feature")`` — дельты по каждой фиче.
    """
    seg_col = segments if isinstance(segments, str) else None
    features = [c for c in (X_train.columns if features is None else features) if c != seg_col]
    if cat_features is None:
        cat_features = detect_categorical(X_train[features])
    ev = _HoldoutEvaluator(X_train, y_train, X_valid, y_valid, model=model, params=params,
                           cat_features=cat_features, segments=segments,
                           min_segment_size=min_segment_size, metric=metric,
                           early_stopping_rounds=early_stopping_rounds)
    name = ev.metric.name
    if verbose and ev.valid_segments:
        print(f"Сегменты (размер >= {min_segment_size}): {ev.valid_segments}")

    current = list(features)
    history = []
    step = 0
    while len(current) >= min_features:
        metrics, m = ev(current)
        imp = m.feature_importance(importance)
        worst = imp.idxmin()
        history.append({"step": step, "n_features": len(current), **metrics,
                        "removed_feature": worst, "removed_importance": float(imp.min())})
        if verbose:
            print(f"Step {step:>3} | n={len(current):>3} | train={metrics[f'{name}_train']:.4f} "
                  f"valid={metrics[f'{name}_valid']:.4f} | drop: {worst}")
        current.remove(worst)
        step += 1
        if not current:
            break
    if current:  # финальный замер — чтобы посчитать дельту для последнего удалённого
        metrics, _ = ev(current)
        history.append({"step": step, "n_features": len(current), **metrics,
                        "removed_feature": None, "removed_importance": np.nan})

    table = pd.DataFrame(history)
    score_cols = [c for c in table.columns if c.startswith(f"{name}_")]
    for col in score_cols:
        table[col.replace(f"{name}_", "delta_after_remove_", 1)] = (
            table[col].shift(-1) - table[col] if ev.metric.greater_is_better
            else table[col] - table[col].shift(-1))
    table = table[table["removed_feature"].notna()].reset_index(drop=True)
    info = {"metric": name, "segments": ev.valid_segments}
    return Result("backward_elimination", table, None, info, plotter=_plot_backward)


def _plot_backward(result: Result, kind: str = "history", top_n: int | None = None,
                   n_cols: int = 3) -> go.Figure:
    t, name, segs = result.table, result.info["metric"], result.info["segments"]
    if kind == "per_feature":
        return _plot_backward_per_feature(t, name, segs, top_n, n_cols)
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f"Общий {name}", f"{name} по сегментам",
        f"Изменение valid {name} после удаления признака",
        f"Изменение {name} по сегментам после удаления"))
    x = t["n_features"]
    fig.add_scatter(x=x, y=t[f"{name}_train"], name="train", mode="lines+markers", row=1, col=1)
    fig.add_scatter(x=x, y=t[f"{name}_valid"], name="valid", mode="lines+markers", row=1, col=1)
    fig.add_bar(x=x, y=t["delta_after_remove_valid"], name="Δ valid", showlegend=False,
                marker_color=bar_colors(t["delta_after_remove_valid"].fillna(0)),
                customdata=t["removed_feature"],
                hovertemplate="удалён %{customdata}: Δ=%{y:.4f}<extra></extra>", row=2, col=1)
    for i, seg in enumerate(segs):
        c = color(i + 2)
        fig.add_scatter(x=x, y=t[f"{name}_seg={seg}"], name=str(seg), mode="lines+markers",
                        line={"color": c}, legendgroup=str(seg), row=1, col=2)
        fig.add_scatter(x=x, y=t[f"delta_after_remove_seg={seg}"], name=str(seg),
                        mode="lines+markers", line={"color": c}, legendgroup=str(seg),
                        showlegend=False, row=2, col=2)
    fig.update_xaxes(autorange="reversed", title_text="число признаков")
    return style(fig, "Backward elimination", height=800)


def _plot_backward_per_feature(t, name, segs, top_n, n_cols) -> go.Figure:
    df = t.assign(abs_delta=t["delta_after_remove_valid"].abs())
    df = df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    n_rows = max(1, -(-len(df) // n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=df["removed_feature"].tolist(),
                        vertical_spacing=min(0.3 / n_rows + 0.05, 0.15))
    labels = ["ALL"] + [str(s) for s in segs]
    for i, row in df.iterrows():
        vals = [row["delta_after_remove_valid"]] + [
            row.get(f"delta_after_remove_seg={s}", np.nan) for s in segs]
        fig.add_bar(x=labels, y=vals, marker_color=bar_colors(np.nan_to_num(vals)),
                    text=[f"{v:+.4f}" for v in vals], showlegend=False,
                    row=i // n_cols + 1, col=i % n_cols + 1)
    return style(fig, f"Δ {name} по сегментам при удалении каждого признака",
                 height=280 * n_rows + 80)


# ----------------------------------------------------------------------------------------
# Forward addition
# ----------------------------------------------------------------------------------------
def forward_addition(
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    y_valid,
    base_features: Sequence[str],
    candidate_features: Sequence[str] | dict[str, Sequence[str]],
    *,
    mode: str = "one_by_one",
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    segments=None,
    min_segment_size: int = 100,
    metric: str = "auc",
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> Result:
    """Эффект добавления признаков к базовому набору (бывший ``forward_feature_addition``).

    Parameters
    ----------
    mode : {"one_by_one", "all_together", "groups"}
        ``"one_by_one"`` — каждый кандидат отдельно; ``"all_together"`` — все разом;
        ``"groups"`` — ``candidate_features`` = ``{имя_группы: [признаки]}``.

    Returns
    -------
    Result
        ``table``: строка ``BASE`` + по строке на эксперимент с метриками и дельтами.
    """
    base_features = list(base_features)
    if mode == "one_by_one":
        experiments = [(f, [f]) for f in candidate_features]
    elif mode == "all_together":
        cands = list(candidate_features)
        experiments = [("+".join(cands) if len(cands) <= 3 else f"ALL ({len(cands)})", cands)]
    elif mode == "groups":
        if not isinstance(candidate_features, dict):
            raise ValueError("Для mode='groups' candidate_features должен быть dict")
        experiments = [(k, list(v)) for k, v in candidate_features.items()]
    else:
        raise ValueError(f"Неизвестный mode: {mode}")

    all_feats = base_features + [f for _, fs in experiments for f in fs]
    if cat_features is None:
        cat_features = detect_categorical(X_train[list(dict.fromkeys(all_feats))])
    ev = _HoldoutEvaluator(X_train, y_train, X_valid, y_valid, model=model, params=params,
                           cat_features=cat_features, segments=segments,
                           min_segment_size=min_segment_size, metric=metric,
                           early_stopping_rounds=early_stopping_rounds)
    name = ev.metric.name
    base_metrics, _ = ev(base_features)
    if verbose:
        print(f"BASE ({len(base_features)} признаков) | valid={base_metrics[f'{name}_valid']:.4f}")
    rows = [{"added": "BASE", "n_features": len(base_features), **base_metrics}]
    for label, new in experiments:
        new = [f for f in new if f not in base_features]
        if not new:
            if verbose:
                print(f"{label}: все признаки уже в базе, пропускаю")
            continue
        metrics, _ = ev(base_features + new)
        row = {"added": label, "n_features": len(base_features) + len(new), **metrics}
        for k, v in metrics.items():
            row[f"delta_{k}"] = ev.metric.delta(v, base_metrics[k])
        rows.append(row)
        if verbose:
            print(f"+ {label:<30} | valid={metrics[f'{name}_valid']:.4f} "
                  f"(Δ={row[f'delta_{name}_valid']:+.4f})")
    info = {"metric": name, "segments": ev.valid_segments, "mode": mode}
    return Result("forward_addition", pd.DataFrame(rows), None, info, plotter=_plot_forward)


def _plot_forward(result: Result) -> go.Figure:
    name, segs = result.info["metric"], result.info["segments"]
    df = result.table[result.table["added"] != "BASE"]
    df = df.sort_values(f"delta_{name}_valid")
    fig = go.Figure()
    fig.add_bar(y=df["added"], x=df[f"delta_{name}_valid"], orientation="h", name="valid (all)",
                marker_color=bar_colors(df[f"delta_{name}_valid"]),
                marker_line={"color": "black", "width": 1})
    for i, seg in enumerate(segs):
        col = f"delta_{name}_seg={seg}"
        if col in df.columns:
            fig.add_bar(y=df["added"], x=df[col], orientation="h", name=str(seg),
                        marker_color=color(i + 2), opacity=0.85)
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(barmode="group")
    return style(fig, "Эффект добавления признаков к базовому набору",
                 height=max(400, 40 * len(df) * (1 + len(segs)) // 2 + 150),
                 xaxis_title=f"Δ {name} относительно базы")


# ----------------------------------------------------------------------------------------
# Incremental feature eval
# ----------------------------------------------------------------------------------------
def incremental_feature_eval(
    X: pd.DataFrame,
    y,
    features: Sequence[str],
    *,
    cat_features: Iterable[str] | None = None,
    direction: str = "forward",
    mode: str = "ordered",
    n_splits: int = 5,
    test_size: float = 0.2,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    metric: str = "auc",
    max_steps: int | None = None,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Result:
    """Пошагово добавляет (forward) или убирает (backward) признаки и считает CV-метрику.

    Parameters
    ----------
    direction : {"forward", "backward"}
    mode : {"ordered", "greedy"}
        ``"ordered"`` — порядок из ``features`` (forward берёт первый оставшийся,
        backward убирает последний); ``"greedy"`` — на каждом шаге лучший вариант.
    n_jobs : int
        ``ordered`` — параллель по фолдам, ``greedy`` — по кандидатам.

    Returns
    -------
    Result
        ``table``: ``step, n_features, feature_changed, action, auc_mean, auc_std,
        n_folds_ok, auc_delta, auc_ratio, auc_signal_ratio, features_set``
        (префикс ``auc`` заменяется на имя метрики); ``selected`` — лучший набор.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction: 'forward' или 'backward'")
    if mode not in ("ordered", "greedy"):
        raise ValueError("mode: 'ordered' или 'greedy'")
    features = list(features)
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"Признаки не найдены в X: {missing}")
    if cat_features is None:
        cat_features = detect_categorical(X[features])
    cat_features = list(cat_features)
    metric_obj = get_metric(metric)
    name = metric_obj.name
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    folds = make_folds(y, n_splits, test_size=test_size, random_state=random_state)
    base_model = make_model(model, merge_params(LGBM_INCREMENTAL_PARAMS, params)
                            if model == "lgbm" else params,
                            early_stopping_rounds=early_stopping_rounds)
    n_fold_jobs = min(abs(n_jobs), len(folds)) if n_jobs not in (0, 1) else 1

    def cv(feats: list[str], jobs: int = 1):
        if not feats:
            return np.nan, np.nan, 0
        r = cross_validate(X, y, features=feats,
                           cat_features=[c for c in cat_features if c in feats],
                           model=base_model, folds=folds, metric=metric_obj,
                           early_stopping_rounds=early_stopping_rounds, return_oof=False,
                           n_jobs=jobs)
        ok = [s for s in r.fold_scores if not np.isnan(s)]
        return (float(np.mean(ok)), float(np.std(ok)), len(ok)) if ok else (np.nan, np.nan, 0)

    full_mean, _, _ = cv(features, n_fold_jobs)
    if verbose:
        print(f"[baseline] {name} на всех {len(features)} признаках: {full_mean:.5f}")

    remaining = list(features)
    current: list[str] = [] if direction == "forward" else list(features)
    total = len(features) if max_steps is None else min(max_steps, len(features))
    records, prev = [], np.nan
    for step in tqdm(range(1, total + 1), desc=f"{direction}/{mode}", disable=not verbose):
        candidates = remaining if direction == "forward" else current
        if not candidates:
            break
        if mode == "ordered":
            chosen = candidates[0] if direction == "forward" else candidates[-1]
            trial = current + [chosen] if direction == "forward" else [
                f for f in current if f != chosen]
            mean, std, n_ok = cv(trial, n_fold_jobs)
        else:
            def attempt(f, cur=tuple(current)):
                t = [*cur, f] if direction == "forward" else [c for c in cur if c != f]
                return (f, *cv(t))
            results = (Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(attempt)(f) for f in candidates) if n_jobs != 1
                else [attempt(f) for f in candidates])
            valid = [r for r in results if not np.isnan(r[1])]
            if not valid:
                break
            chosen, mean, std, n_ok = max(valid, key=lambda r: r[1] if metric_obj.greater_is_better
                                          else -r[1])
            trial = current + [chosen] if direction == "forward" else [
                f for f in current if f != chosen]
        denom = full_mean - 0.5
        records.append({
            "step": step, "n_features": len(trial), "feature_changed": chosen,
            "action": "add" if direction == "forward" else "remove",
            f"{name}_mean": mean, f"{name}_std": std, "n_folds_ok": n_ok,
            f"{name}_delta": mean - prev if not np.isnan(prev) else np.nan,
            f"{name}_ratio": mean / full_mean if full_mean else np.nan,
            f"{name}_signal_ratio": (mean - 0.5) / denom if denom else np.nan,
            "features_set": list(trial),
        })
        current = trial
        remaining = [f for f in remaining if f != chosen]
        prev = mean

    table = pd.DataFrame(records)
    best = None
    if not table.empty:
        idx = (table[f"{name}_mean"].idxmax() if metric_obj.greater_is_better
               else table[f"{name}_mean"].idxmin())
        best = table.loc[idx, "features_set"]
    info = {"metric": name, "score_full": full_mean, "direction": direction, "mode": mode}
    return Result("incremental_feature_eval", table, best, info, plotter=_plot_incremental)


def _plot_incremental(result: Result) -> go.Figure:
    t, name, full = result.table, result.info["metric"], result.info["score_full"]
    x, mean, std = t["n_features"], t[f"{name}_mean"], t[f"{name}_std"].fillna(0)
    signal = t[f"{name}_signal_ratio"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"{name} vs число признаков ({result.info['direction']})", "Доля полезного сигнала"))
    fig.add_scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([mean + std, (mean - std)[::-1]]),
                    fill="toself", line={"width": 0}, fillcolor="rgba(99,110,250,0.2)",
                    name="±std", hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=x, y=mean, mode="lines+markers", name=name, customdata=t["feature_changed"],
                    hovertemplate="%{customdata}: %{y:.4f}<extra></extra>", row=1, col=1)
    fig.add_hline(y=full, line_dash="dot", line_color="red", row=1, col=1,
                  annotation_text=f"все признаки ({full:.4f})")
    fig.add_scatter(x=x, y=signal, mode="lines+markers", name="signal ratio",
                    line={"color": "#2ca02c"}, row=1, col=2)
    fig.add_hline(y=0.95, line_dash="dash", line_color="orange", row=1, col=2,
                  annotation_text="95%")
    fig.update_xaxes(title_text="число признаков")
    return style(fig, "Incremental feature eval", height=450)
