"""Рекурсивное исключение признаков (RFE) с допуском по метрике и контролем по сегментам."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from collection_lab.config import RANDOM_STATE
from collection_lab.core.cv import CVResult, cross_validate, make_folds
from collection_lab.core.models import BaseModel, LGBMModel, make_model
from collection_lab.core.results import Result
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import get_metric
from collection_lab.plotting.theme import color, style


def _iterations_param(model: BaseModel, n: int) -> dict[str, int]:
    return {"n_estimators": n} if isinstance(model, LGBMModel) else {"iterations": n}


def rfe(
    X: pd.DataFrame,
    y,
    features: Sequence[str] | None = None,
    *,
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    min_features: int = 1,
    step: int = 1,
    tol: float = 0.0,
    segments=None,
    segment_tol: float | None = None,
    n_splits: int = 5,
    folds=None,
    metric: str = "auc",
    missing_threshold: float = 0.999,
    importance: str = "gain",
    early_stopping_rounds: int = 50,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Result:
    """RFE с допуском: признак удаляется, только если CV-метрика после удаления ухудшается
    не больше чем на ``tol`` (и в каждом сегменте — не больше чем на ``segment_tol``).

    Объединяет прежние ``rfe_lightgbm_binary_auc``, ``rfe_lightgbm_binary_auc_segment`` и
    ``rfe_catboost_binary_auc``.

    Алгоритм: префильтры (доля пропусков ``>= missing_threshold``, константы) → базовая
    CV-метрика → на каждой итерации модель на всех данных даёт важности, ``step`` самых
    слабых признаков по очереди пробуют удалить → если за итерацию не удалён ни один,
    отбор останавливается.

    Parameters
    ----------
    model : {"lgbm", "catboost"} или BaseModel
    params : dict, optional
        Параметры модели. Для модели важностей число деревьев берётся равным среднему
        ``best_iteration`` из последней CV.
    step : int
        Сколько самых слабых признаков пробовать удалить за итерацию.
    tol : float
        Допустимое ухудшение общей метрики.
    segments : str | array-like, optional
        Колонка ``X`` или массив меток сегментов.
    segment_tol : float, optional
        Допустимое ухудшение в каждом сегменте (по умолчанию 0, если заданы сегменты).
    metric : str
        ``'auc'``, ``'gini'``, ``'logloss'``, ... — направление учитывается автоматически.

    Returns
    -------
    Result
        ``table`` — лог с колонками ``iteration, action, feature, reason, importance,
        score_before, score_after, delta, n_features_before, n_features_after``
        (+ ``min_segment_delta, worst_segment, seg_before_*, seg_after_*``);
        ``selected`` — итоговые признаки.
    """
    metric_obj = get_metric(metric)
    seg_col = segments if isinstance(segments, str) else None
    features = [c for c in (X.columns if features is None else features) if c != seg_col]
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    seg_values = None
    if segments is not None:
        seg_values = X[segments] if isinstance(segments, str) else pd.Series(np.asarray(segments))
        seg_values = seg_values.astype(object).fillna("NaN").to_numpy()
        segment_tol = 0.0 if segment_tol is None else segment_tol
    if cat_features is None:
        cat_features = detect_categorical(X[features])
    cat_features = list(cat_features)
    base_model = make_model(model, params, early_stopping_rounds=early_stopping_rounds)
    if folds is None:
        folds = make_folds(y, n_splits, random_state=random_state)

    def say(msg: str) -> None:
        if verbose:
            print(msg)

    def cv(feats: list[str]) -> CVResult:
        return cross_validate(X, y, features=feats,
                              cat_features=[c for c in cat_features if c in feats],
                              model=base_model, folds=folds, metric=metric_obj,
                              segments=seg_values, early_stopping_rounds=early_stopping_rounds,
                              return_oof=False, n_jobs=n_jobs)

    logs: list[dict[str, Any]] = []

    def log(iteration, action, feature, reason, *, imp=np.nan, before=None, after=None,
            n_before, n_after) -> None:
        row = {"iteration": iteration, "action": action, "feature": feature, "reason": reason,
               "importance": imp,
               "score_before": before.mean if before else np.nan,
               "score_after": after.mean if after else np.nan,
               "delta": metric_obj.delta(after.mean, before.mean) if before and after else np.nan,
               "n_features_before": n_before, "n_features_after": n_after}
        if seg_values is not None:
            sb = before.segment_means if before else {}
            sa = after.segment_means if after else {}
            deltas = {s: metric_obj.delta(sa.get(s, np.nan), v) for s, v in sb.items()}
            deltas = {s: d for s, d in deltas.items() if not np.isnan(d)}
            row["min_segment_delta"] = min(deltas.values()) if deltas else np.nan
            row["worst_segment"] = min(deltas, key=deltas.get) if deltas else ""
            row.update({f"seg_before_{s}": v for s, v in sb.items()})
            row.update({f"seg_after_{s}": v for s, v in sa.items()})
        logs.append(row)

    # --- префильтры ---
    current = list(features)
    miss = X[current].isna().mean()
    for f in miss[miss >= missing_threshold].index:
        log(0, "drop", f, f"pre_filter_missing_frac>={missing_threshold}",
            n_before=len(current), n_after=len(current) - 1)
        current.remove(f)
    nun = X[current].nunique(dropna=False)
    for f in nun[nun <= 1].index:
        log(0, "drop", f, "pre_filter_constant", n_before=len(current), n_after=len(current) - 1)
        current.remove(f)
    if len(current) < min_features:
        raise ValueError(f"После префильтров осталось {len(current)} < {min_features=} признаков")

    say(f"RFE ({base_model.name}, {metric_obj.name}): старт с {len(current)} признаков")
    baseline = cv(current)
    history = [{"n_features": len(current), "score": baseline.mean, **{
        f"seg={s}": v for s, v in baseline.segment_means.items()}}]
    iteration = 0

    while len(current) > min_features:
        iteration += 1
        n_iter = baseline.mean_best_iteration
        imp_model = base_model.clone(_iterations_param(base_model, n_iter) if n_iter else None)
        imp_model.fit(X[current], y, cat_features=[c for c in cat_features if c in current])
        importances = imp_model.feature_importance(importance).sort_values()
        tries = min(step, len(current) - min_features)
        dropped = 0
        for f in importances.index[:tries]:
            if len(current) <= min_features:
                break
            trial = cv([c for c in current if c != f])
            delta = metric_obj.delta(trial.mean, baseline.mean)
            global_ok = delta >= -tol
            seg_ok, min_seg = True, np.nan
            if seg_values is not None:
                seg_deltas = [metric_obj.delta(trial.segment_means.get(s, np.nan), v)
                              for s, v in baseline.segment_means.items()]
                min_seg = float(np.nanmin(seg_deltas)) if seg_deltas else np.nan
                seg_ok = bool(np.isnan(min_seg) or min_seg >= -segment_tol)
            if global_ok and seg_ok:
                reason = f"drop: delta={delta:.6f} >= -{tol}"
                if seg_values is not None:
                    reason += f", min_segment_delta={min_seg:.6f} >= -{segment_tol}"
                log(iteration, "drop", f, reason, imp=float(importances[f]), before=baseline,
                    after=trial, n_before=len(current), n_after=len(current) - 1)
                current.remove(f)
                baseline = trial
                dropped += 1
                history.append({"n_features": len(current), "score": trial.mean, **{
                    f"seg={s}": v for s, v in trial.segment_means.items()}})
                say(f"  итерация {iteration}: удалён {f} ({metric_obj.name}={trial.mean:.4f})")
            else:
                parts = []
                if not global_ok:
                    parts.append(f"delta={delta:.6f} < -{tol}")
                if not seg_ok:
                    parts.append(f"min_segment_delta={min_seg:.6f} < -{segment_tol}")
                log(iteration, "keep", f, "keep: " + "; ".join(parts),
                    imp=float(importances[f]), before=baseline, after=trial,
                    n_before=len(current), n_after=len(current))
        if dropped == 0:
            log(iteration, "stop", "", "ни один признак нельзя удалить с заданным допуском",
                before=baseline, after=baseline, n_before=len(current), n_after=len(current))
            break

    log(iteration + 1, "final", ",".join(current), "final_selected_features",
        before=baseline, after=baseline, n_before=len(current), n_after=len(current))
    say(f"RFE: итог {len(current)} признаков, {metric_obj.name}={baseline.mean:.4f}")
    info = {"metric": metric_obj.name, "score_final": baseline.mean, "tol": tol,
            "segment_tol": segment_tol, "model": base_model.name,
            "history": pd.DataFrame(history)}
    return Result("rfe", pd.DataFrame(logs), current, info, plotter=_plot_rfe)


def _plot_rfe(result: Result) -> go.Figure:
    h: pd.DataFrame = result.info["history"]
    fig = go.Figure()
    fig.add_scatter(x=h["n_features"], y=h["score"], mode="lines+markers", name="ALL",
                    line={"width": 3, "color": "black"})
    for i, col in enumerate(c for c in h.columns if c.startswith("seg=")):
        fig.add_scatter(x=h["n_features"], y=h[col], mode="lines+markers", name=col[4:],
                        line={"color": color(i)})
    fig.update_xaxes(autorange="reversed")
    return style(fig, f"RFE: {result.info['metric']} по мере удаления признаков", height=450,
                 xaxis_title="число признаков", yaxis_title=result.info["metric"])
