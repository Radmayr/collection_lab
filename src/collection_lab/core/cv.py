"""Кросс-валидация — общий «движок» для обучения, отбора признаков и сравнения моделей."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from collection_lab.config import EARLY_STOPPING_ROUNDS, RANDOM_STATE
from collection_lab.core.models import BaseModel, make_model
from collection_lab.data.types import detect_categorical
from collection_lab.metrics.classification import Metric, get_metric

Folds = list[tuple[np.ndarray, np.ndarray]]


def make_folds(
    y,
    n_splits: int = 5,
    *,
    stratify: bool = True,
    shuffle: bool = True,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Folds:
    """Позиционные индексы фолдов ``[(train_idx, valid_idx), ...]``.

    При ``n_splits <= 1`` — один holdout-сплит с долей валидации ``test_size``.
    Фиксированные фолды передаются во все сравнения, чтобы разница метрик
    объяснялась признаками/параметрами, а не разбиением.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    if n_splits <= 1:
        tr, va = train_test_split(idx, test_size=test_size, random_state=random_state,
                                  stratify=y if stratify else None)
        return [(np.sort(tr), np.sort(va))]
    splitter = (StratifiedKFold if stratify else KFold)(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None
    )
    return [(tr, va) for tr, va in splitter.split(idx, y)]


@dataclass
class CVResult:
    """Результат :func:`cross_validate`.

    Attributes
    ----------
    fold_scores : list[float]
        Метрика на каждом фолде.
    segment_scores : pd.DataFrame | None
        Метрика по сегментам: строки — сегменты, колонки ``fold_1..k``, ``mean``, ``n``.
    oof : pd.Series | None
        Out-of-fold прогноз с индексом исходного ``X``.
    best_iterations : list[int | None]
        Число деревьев после early stopping на каждом фолде.
    """

    metric: Metric
    features: list[str]
    fold_scores: list[float]
    best_iterations: list[int | None]
    segment_scores: pd.DataFrame | None = None
    oof: pd.Series | None = None
    models: list[BaseModel] = field(default_factory=list, repr=False)

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.fold_scores)) if self.fold_scores else float("nan")

    @property
    def std(self) -> float:
        return float(np.nanstd(self.fold_scores)) if self.fold_scores else float("nan")

    @property
    def segment_means(self) -> dict[Any, float]:
        """Средняя метрика по сегментам: ``{segment: value}``."""
        if self.segment_scores is None:
            return {}
        return self.segment_scores["mean"].to_dict()

    @property
    def mean_best_iteration(self) -> int | None:
        its = [i for i in self.best_iterations if i]
        return int(round(np.mean(its))) if its else None

    def summary(self) -> dict[str, Any]:
        out = {f"{self.metric.name}_mean": self.mean, f"{self.metric.name}_std": self.std,
               "n_folds": len(self.fold_scores), "n_features": len(self.features),
               "best_iteration": self.mean_best_iteration}
        for seg, value in self.segment_means.items():
            out[f"{self.metric.name}_seg={seg}"] = value
        return out

    def __repr__(self) -> str:
        return (f"CVResult({self.metric.name}={self.mean:.4f} ± {self.std:.4f}, "
                f"folds={len(self.fold_scores)}, features={len(self.features)})")


def _resolve_segments(X: pd.DataFrame, segments) -> np.ndarray | None:
    if segments is None:
        return None
    if isinstance(segments, str):
        return X[segments].astype(object).fillna("NaN").to_numpy()
    seg = pd.Series(np.asarray(segments, dtype=object))
    if len(seg) != len(X):
        raise ValueError("Длина segments не совпадает с X.")
    return seg.fillna("NaN").to_numpy()


def _run_fold(
    model: BaseModel, X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, va: np.ndarray,
    cat_features: list[str], metric: Metric, segments: np.ndarray | None, min_segment_size: int,
    use_early_stopping: bool,
) -> tuple[float, np.ndarray, dict, int | None, BaseModel]:
    m = model.clone()
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    m.fit(X_tr, y[tr], eval_set=(X_va, y[va]) if use_early_stopping else None,
          cat_features=cat_features)
    pred = m.predict(X_va)
    score = metric(y[va], pred)
    seg_scores = {}
    if segments is not None:
        seg_va = segments[va]
        for seg in pd.unique(seg_va):
            mask = seg_va == seg
            if mask.sum() >= max(min_segment_size, 2):
                seg_scores[seg] = (metric(y[va][mask], pred[mask]), int(mask.sum()))
    return score, pred, seg_scores, m.best_iteration_, m


def cross_validate(
    X: pd.DataFrame,
    y,
    *,
    features: Sequence[str] | None = None,
    cat_features: Iterable[str] | None = None,
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    task: str = "binary",
    folds: Folds | None = None,
    n_splits: int = 5,
    metric: str | Metric = "auc",
    segments=None,
    min_segment_size: int = 0,
    early_stopping_rounds: int | None = EARLY_STOPPING_ROUNDS,
    return_oof: bool = True,
    keep_models: bool = False,
    n_jobs: int = 1,
    random_state: int = RANDOM_STATE,
) -> CVResult:
    """Обучает модель на фолдах и считает метрику (общую и по сегментам).

    Parameters
    ----------
    X : pd.DataFrame
        Признаки (могут быть лишние колонки — берутся ``features``).
    y : array-like
        Таргет.
    features : list[str], optional
        Признаки модели. По умолчанию все колонки ``X`` (кроме колонки сегмента, если
        ``segments`` задан строкой).
    cat_features : list[str], optional
        Категориальные признаки; по умолчанию определяются автоматически.
    model : {"lgbm", "catboost"} или BaseModel
        Модель; ``params`` дополняют её параметры.
    folds : list, optional
        Готовые фолды из :func:`make_folds` (для сопоставимых сравнений). Иначе создаются
        ``n_splits`` стратифицированных фолдов (``n_splits <= 1`` — holdout 20%).
    metric : str | Metric
        ``'auc'``, ``'gini'``, ``'ks'``, ``'logloss'`` или своя метрика.
    segments : str | array-like, optional
        Имя колонки в ``X`` или массив меток: метрика считается и внутри каждого сегмента.
    min_segment_size : int
        Минимальный размер сегмента в валидационном фолде.
    early_stopping_rounds : int | None
        Ранняя остановка по валидационному фолду (``None`` — без неё).
    n_jobs : int
        Параллельность по фолдам (потоки). При ``n_jobs > 1`` уменьшите ``n_jobs`` модели.

    Returns
    -------
    CVResult
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X должен быть pandas.DataFrame")
    y_arr = np.asarray(y)
    if len(y_arr) != len(X):
        raise ValueError("Длины X и y не совпадают.")

    seg_arr = _resolve_segments(X, segments)
    if features is None:
        features = [c for c in X.columns if not (isinstance(segments, str) and c == segments)]
    features = list(features)
    Xf = X[features]
    if cat_features is None:
        cat_features = detect_categorical(Xf)
    cat_features = [c for c in cat_features if c in features]

    metric_obj = get_metric(metric)
    base = make_model(model, params, task=task, early_stopping_rounds=early_stopping_rounds)
    if isinstance(model, BaseModel):
        base.early_stopping_rounds = early_stopping_rounds
    if folds is None:
        folds = make_folds(y_arr, n_splits, stratify=task == "binary", random_state=random_state)

    run = delayed(_run_fold)
    args = dict(X=Xf, y=y_arr, cat_features=cat_features, metric=metric_obj,
                segments=seg_arr, min_segment_size=min_segment_size,
                use_early_stopping=bool(early_stopping_rounds))
    if n_jobs == 1:
        outputs = [_run_fold(base, tr=tr, va=va, **args) for tr, va in folds]
    else:
        outputs = Parallel(n_jobs=n_jobs, prefer="threads")(
            run(base, tr=tr, va=va, **args) for tr, va in folds
        )

    oof = np.full(len(X), np.nan)
    fold_scores, best_its, models, seg_rows = [], [], [], {}
    for k, ((_, va), (score, pred, seg_scores, best_it, m)) in enumerate(
        zip(folds, outputs, strict=True), start=1
    ):
        fold_scores.append(score)
        best_its.append(best_it)
        oof[va] = pred
        if keep_models:
            models.append(m)
        for seg, (value, n) in seg_scores.items():
            row = seg_rows.setdefault(seg, {"n": 0})
            row[f"fold_{k}"] = value
            row["n"] += n

    segment_df = None
    if seg_arr is not None:
        segment_df = pd.DataFrame.from_dict(seg_rows, orient="index")
        fold_cols = [c for c in segment_df.columns if c.startswith("fold_")]
        segment_df["mean"] = segment_df[fold_cols].mean(axis=1)
        segment_df = segment_df[fold_cols + ["mean", "n"]].sort_index()
        segment_df.index.name = "segment"

    return CVResult(
        metric=metric_obj,
        features=features,
        fold_scores=fold_scores,
        best_iterations=best_its,
        segment_scores=segment_df,
        oof=pd.Series(oof, index=X.index, name="oof") if return_oof else None,
        models=models,
    )
