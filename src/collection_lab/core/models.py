"""Адаптеры моделей с единым интерфейсом.

Любой алгоритм библиотеки (CV, отбор признаков, обучение) работает с моделью через
:class:`BaseModel`: ``fit(X, y, eval_set=..., cat_features=...)``, ``predict(X)``,
``feature_importance()``. Категориальные признаки адаптер готовит сам: категории
запоминаются на train и одинаково применяются к val/test/скорингу.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from collection_lab.config import (
    CATBOOST_PARAMS,
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    merge_params,
)
from collection_lab.data.types import detect_categorical, normalize_missing, to_numeric_frame

TASKS = ("binary", "regression")


class FeaturePreparer:
    """Приводит признаки к виду, который принимают бустинги.

    ``fit`` запоминает список категориальных колонок и их категории (по train);
    ``transform`` превращает категориальные в ``category`` с этими же категориями
    (новые значения → NaN), остальные — в ``float``.

    Parameters
    ----------
    cat_features : list[str], optional
        Категориальные колонки. По умолчанию определяются автоматически.
    """

    def __init__(self, cat_features: Iterable[str] | None = None):
        self.cat_features = None if cat_features is None else list(cat_features)

    def fit(self, X: pd.DataFrame) -> FeaturePreparer:
        cats = detect_categorical(X) if self.cat_features is None else self.cat_features
        self.features_ = list(X.columns)
        self.cat_features_ = [c for c in cats if c in X.columns]
        self.categories_ = {}
        for c in self.cat_features_:
            values = normalize_missing(X[c]).dropna().astype(str)
            self.categories_[c] = sorted(values.unique())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features_ if c not in X.columns]
        if missing:
            raise KeyError(f"В данных нет признаков: {missing}")
        out = to_numeric_frame(X[self.features_], cat_cols=self.cat_features_)
        for c in self.cat_features_:
            s = normalize_missing(out[c])
            s = s.where(s.isna(), s.astype(str))
            out[c] = pd.Categorical(s, categories=self.categories_[c])
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def make_lgbm_ready(
    X: pd.DataFrame, cat_cols: Iterable[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Совместимость со старым ``scr.model_tools.make_lgbm_ready``: ``(X_ready, cat_cols)``.

    Для train/val/test лучше использовать один :class:`FeaturePreparer` (fit на train),
    чтобы категории кодировались одинаково.
    """
    prep = FeaturePreparer(cat_cols).fit(X)
    return prep.transform(X), prep.cat_features_


class BaseModel:
    """Общий интерфейс адаптера.

    Parameters
    ----------
    params : dict, optional
        Параметры поверх дефолтов из :mod:`collection_lab.config`.
    task : {"binary", "regression"}
    early_stopping_rounds : int | None
        Ранняя остановка по ``eval_set`` (если он передан в ``fit``).
    verbose : int
        Период логирования итераций (0 — молча).
    """

    name = "base"
    default_params: dict[str, Any] = {}

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        task: str = "binary",
        early_stopping_rounds: int | None = EARLY_STOPPING_ROUNDS,
        verbose: int = 0,
    ):
        if task not in TASKS:
            raise ValueError(f"task должен быть одним из {TASKS}")
        self.params = merge_params(self.default_params, params)
        self.task = task
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.estimator_ = None
        self.best_iteration_: int | None = None

    # --- общий API -------------------------------------------------------------------
    def clone(self, params: dict[str, Any] | None = None) -> BaseModel:
        """Необученная копия с теми же настройками (``params`` — дополнительные правки)."""
        return type(self)(
            merge_params(self.params, params),
            task=self.task,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y,
        *,
        eval_set: tuple[pd.DataFrame, Any] | None = None,
        cat_features: Iterable[str] | None = None,
        sample_weight=None,
    ) -> BaseModel:
        self.preparer_ = FeaturePreparer(cat_features).fit(X)
        self.features_ = self.preparer_.features_
        self.cat_features_ = self.preparer_.cat_features_
        self._fit(X, np.asarray(y), eval_set, sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Вероятность класса 1 (binary) или прогноз (regression)."""
        self._check_fitted()
        return self._predict(X[self.features_])

    def feature_importance(self, kind: str = "gain") -> pd.Series:
        """Важности признаков (``'gain'`` или ``'split'``), по убыванию."""
        self._check_fitted()
        values = self._importance(kind)
        return pd.Series(values, index=self.features_, name=kind).sort_values(ascending=False)

    @property
    def n_iterations_(self) -> int:
        """Число деревьев, которое реально используется при прогнозе."""
        self._check_fitted()
        return self.best_iteration_ or self._total_iterations()

    def _check_fitted(self) -> None:
        if self.estimator_ is None:
            raise RuntimeError(f"Модель {self.name} ещё не обучена — вызовите fit().")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task!r}, params={self.params})"

    # --- реализация в наследниках ----------------------------------------------------
    def _fit(self, X, y, eval_set, sample_weight) -> None:
        raise NotImplementedError

    def _predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def _importance(self, kind: str) -> np.ndarray:
        raise NotImplementedError

    def _total_iterations(self) -> int:
        raise NotImplementedError


@lru_cache(maxsize=1)
def _lgbm_has_eval_xy() -> bool:
    import lightgbm as lgb

    return "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters


class LGBMModel(BaseModel):
    """LightGBM (sklearn-API)."""

    name = "lgbm"
    default_params = LGBM_PARAMS

    def _fit(self, X, y, eval_set, sample_weight) -> None:
        import lightgbm as lgb

        params = dict(self.params)
        if self.task == "binary":
            params.setdefault("objective", "binary")
            est = lgb.LGBMClassifier(**params)
        else:
            params.setdefault("objective", "regression")
            est = lgb.LGBMRegressor(**params)

        Xp = self.preparer_.transform(X)
        fit_kwargs: dict[str, Any] = {
            "sample_weight": sample_weight,
            "categorical_feature": self.cat_features_ or "auto",
        }
        callbacks = [lgb.log_evaluation(period=self.verbose)]
        if eval_set is not None:
            X_va, y_va = self.preparer_.transform(eval_set[0]), np.asarray(eval_set[1])
            if _lgbm_has_eval_xy():  # lightgbm >= 4.7
                fit_kwargs.update(eval_X=(X_va,), eval_y=(y_va,))
            else:
                fit_kwargs["eval_set"] = [(X_va, y_va)]
            if self.early_stopping_rounds:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=bool(self.verbose))
                )
        est.fit(Xp, y, callbacks=callbacks, **fit_kwargs)
        self.estimator_ = est
        self.best_iteration_ = getattr(est, "best_iteration_", None) or None

    def _predict(self, X) -> np.ndarray:
        Xp = self.preparer_.transform(X)
        if self.task == "binary":
            return self.estimator_.predict_proba(Xp)[:, 1]
        return self.estimator_.predict(Xp)

    def _importance(self, kind: str) -> np.ndarray:
        if kind not in ("gain", "split"):
            raise ValueError("kind для LightGBM: 'gain' или 'split'")
        return self.estimator_.booster_.feature_importance(importance_type=kind)

    def _total_iterations(self) -> int:
        return self.estimator_.booster_.current_iteration()


class CatBoostModel(BaseModel):
    """CatBoost. Требует ``pip install collection_lab[catboost]``."""

    name = "catboost"
    default_params = CATBOOST_PARAMS

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self.preparer_.transform(X)
        for c in self.cat_features_:  # CatBoost хочет строки без NaN
            out[c] = out[c].astype(object).where(out[c].notna(), "NaN").astype(str)
        return out

    def _fit(self, X, y, eval_set, sample_weight) -> None:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as e:  # pragma: no cover
            raise ImportError("Нужен catboost: pip install collection_lab[catboost]") from e

        params = dict(self.params)
        params.setdefault("verbose", self.verbose or False)
        if self.task == "binary":
            params.setdefault("loss_function", "Logloss")
            params.setdefault("eval_metric", "AUC")
            est = CatBoostClassifier(**params)
        else:
            params.setdefault("loss_function", "RMSE")
            est = CatBoostRegressor(**params)

        fit_kwargs: dict[str, Any] = {"cat_features": self.cat_features_,
                                      "sample_weight": sample_weight}
        if eval_set is not None:
            X_va, y_va = eval_set
            fit_kwargs.update(eval_set=(self._prepare(X_va), np.asarray(y_va)), use_best_model=True,
                              early_stopping_rounds=self.early_stopping_rounds)
        est.fit(self._prepare(X), y, **fit_kwargs)
        self.estimator_ = est
        best = est.get_best_iteration() if eval_set is not None else None
        self.best_iteration_ = None if best is None else best + 1

    def _predict(self, X) -> np.ndarray:
        Xp = self._prepare(X)
        if self.task == "binary":
            return self.estimator_.predict_proba(Xp)[:, 1]
        return self.estimator_.predict(Xp)

    def _importance(self, kind: str) -> np.ndarray:
        types = {"gain": "FeatureImportance", "split": "FeatureImportance",
                 "prediction_change": "PredictionValuesChange"}
        if kind not in types:
            raise ValueError(f"kind для CatBoost: {list(types)}")
        return self.estimator_.get_feature_importance(type=types[kind])

    def _total_iterations(self) -> int:
        return self.estimator_.tree_count_


MODELS: dict[str, type[BaseModel]] = {
    "lgbm": LGBMModel,
    "lightgbm": LGBMModel,
    "catboost": CatBoostModel,
    "cb": CatBoostModel,
}


def make_model(
    model: str | BaseModel = "lgbm",
    params: dict[str, Any] | None = None,
    **kwargs,
) -> BaseModel:
    """Создаёт адаптер по имени (``'lgbm'``, ``'catboost'``) или клонирует готовый.

    ``kwargs`` — ``task``, ``early_stopping_rounds``, ``verbose`` (только для имени).
    """
    if isinstance(model, BaseModel):
        return model.clone(params)
    try:
        cls = MODELS[model.lower()]
    except KeyError:
        raise ValueError(f"Неизвестная модель {model!r}. Доступны: {sorted(MODELS)}") from None
    return cls(deepcopy(params), **kwargs)
