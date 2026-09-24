"""Цепочка шагов отбора признаков с общим логом.

Пример::

    pipe = SelectionPipeline([
        QualityFilter(num_missing_threshold=0.999),
        CorrelationFilter(threshold=0.8),
        CumulativeImportance(threshold=0.9),
        RFE(tol=0.01, n_splits=3),
    ])
    pipe.fit(split.train, target="target", features=features)
    pipe.selected_, pipe.log_, pipe.summary(), pipe.plot()
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from collection_lab.core.results import Result
from collection_lab.data.types import detect_categorical
from collection_lab.plotting.theme import style
from collection_lab.selection.correlation import correlation_filter
from collection_lab.selection.filters import quality_filter
from collection_lab.selection.importance import cumulative_importance_selection
from collection_lab.selection.rfe import rfe
from collection_lab.selection.univariate import univariate_filter
from collection_lab.utils.io import write_csv


class Step:
    """Шаг пайплайна. Наследник реализует :meth:`run`."""

    name = "step"

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self, df: pd.DataFrame, target: str, features: list[str],
            cat_features: list[str]) -> Result:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{type(self).__name__}({args})"


class QualityFilter(Step):
    """:func:`~collection_lab.selection.filters.quality_filter`."""

    name = "quality_filter"

    def run(self, df, target, features, cat_features):
        return quality_filter(df, features, cat_features=cat_features, **self.kwargs)


class CorrelationFilter(Step):
    """:func:`~collection_lab.selection.correlation.correlation_filter`."""

    name = "correlation_filter"

    def run(self, df, target, features, cat_features):
        return correlation_filter(df, df[target], features, **self.kwargs)


class UnivariateFilter(Step):
    """:func:`~collection_lab.selection.univariate.univariate_filter`."""

    name = "univariate_filter"

    def run(self, df, target, features, cat_features):
        return univariate_filter(df, df[target], features,
                                 cat_features=[c for c in cat_features if c in features],
                                 **self.kwargs)


class CumulativeImportance(Step):
    """:func:`~collection_lab.selection.importance.cumulative_importance_selection`."""

    name = "cumulative_importance"

    def run(self, df, target, features, cat_features):
        return cumulative_importance_selection(
            df, target, features, cat_features=[c for c in cat_features if c in features],
            **self.kwargs)


class RFE(Step):
    """:func:`~collection_lab.selection.rfe.rfe`."""

    name = "rfe"

    def run(self, df, target, features, cat_features):
        return rfe(df, df[target], features,
                   cat_features=[c for c in cat_features if c in features], **self.kwargs)


class Custom(Step):
    """Произвольный шаг: ``func(df, target, features, cat_features) -> list[str] | Result``."""

    def __init__(self, func: Callable[..., Any], name: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.func = func
        self.name = name or getattr(func, "__name__", "custom")

    def run(self, df, target, features, cat_features):
        out = self.func(df, target, features, cat_features, **self.kwargs)
        if isinstance(out, Result):
            return out
        selected = list(out)
        table = pd.DataFrame({"feature": features,
                              "action": ["keep" if f in selected else "drop" for f in features],
                              "reason": ["" if f in selected else self.name for f in features]})
        return Result(self.name, table, selected)


class SelectionPipeline:
    """Последовательный отбор признаков.

    Каждый шаг получает признаки, оставшиеся после предыдущего. После ``fit`` доступны:

    - ``selected_`` — итоговый список признаков;
    - ``results_`` — ``{имя_шага: Result}`` с полными логами шагов;
    - ``log_`` — общий лог: ``step, feature, action, reason`` для всех выбывших;
    - ``summary()`` — сколько признаков вошло/вышло на каждом шаге; ``plot()`` — воронка.
    """

    def __init__(self, steps: Sequence[Step]):
        self.steps = list(steps)

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        features: Iterable[str] | None = None,
        cat_features: Iterable[str] | None = None,
        *,
        verbose: bool = True,
    ) -> SelectionPipeline:
        features = [c for c in (df.columns if features is None else features) if c != target]
        cat_features = list(detect_categorical(df[features]) if cat_features is None
                            else cat_features)
        self.results_: dict[str, Result] = {}
        self._counts = []
        logs = []
        current = list(features)
        for i, step in enumerate(self.steps, start=1):
            key = f"{i}_{step.name}"
            if verbose:
                print(f"[{i}/{len(self.steps)}] {step.name}: на входе {len(current)} признаков")
            res = step.run(df, target, current, cat_features)
            selected = [f for f in res.selected if f in current]
            dropped = [f for f in current if f not in selected]
            reasons = {}
            t = res.table
            if {"feature", "action", "reason"} <= set(t.columns):
                drops = t[t["action"] == "drop"].drop_duplicates("feature", keep="last")
                reasons = dict(zip(drops["feature"], drops["reason"], strict=True))
            logs += [{"step": key, "feature": f, "action": "drop",
                      "reason": reasons.get(f, "")} for f in dropped]
            self.results_[key] = res
            self._counts.append({"step": key, "n_in": len(current), "n_out": len(selected),
                                 "n_dropped": len(dropped)})
            current = selected
            if verbose:
                print(f"      → осталось {len(current)}")
        self.selected_ = current
        self.log_ = pd.DataFrame(logs, columns=["step", "feature", "action", "reason"])
        return self

    def summary(self) -> pd.DataFrame:
        """Таблица ``step, n_in, n_out, n_dropped``."""
        return pd.DataFrame(self._counts)

    def plot(self) -> go.Figure:
        """Воронка: сколько признаков осталось после каждого шага."""
        s = self.summary()
        labels = ["вход"] + s["step"].tolist()
        values = [int(s["n_in"].iloc[0])] + s["n_out"].tolist() if len(s) else []
        fig = go.Figure(go.Funnel(y=labels, x=values, textinfo="value+percent initial"))
        return style(fig, "Отбор признаков по шагам", height=120 + 60 * len(labels))

    def save(self, directory) -> None:
        """Сохраняет результаты всех шагов и общий лог в папку."""
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for key, res in self.results_.items():
            res.save(directory, prefix=key)
        write_csv(self.log_, directory / "selection_log.csv")
        write_csv(self.summary(), directory / "selection_summary.csv")
