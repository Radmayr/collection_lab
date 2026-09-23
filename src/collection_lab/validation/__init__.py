"""Валидация: стабильность признаков (plot_stab), метрики в динамике, отчёт по модели."""

from collection_lab.validation.dynamics import (
    feature_metric_dynamics,
    learning_curve,
    metric_dynamics,
)
from collection_lab.validation.report import ModelReport, compare_scores, model_report
from collection_lab.validation.stability import plot_stab, stability_table

__all__ = [
    "ModelReport",
    "compare_scores",
    "feature_metric_dynamics",
    "learning_curve",
    "metric_dynamics",
    "model_report",
    "plot_stab",
    "stability_table",
]
