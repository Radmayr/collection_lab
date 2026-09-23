"""Отбор признаков: фильтры, корреляции, однофакторный анализ, важности, RFE, stepwise."""

from collection_lab.selection.correlation import correlation_filter, correlation_groups
from collection_lab.selection.filters import quality_filter
from collection_lab.selection.importance import (
    cumulative_importance_selection,
    drop_column_importance,
    importance_plot,
    permutation_importance,
)
from collection_lab.selection.pipeline import (
    RFE,
    CorrelationFilter,
    CumulativeImportance,
    Custom,
    QualityFilter,
    SelectionPipeline,
    Step,
    UnivariateFilter,
)
from collection_lab.selection.rfe import rfe
from collection_lab.selection.stepwise import (
    backward_elimination,
    forward_addition,
    incremental_feature_eval,
)
from collection_lab.selection.univariate import direct_auc, univariate_filter, univariate_scores

__all__ = [
    "RFE",
    "CorrelationFilter",
    "CumulativeImportance",
    "Custom",
    "QualityFilter",
    "SelectionPipeline",
    "Step",
    "UnivariateFilter",
    "backward_elimination",
    "correlation_filter",
    "correlation_groups",
    "cumulative_importance_selection",
    "direct_auc",
    "drop_column_importance",
    "forward_addition",
    "importance_plot",
    "incremental_feature_eval",
    "permutation_importance",
    "quality_filter",
    "rfe",
    "univariate_filter",
    "univariate_scores",
]
