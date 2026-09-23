"""Метрики: AUC/Gini/KS, стабильность (PSI/CSI), калибровка (gain chart, HL), WoE/IV."""

from collection_lab.metrics.classification import (
    METRICS,
    Metric,
    get_metric,
    gini,
    ks,
    logloss,
    metrics_by_segment,
    roc_auc,
)
from collection_lab.metrics.stability import (
    feature_psi,
    psi,
    psi_by_period,
    psi_label,
    psi_table,
    quantile_edges,
)

__all__ = [
    "METRICS",
    "Metric",
    "feature_psi",
    "get_metric",
    "gini",
    "ks",
    "logloss",
    "metrics_by_segment",
    "psi",
    "psi_by_period",
    "psi_label",
    "psi_table",
    "quantile_edges",
    "roc_auc",
]
