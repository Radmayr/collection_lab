"""Метрики: AUC/Gini/KS, стабильность (PSI/CSI), калибровка (gain chart, HL), WoE/IV."""

from collection_lab.metrics.binning import (
    information_value,
    iv_table,
    quantile_buckets,
    woe_iv_table,
)
from collection_lab.metrics.calibration import (
    calibration_offset,
    full_calibration,
    gain_chart,
    gain_chart_metrics,
    gain_chart_table,
    hosmer_lemeshow,
    prob_to_logit,
)
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
    psi_from_counts,
    psi_label,
    psi_table,
    quantile_edges,
)

__all__ = [
    "METRICS",
    "Metric",
    "calibration_offset",
    "feature_psi",
    "full_calibration",
    "gain_chart",
    "gain_chart_metrics",
    "gain_chart_table",
    "get_metric",
    "gini",
    "hosmer_lemeshow",
    "information_value",
    "iv_table",
    "ks",
    "logloss",
    "metrics_by_segment",
    "prob_to_logit",
    "psi",
    "psi_by_period",
    "psi_from_counts",
    "psi_label",
    "psi_table",
    "quantile_buckets",
    "quantile_edges",
    "roc_auc",
    "woe_iv_table",
]
