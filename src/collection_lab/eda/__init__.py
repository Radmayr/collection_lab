"""Разведочный анализ: обзор таблицы, распределения, динамика таргета, винтажи транзакций."""

from collection_lab.eda.distributions import plot_distribution, plot_target_rate_by_bins
from collection_lab.eda.overview import overview, target_summary
from collection_lab.eda.target import target_dynamics, wilson_ci
from collection_lab.eda.vintage import (
    add_days_since,
    eda_transactions,
    maturation_transactions,
    plot_vintage,
    vintage_by_type,
)

__all__ = [
    "add_days_since",
    "eda_transactions",
    "maturation_transactions",
    "overview",
    "plot_distribution",
    "plot_target_rate_by_bins",
    "plot_vintage",
    "target_dynamics",
    "target_summary",
    "vintage_by_type",
    "wilson_ci",
]
