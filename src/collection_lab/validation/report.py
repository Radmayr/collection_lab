"""Итоговый отчёт по модели на train / val / test."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from collection_lab.core.models import BaseModel
from collection_lab.data.split import DataSplit
from collection_lab.metrics.calibration import gain_chart, gain_chart_metrics, prob_to_logit
from collection_lab.metrics.classification import gini, ks, logloss, metrics_by_segment, roc_auc
from collection_lab.metrics.stability import PSI_THRESHOLDS, feature_psi, psi, psi_by_period
from collection_lab.plotting.theme import NEGATIVE, combine
from collection_lab.selection.importance import importance_plot
from collection_lab.utils.io import write_csv
from collection_lab.validation.dynamics import metric_dynamics


@dataclass
class ModelReport:
    """Результат :func:`model_report`.

    Attributes
    ----------
    metrics : pd.DataFrame
        По частям выборки: ``n, target_rate, mean_score, auc, gini, ks, logloss, score_psi``.
    calibration : pd.DataFrame
        Метрики gain chart по частям (``roc_auc, hl, n_buck, offset, full_calib_*``).
    feature_psi : pd.DataFrame
        PSI признаков train → последняя часть (test, если есть).
    feature_psi_by_period : pd.DataFrame | None
        PSI признаков по периодам последней части относительно train (если задан ``date_col``):
        ``feature, part, period, n, psi, status``. График — :meth:`plot_feature_psi`.
    segments : pd.DataFrame | None
        Метрики по сегментам в каждой части.
    figures : dict[str, go.Figure]
    """

    metrics: pd.DataFrame
    calibration: pd.DataFrame
    feature_psi: pd.DataFrame
    segments: pd.DataFrame | None = None
    dynamics: pd.DataFrame | None = None
    feature_psi_by_period: pd.DataFrame | None = None
    figures: dict[str, go.Figure] = field(default_factory=dict, repr=False)

    def plot_feature_psi(self, features: list[str] | None = None, *, top_k: int | None = None,
                         n_cols: int = 3, size: int = 350) -> go.Figure:
        """PSI каждого признака по периодам относительно train — по графику на признак.

        Parameters
        ----------
        features : list[str], optional
            Какие признаки показать (в этом порядке). По умолчанию все, по убыванию
            максимального PSI — самые нестабильные первыми.
        top_k : int, optional
            Оставить ``top_k`` первых признаков.
        n_cols, size
            Число графиков в строке и размер одного графика в пикселях.

        Пунктиры — пороги 0.1 (moderate) и 0.25 (significant).
        """
        fig = _plot_feature_psi(self.feature_psi_by_period, features, top_k, n_cols, size)
        from collection_lab.tracking.experiment import auto_figure

        auto_figure(fig, "feature_psi_by_period")
        return fig

    def show(self) -> None:
        """Показать таблицы и графики (в Jupyter)."""
        try:
            from IPython.display import display
        except ImportError:  # pragma: no cover
            display = print
        for name in ("metrics", "calibration", "segments", "feature_psi"):
            obj = getattr(self, name)
            if obj is not None:
                print(f"--- {name} ---")
                display(obj)
        for fig in self.figures.values():
            fig.show()

    def to_html(self, path: str | Path) -> Path:
        """Один HTML-файл со всеми таблицами и интерактивными графиками."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        parts = ["<html><head><meta charset='utf-8'><title>Model report</title>",
                 "<style>body{font-family:sans-serif;margin:24px} table{border-collapse:"
                 "collapse} td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}"
                 "</style></head><body><h1>Model report</h1>"]
        for name in ("metrics", "calibration", "segments", "feature_psi"):
            obj = getattr(self, name)
            if obj is not None:
                parts.append(f"<h2>{name}</h2>" + obj.to_html(float_format=lambda v: f"{v:.4f}"))
        for i, (name, fig) in enumerate(self.figures.items()):
            parts.append(f"<h2>{name}</h2>")
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
        parts.append("</body></html>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path

    def save(self, directory: str | Path, *, to_clearml: bool | None = None) -> Path:
        """Таблицы в csv + ``report.html`` в папку.

        ``to_clearml``: ``None`` — дублировать в ClearML при активном ``Experiment(clearml=True)``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("metrics", "calibration", "segments", "feature_psi", "dynamics",
                     "feature_psi_by_period"):
            obj = getattr(self, name)
            if obj is not None:
                write_csv(obj, directory / f"report_{name}.csv", index=True)
        self.to_html(directory / "report.html")
        from collection_lab.core.results import _maybe_send

        _maybe_send(self, directory.name or "report", to_clearml)
        return directory


def _feature_psi_by_period(train: pd.DataFrame, data: pd.DataFrame, features: list[str],
                           cat_features, date_col: str, freq: str, part: str) -> pd.DataFrame:
    cat_features = set(cat_features or ())
    tables = []
    for f in features:
        t = psi_by_period(data, f, date_col, freq=freq, reference=train[f],
                          categorical=True if f in cat_features else None)
        t.insert(0, "feature", f)
        t.insert(1, "part", part)
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def _plot_feature_psi(table: pd.DataFrame | None, features: list[str] | None,
                      top_k: int | None, n_cols: int, size: int) -> go.Figure:
    if table is None:
        raise ValueError("PSI по периодам не посчитан: передайте date_col в model_report.")
    if features is None:
        features = table.groupby("feature")["psi"].max().sort_values(ascending=False).index
    features = list(features)[:top_k]
    unknown = [f for f in features if f not in set(table["feature"])]
    if unknown:
        raise ValueError(f"Нет в отчёте: {unknown}")
    figs = []
    for f in features:
        t = table[table["feature"] == f]
        figs.append(go.Figure(go.Scatter(
            x=t["period"], y=t["psi"], mode="lines+markers", name=f, customdata=t["n"],
            hovertemplate="%{x|%Y-%m}: PSI %{y:.3f}, n = %{customdata}<extra></extra>")))
    part = table["part"].iloc[0]
    fig = combine(figs, features, n_cols=min(n_cols, len(figs)), size=size,
                  title=f"PSI признаков по периодам: {part} относительно train")
    for level, color in zip(PSI_THRESHOLDS, ("orange", NEGATIVE), strict=True):
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1)
    fig.update_yaxes(rangemode="tozero")
    return fig


def model_report(
    model: BaseModel,
    split: DataSplit,
    *,
    features: list[str] | None = None,
    date_col: str | None = None,
    segment: str | None = None,
    n_buckets: int = 20,
    freq: str = "M",
) -> ModelReport:
    """Отчёт по обученной модели на всех частях :class:`DataSplit`.

    Parameters
    ----------
    model : BaseModel
        Обученный адаптер (``collection_lab.modeling.train_model`` или ``core.make_model``).
    features : list[str], optional
        По умолчанию — признаки модели.
    date_col : str, optional
        Колонка даты — добавляет динамику AUC по периодам и PSI признаков по периодам
        последней части (test, если есть) относительно train.
    segment : str, optional
        Колонка сегмента — метрики по сегментам.
    n_buckets : int
        Бакетов в gain chart.
    freq : str
        Период для динамики: ``"M"``, ``"Q"``, ...
    """
    features = list(model.features_ if features is None else features)
    target = split.target
    parts = dict(split.items())
    scores = {name: model.predict(df[features]) for name, df in parts.items()}
    base_scores = scores["train"]

    rows, cal_items, cal_names, gain_figs, seg_tables = [], [], [], [], []
    for name, df in parts.items():
        y, p = df[target].to_numpy(), scores[name]
        rows.append({"part": name, "n": len(df), "target_rate": y.mean(), "mean_score": p.mean(),
                     "auc": roc_auc(y, p), "gini": gini(y, p), "ks": ks(y, p),
                     "logloss": logloss(y, p),
                     "score_psi": psi(base_scores, p) if name != "train" else 0.0})
        fig, m = gain_chart(prob_to_logit(p), y, n_buckets=n_buckets, calib=True,
                            logit_name=f"Model {name}", return_plotly_fig=True)
        cal_items.append(m)
        cal_names.append(name)
        gain_figs.append(fig)
        if segment is not None:
            t = metrics_by_segment(y, p, df[segment], metrics=("auc", "gini"), add_total=False)
            seg_tables.append(t.assign(part=name).reset_index())

    last = list(parts)[-1]
    fpsi = feature_psi(parts["train"][features], parts[last][features],
                       cat_features=model.cat_features_)
    figures = {
        "gain_charts": combine(gain_figs, [f"Model {n}" for n in cal_names], n_cols=len(gain_figs),
                               size=450),
        "feature_importance": importance_plot(model.feature_importance("gain")),
    }
    dyn = psi_dyn = None
    if date_col is not None:
        labeled = pd.concat([df.assign(_score=scores[n], _part=n) for n, df in parts.items()])
        res = metric_dynamics(labeled, target, "_score", date_col, "_part", freq=freq)
        dyn = res.table
        figures["auc_dynamics"] = res.plot(title="AUC по периодам (train / val / test)")
        psi_dyn = _feature_psi_by_period(parts["train"], parts[last], features,
                                         model.cat_features_, date_col, freq, last)
        figures["feature_psi_by_period"] = _plot_feature_psi(psi_dyn, None, None, 3, 350)
    seg_df = None
    if seg_tables:
        seg_df = pd.concat(seg_tables, ignore_index=True).set_index(["part", "segment"])
    return ModelReport(
        metrics=pd.DataFrame(rows).set_index("part"),
        calibration=gain_chart_metrics(cal_items, cal_names),
        feature_psi=fpsi,
        segments=seg_df,
        dynamics=dyn,
        feature_psi_by_period=psi_dyn,
        figures=figures,
    )


def compare_scores(y_true, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Сравнение нескольких скоров на одной выборке: AUC, Gini, KS."""
    return pd.DataFrame([{"score": k, "auc": roc_auc(y_true, v), "gini": gini(y_true, v),
                          "ks": ks(y_true, v)} for k, v in scores.items()]).set_index("score")
