"""Калибровка: gain chart (как ``risk_instruments.model_analysis.gain_chart``), Хосмер–Лемешоу.

Восстановлено по выводу исходной функции и сверено с её числами:

- бакеты — квантили логита (одинаковые значения не разрываются), бакет 1 — самый рисковый;
- доверительный интервал badrate — нормальное приближение, 99% (z = 2.576), симметричный;
- «Model Prediction» — средняя предсказанная вероятность в бакете;
- ``offset`` — сдвиг ``b`` в ``sigmoid(logit + b)``, при котором средний прогноз равен
  среднему таргету (калибровка на офсет);
- ``full calib coefs`` — логистическая регрессия таргета на логит: ``sigmoid(k·logit + b)``;
- ``hl`` — статистика Хосмера–Лемешоу по бакетам графика;
- ``n_buck`` — размер бакета с минимальным логитом.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import chi2, norm

from collection_lab.metrics.classification import roc_auc

Z_99 = float(norm.ppf(0.995))


def prob_to_logit(p) -> np.ndarray:
    """Логит вероятности (с защитой от 0 и 1)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def calibration_offset(logit, target) -> float:
    """Сдвиг ``b``: ``mean(sigmoid(logit + b)) == mean(target)`` (MLE интерсепта при офсете)."""
    logit = np.asarray(logit, dtype=float)
    rate = float(np.mean(target))
    if rate <= 0 or rate >= 1:
        return float("nan")
    f = lambda b: expit(logit + b).mean() - rate  # noqa: E731
    return float(brentq(f, -50, 50))


def full_calibration(logit, target) -> tuple[float, float]:
    """Коэффициенты ``(k, b)`` логистической регрессии ``target ~ k·logit + b``
    (без регуляризации)."""
    logit = np.asarray(logit, dtype=float)
    y = np.asarray(target, dtype=float)

    def nll(w):
        z = w[0] * logit + w[1]
        return float(np.sum(np.logaddexp(0, z) - y * z))

    def grad(w):
        r = expit(w[0] * logit + w[1]) - y
        return np.array([np.sum(r * logit), np.sum(r)])

    res = minimize(nll, x0=np.array([1.0, 0.0]), jac=grad, method="BFGS")
    return float(res.x[0]), float(res.x[1])


def hosmer_lemeshow(y_true, y_prob, buckets) -> tuple[float, float]:
    """Статистика Хосмера–Лемешоу и p-value (df = число бакетов − 2).

    ``HL = Σ (O_g − E_g)² / (E_g · (1 − E_g / n_g))``, где ``E_g = Σ p_i`` в бакете.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_prob, float),
                       "b": np.asarray(buckets)})
    g = df.groupby("b").agg(n=("y", "size"), o=("y", "sum"), e=("p", "sum"))
    denom = g["e"] * (1 - g["e"] / g["n"])
    stat = float((((g["o"] - g["e"]) ** 2) / denom.where(denom > 0)).sum())
    dof = max(len(g) - 2, 1)
    return stat, float(chi2.sf(stat, dof))


def logit_buckets(logit, n_buckets: int) -> np.ndarray:
    """Номер бакета (1 = максимальный логит) по квантилям логита без разрыва одинаковых значений."""
    s = pd.Series(np.asarray(logit, dtype=float))
    codes = pd.qcut(s, n_buckets, labels=False, duplicates="drop").to_numpy()
    n_real = int(np.nanmax(codes)) + 1
    return n_real - codes.astype(int)


def gain_chart_table(logit, target, n_buckets: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Расчётная часть gain chart для одной группы.

    Returns
    -------
    table : pd.DataFrame
        ``bucket, n, n_target, badrate, ci, pred, calibrated, full_calibrated, logit_min,
        logit_max``, бакет 1 — самый рисковый.
    metrics : dict
        ``roc_auc, hl, n_buck, offset, full calib coefs`` — как в ``risk_instruments``.
    """
    logit = np.asarray(logit, dtype=float)
    y = np.asarray(target, dtype=float)
    if np.isnan(logit).any() or np.isnan(y).any():
        raise ValueError("В логитах или таргете есть пропуски.")
    buckets = logit_buckets(logit, n_buckets)
    offset = calibration_offset(logit, y)
    k, b = full_calibration(logit, y)
    df = pd.DataFrame({"bucket": buckets, "y": y, "logit": logit, "p": expit(logit),
                       "p_cal": expit(logit + offset), "p_full": expit(k * logit + b)})
    t = df.groupby("bucket").agg(
        n=("y", "size"), n_target=("y", "sum"), badrate=("y", "mean"), pred=("p", "mean"),
        calibrated=("p_cal", "mean"), full_calibrated=("p_full", "mean"),
        logit_min=("logit", "min"), logit_max=("logit", "max")).reset_index()
    t["ci"] = Z_99 * np.sqrt(t["badrate"] * (1 - t["badrate"]) / t["n"])
    hl, _ = hosmer_lemeshow(y, df["p"], buckets)
    metrics = {
        "roc_auc": roc_auc(y, logit),
        "hl": hl,
        "n_buck": int(t.loc[t["bucket"].idxmax(), "n"]),
        "offset": offset,
        "full calib coefs": {"offset": b, "coef": k},
    }
    return t, metrics


def _add_group_traces(fig, t: pd.DataFrame, row: int, col: int, calib: bool, calib_full: bool,
                      first: bool) -> None:
    x = t["bucket"].to_numpy()
    common = {"row": row, "col": col}
    fig.add_scatter(x=x, y=t["pred"], name="Model Prediction", legendgroup="group1",
                    showlegend=first, hovertemplate="%{y:.4f}", line={"color": "#EF553B"},
                    line_shape="spline",
                    marker={"color": "#EF553B", "size": 5, "line": {"color": "black", "width": 1}},
                    **common)
    fig.add_bar(x=x, y=t["badrate"], name="Badrate", legendgroup="group2", showlegend=first,
                hovertemplate="%{y:.4f}", marker_color="#636EFA", **common)
    if calib:
        fig.add_scatter(x=x, y=t["calibrated"], name="Calibrated Model", legendgroup="group3",
                        showlegend=first, hovertemplate="%{y:.4f}", line_shape="spline",
                        line={"color": "#00CC96"},
                        marker={"color": "#00CC96", "size": 5,
                                "line": {"color": "black", "width": 1}}, **common)
    if calib_full:
        fig.add_scatter(x=x, y=t["full_calibrated"], name="Full Calibrated Model",
                        legendgroup="group4", showlegend=first, hovertemplate="%{y:.4f}",
                        line_shape="spline", line={"color": "#AB63FA"},
                        marker={"color": "#AB63FA", "size": 5,
                                "line": {"color": "black", "width": 1}}, **common)
    fig.add_scatter(x=x, y=t["badrate"], mode="markers", name="Confidence Interval",
                    legendgroup="group5", showlegend=first, hoverinfo="skip",
                    marker={"size": 1, "color": "black"},
                    error_y={"type": "data", "array": t["ci"].round(4), "symmetric": True,
                             "color": "black"}, **common)


def gain_chart(
    values,
    target,
    groups=None,
    n_buckets: int | Sequence[int] | None = None,
    subplots_shape: tuple[int, int] | None = None,
    logit_name: str = "logit",
    subplot_title_size: int = 10,
    calib: bool = False,
    calib_full: bool = False,
    interactive_plot: bool = True,
    save_plot: bool = False,
    path_to_save: str = "images",
    plots_nm: list[str] | None = None,
    to_clearml: bool = False,
    return_plotly_fig: bool = False,
    return_dict: bool = True,
):
    """Gain chart — попадание модели в вероятности по бакетам логита.

    Повторяет интерфейс ``risk_instruments.model_analysis.gain_chart``. На вход подаётся
    **логит** (``collection_lab.metrics.prob_to_logit(p)`` для вероятностей).

    Parameters
    ----------
    values, target : array-like
        Логит и таргет (без пропусков).
    groups : array-like, optional
        Группа для каждого объекта — каждая рисуется в своём подграфике.
    n_buckets : int | list[int], optional
        Число бакетов (одно на все группы или список по группам); по умолчанию 10.
    subplots_shape : (rows, cols), optional
        Сетка подграфиков; по умолчанию — все в одну строку.
    calib, calib_full : bool
        Добавить линию калибровки на офсет / полной калибровки.
    interactive_plot : bool
        Показать график (если не возвращается объект фигуры). ``False`` — статичный.
    save_plot, path_to_save, plots_nm
        Сохранить график в ``path_to_save/plots_nm[0].html``.
    to_clearml : bool
        Залогировать график в текущую задачу ClearML под именем ``plots_nm[0]``.
    return_plotly_fig, return_dict : bool
        Что вернуть: ``dict`` с метриками, фигуру или кортеж ``(fig, dict)``.

    Returns
    -------
    dict | go.Figure | tuple[go.Figure, dict]
        Словарь ``{группа: {"roc_auc", "hl", "n_buck", "offset", "full calib coefs"}}``;
        без ``groups`` группа называется ``"Group"``.
    """
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    if groups is None:
        group_arr = np.array(["Group"] * len(values), dtype=object)
        group_names = ["Group"]
    else:
        group_arr = np.asarray(groups, dtype=object)
        group_names = list(pd.unique(group_arr))
    n_groups = len(group_names)
    if n_buckets is None:
        n_buckets = [10] * n_groups
    elif isinstance(n_buckets, int):
        n_buckets = [n_buckets] * n_groups
    rows, cols = subplots_shape or (1, n_groups)

    tables, metrics = {}, {}
    for g, nb in zip(group_names, n_buckets, strict=True):
        mask = group_arr == g
        tables[g], metrics[g] = gain_chart_table(values[mask], target[mask], nb)

    def title(g) -> str:
        m = metrics[g]
        name = "" if isinstance(g, (int, np.integer)) else f"<b>{g}</b><br>"
        return (f"{name}<b>ROC AUC = {m['roc_auc']:.3f}</b><br><b>offset = {m['offset']:.2f}</b>"
                f"<br>{m['n_buck']} objects in bucket")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[title(g) for g in group_names])
    for i, g in enumerate(group_names):
        _add_group_traces(fig, tables[g], i // cols + 1, i % cols + 1, calib, calib_full, i == 0)
    fig.update_annotations(font_size=subplot_title_size)
    fig.update_xaxes(dtick=1)
    fig.update_layout(title={"text": f"Gain chart for {logit_name}", "x": 0.45, "y": 0.99},
                      hovermode="x unified", height=500 * rows, width=600 * cols,
                      margin={"t": 100, "b": 50, "l": 50, "r": 50})

    name = plots_nm[0] if plots_nm else f"gain_chart_{logit_name}".strip().replace(" ", "_")
    if save_plot:
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        fig.write_html(Path(path_to_save) / f"{name}.html")
    if to_clearml:
        from collection_lab.tracking.clearml import report_figure

        report_figure(fig, title=name)
    if not return_plotly_fig:
        fig.show(config=None if interactive_plot else {"staticPlot": True})
    if return_plotly_fig and return_dict:
        return fig, metrics
    if return_plotly_fig:
        return fig
    return metrics if return_dict else None


def gain_chart_metrics(metrics: Sequence[dict], segments: Sequence[str]) -> pd.DataFrame:
    """Сводная таблица метрик нескольких ``gain_chart`` (бывший ``show_gainchart_metrics``).

    ``metrics`` — список словарей, которые вернул ``gain_chart`` (по одному на выборку).
    """
    rows = []
    for item, segment in zip(metrics, segments, strict=True):
        for group, m in item.items():
            rows.append({
                "segment": segment if len(item) == 1 else f"{segment} / {group}",
                "roc_auc": round(m["roc_auc"], 4), "hl": round(m["hl"], 1),
                "n_buck": int(m["n_buck"]), "offset": round(m["offset"], 4),
                "full_calib_coef": round(m["full calib coefs"]["coef"], 4),
                "full_calib_offset": round(m["full calib coefs"]["offset"], 4),
            })
    return pd.DataFrame(rows).set_index("segment")
