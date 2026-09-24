"""Стабильность признака во времени — повторение ``risk_instruments.feature_analysis.plot_stab``.

Две фигуры по две панели:

1. WoE по бакетам от времени (± 1 стандартная ошибка) | распределение по бакетам;
2. Bad rate по бакетам от времени (± 1 стандартная ошибка) | IV от времени.

Бакеты строятся один раз по всей выборке (:func:`~collection_lab.metrics.binning.
quantile_buckets` — одинаковые значения не разрываются), подписи — ``'var in [min, max]'``,
легенда отсортирована по среднему WoE. Для непрерывного таргета вместо WoE — медиана с
доверительным интервалом по квантилям, вместо IV — R² (corr²).

Отличие от оригинала: на непрерывных признаках граница бакета может сдвинуться на одно
соседнее значение (точный алгоритм разбиения исходной функции недоступен).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collection_lab.metrics.binning import (
    bucket_labels,
    iv_from_auc,
    quantile_buckets,
    uniform_buckets,
    woe_ci,
    woe_iv_table,
)
from collection_lab.metrics.classification import roc_auc
from collection_lab.metrics.stability import psi_from_counts
from collection_lab.plotting.theme import PALETTE

# Полосы на графиках plot_stab — ±1 стандартная ошибка (сверено с выводом исходной функции:
# WoE и badrate совпадают, ширина полос соответствует z ≈ 1).
Z_STAB = 1.0


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _apply_transform(x: pd.Series, transform: dict[str, Any] | None) -> pd.Series:
    if not transform:
        return x
    x = x.clip(lower=transform.get("lower"), upper=transform.get("upper"))
    if transform.get("power") is not None:
        x = np.sign(x) * np.abs(x) ** transform["power"]
    if transform.get("log"):
        x = np.log(x.where(x > 0))
    return x


def stability_table(
    values,
    target,
    time,
    n_buckets: int,
    *,
    feature_nm: str = "var",
    period: str | None = None,
    transform: dict[str, Any] | None = None,
    bins_method: str = "quantile",
    avoid_null_mode: bool = True,
    null_bucket: bool = False,
    quantiles: tuple[float, float] = (0.45, 0.55),
    binary_target: bool = True,
    iv_type: str = "IV_empirical",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Расчётная часть :func:`plot_stab`.

    Returns
    -------
    buckets : pd.DataFrame
        По (период, бакет): ``period, bucket, label, n, share, bad, good, bad_rate,
        bad_rate_ci, woe, woe_ci`` (для непрерывного таргета — ``median, ci_low, ci_high``).
    periods : pd.DataFrame
        По периоду: ``period, n, iv`` (или ``r2``), ``psi`` к предыдущему периоду.
    """
    raw = pd.Series(np.asarray(values))  # для подписей бакетов в исходном типе (int / float)
    x = pd.Series(np.asarray(values, dtype="float64"))
    y = pd.Series(np.asarray(target, dtype="float64"))
    t = pd.to_datetime(pd.Series(np.asarray(time)), errors="coerce", format="mixed")
    if period:
        t = t.dt.to_period(period).dt.to_timestamp()
    x = _apply_transform(x, transform)
    if transform:
        raw = x

    keep = t.notna() & (y.notna() if binary_target else True)
    n_y_null = int((y.isna() & t.notna()).sum())
    if n_y_null and verbose:
        print(f"{n_y_null} nulls in target have been deleted.")
    x_null = x.isna()
    if not null_bucket or not binary_target:
        n_null = int((x_null & keep).sum())
        if n_null and verbose:
            print(f"{n_null} nulls have been deleted.")
        keep &= ~x_null
    x, y, t, raw = (s[keep].reset_index(drop=True) for s in (x, y, t, raw))

    make = quantile_buckets if bins_method == "quantile" else uniform_buckets
    if bins_method not in ("quantile", "dense"):
        raise ValueError("bins_method должен быть 'quantile' или 'dense'")
    buckets = make(x, n_buckets)
    labels = bucket_labels(raw.to_numpy(), buckets, feature_nm)
    data = pd.DataFrame({"period": t, "bucket": buckets, "x": x, "y": y})

    rows, prows = [], []
    prev_counts = None
    for p, part in data.groupby("period", sort=True):
        n_p = len(part)
        counts = part["bucket"].value_counts()
        prow: dict[str, Any] = {"period": p, "n": n_p}
        if binary_target:
            wt = woe_iv_table(part["x"], part["y"], part["bucket"].to_numpy())
            if not avoid_null_mode:  # без подстановки: бакеты с нулевыми счётчиками → NaN
                bad0 = (wt["bad"] == 0) | (wt["good"] == 0)
                wt.loc[bad0, ["woe", "iv"]] = np.nan
            wt["woe_ci"] = woe_ci(wt["bad"], wt["good"], z=Z_STAB)
            wt["bad_rate_ci"] = Z_STAB * np.sqrt(wt["bad_rate"] * (1 - wt["bad_rate"]) / wt["n"])
            wt["share"] = wt["n"] / n_p
            wt["period"] = p
            rows.append(wt)
            if iv_type == "IV_auc":
                a = roc_auc(part["y"], part["bucket"].map(wt.set_index("bucket")["woe"]))
                prow["iv"] = iv_from_auc(a)
            else:
                prow["iv"] = float(wt["iv"].sum())
        else:
            g = part.groupby("bucket")["y"]
            wt = pd.DataFrame({
                "n": g.size(), "median": g.median(),
                "ci_low": g.quantile(quantiles[0]), "ci_high": g.quantile(quantiles[1]),
            }).reset_index()
            wt["share"] = wt["n"] / n_p
            wt["period"] = p
            rows.append(wt)
            prow["r2"] = float(part["x"].corr(part["y"]) ** 2) if n_p > 2 else np.nan
        prow["psi"] = psi_from_counts(prev_counts, counts) if prev_counts is not None else np.nan
        prev_counts = counts
        prows.append(prow)

    table = pd.concat(rows, ignore_index=True)
    table["label"] = table["bucket"].map(labels)
    return table, pd.DataFrame(prows)


def plot_stab(
    values,
    target,
    time,
    n_buckets: int,
    feature_nm: str = "var",
    target_nm: str = "target",
    period: str | None = None,
    transform: dict[str, Any] | None = None,
    bins_method: str = "quantile",
    avoid_null_mode: bool = True,
    null_bucket: bool = False,
    quantiles: tuple[float, float] = (0.45, 0.55),
    binary_target: bool = True,
    return_plotly_fig: bool = False,
    interactive_plot: bool = False,
    line_dt: str | list[str] | dict[str, str] | None = None,
    metric_ylim_zero: bool = False,
    save_plot: bool = False,
    path_to_save: str = "images",
    plots_nm: list[str] | None = None,
    add_psi: bool = False,
    size_psi: float = 3.0,
    to_clearml: bool = False,
    ci_opacity: float = 0.5,
    iv_type: str = "IV_empirical",
) -> list[go.Figure] | None:
    """Разбивает признак на бакеты и строит во времени WoE, доли бакетов, badrate и IV.

    Интерфейс повторяет ``risk_instruments.feature_analysis.plot_stab``.

    Parameters
    ----------
    values, target, time : array-like
        Признак, таргет и время. Строки без таргета удаляются с сообщением.
    n_buckets : int
        Число бакетов.
    period : str, optional
        Округление времени: ``"M"`` — месяц, ``"Q"`` — квартал.
    transform : dict, optional
        ``dict(lower=-3, upper=5, power=0.5, log=False)`` — обрезка, степень, логарифм.
    bins_method : {"quantile", "dense"}
        Квантильные бакеты или равные по ширине.
    avoid_null_mode : bool
        Подставлять малое число вместо нулевых счётчиков бакета (иначе WoE = NaN).
    null_bucket : bool
        Отдельный бакет для пропусков (иначе пропуски удаляются с сообщением).
    quantiles : (float, float)
        Квантили ДИ медианы для непрерывного таргета.
    binary_target : bool
        ``False`` — режим непрерывного таргета (медиана, R²).
    line_dt : str | list | dict
        Вертикальные линии: дата, список дат или ``{дата: подпись}``.
    add_psi, size_psi
        Добавить PSI между соседними периодами на график распределения; ``size_psi`` —
        масштаб оси PSI.
    iv_type : {"IV_empirical", "IV_auc"}
        IV по бакетам или через ROC AUC.
    return_plotly_fig : bool
        Вернуть список из двух ``go.Figure`` вместо показа.

    Returns
    -------
    list[go.Figure] | None
    """
    table, periods = stability_table(
        values, target, time, n_buckets, feature_nm=feature_nm, period=period,
        transform=transform, bins_method=bins_method, avoid_null_mode=avoid_null_mode,
        null_bucket=null_bucket, quantiles=quantiles, binary_target=binary_target,
        iv_type=iv_type)
    suffix = f"<br>({feature_nm}, {target_nm})"

    metric_col = "woe" if binary_target else "median"
    order = (table.groupby("bucket")[metric_col].mean().sort_values().index.tolist())
    colors = {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(sorted(order))}
    labels = table.drop_duplicates("bucket").set_index("bucket")["label"].to_dict()

    def band(fig, sub, col_mid, lo, hi, b, col):
        c = colors[b]
        fig.add_scatter(x=pd.concat([sub["period"], sub["period"][::-1]]),
                        y=pd.concat([lo, hi[::-1]]), fill="toself", mode="lines",
                        line={"width": 0}, fillcolor=_hex_rgba(c, ci_opacity * 0.9),
                        hoverinfo="skip", showlegend=False, legendgroup=str(b), row=1, col=col)
        fig.add_scatter(x=sub["period"], y=sub[col_mid], mode="lines+markers", name=labels[b],
                        line={"color": c}, legendgroup=str(b), showlegend=col == 1,
                        row=1, col=col)

    # --- фигура 1: WoE (или медиана) + распределение ---
    left_title = ("WoE по бакетам от времени" if binary_target
                  else "Медиана таргета по бакетам от времени")
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=(
        left_title + suffix, "Распределение по бакетам от времени" + suffix),
        specs=[[{}, {"secondary_y": add_psi}]])
    for b in order[::-1]:
        sub = table[table["bucket"] == b].sort_values("period")
        if binary_target:
            band(fig1, sub, "woe", sub["woe"] - sub["woe_ci"], sub["woe"] + sub["woe_ci"], b, 1)
        else:
            band(fig1, sub, "median", sub["ci_low"], sub["ci_high"], b, 1)
    shares = table.pivot_table(index="period", columns="bucket", values="share",
                               aggfunc="sum").fillna(0)
    for b in order:
        if b in shares.columns:
            fig1.add_scatter(x=shares.index, y=shares[b], mode="lines", stackgroup="share",
                             line={"width": 0.5, "color": colors[b]},
                             fillcolor=_hex_rgba(colors[b], 0.55), name=labels[b],
                             legendgroup=str(b), showlegend=False, row=1, col=2)
    if add_psi:
        fig1.add_scatter(x=periods["period"], y=periods["psi"], mode="lines+markers", name="PSI",
                         line={"color": "black", "dash": "dot"}, row=1, col=2, secondary_y=True)
        top = float(np.nanmax(periods["psi"])) if periods["psi"].notna().any() else 0.1
        fig1.update_yaxes(range=[0, max(top, 1e-6) * size_psi], title_text="PSI", row=1, col=2,
                          secondary_y=True)
    fig1.update_yaxes(title_text="WoE" if binary_target else "median", row=1, col=1)
    fig1.update_yaxes(range=[0, 1], row=1, col=2, secondary_y=False)

    # --- фигура 2: badrate + IV (или R²) ---
    metric_name = iv_type if binary_target else "R^2"
    right_title = (f"Зависимость {metric_name} от времени" + suffix)
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=(
        ("Bad Rate по бакетам от времени" if binary_target else "Доля бакета") + suffix,
        right_title))
    for b in order[::-1]:
        sub = table[table["bucket"] == b].sort_values("period")
        if binary_target:
            band(fig2, sub, "bad_rate", (sub["bad_rate"] - sub["bad_rate_ci"]).clip(lower=0),
                 (sub["bad_rate"] + sub["bad_rate_ci"]).clip(upper=1), b, 1)
        else:
            fig2.add_scatter(x=sub["period"], y=sub["share"], mode="lines+markers",
                             name=labels[b], line={"color": colors[b]}, row=1, col=1)
    metric_col_p = "iv" if binary_target else "r2"
    fig2.add_scatter(x=periods["period"], y=periods[metric_col_p], mode="lines",
                     line={"color": "blue"}, name=metric_name, showlegend=False, row=1, col=2)
    fig2.update_yaxes(title_text="Bad Rate" if binary_target else "share", row=1, col=1)
    if metric_ylim_zero:
        fig2.update_yaxes(rangemode="tozero", row=1, col=2)

    lines = ({line_dt: None} if isinstance(line_dt, str) else
             dict.fromkeys(line_dt) if isinstance(line_dt, list) else (line_dt or {}))
    for fig in (fig1, fig2):
        for dt, text in lines.items():
            x = pd.Timestamp(dt).timestamp() * 1000  # plotly ждёт миллисекунды для дат
            fig.add_vline(x=x, line_dash="dash", line_color="black",
                          annotation_text=text or "", annotation_position="top")
        n_periods = periods["period"].nunique()
        fig.update_xaxes(tickangle=-45, tickformat="%b %-d, %Y",
                         tickvals=periods["period"] if n_periods <= 36 else None)
        fig.update_layout(height=500, width=1500, hovermode="closest",
                          legend={"x": -0.25, "y": 1, "bordercolor": "black", "borderwidth": 1,
                                  "traceorder": "normal"},
                          margin={"l": 220, "r": 30, "t": 100, "b": 80})

    figs = [fig1, fig2]
    names = plots_nm or [f"stab_{feature_nm}_1", f"stab_{feature_nm}_2"]
    from collection_lab.tracking.experiment import auto_figure

    for fig, name in zip(figs, names, strict=False):
        auto_figure(fig, name)
    if save_plot:
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        for fig, name in zip(figs, names, strict=False):
            fig.write_html(Path(path_to_save) / f"{name}.html")
    if to_clearml:
        from collection_lab.tracking.clearml import report_figure

        for fig, name in zip(figs, names, strict=False):
            report_figure(fig, title=name)
    if return_plotly_fig:
        return figs
    for fig in figs:
        fig.show(config=None if interactive_plot else {"staticPlot": True})
    return None
