"""Винтажный анализ транзакций и быстрый EDA транзакционной таблицы."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collection_lab.plotting.theme import color, style

MILESTONES = [(30, "1м"), (90, "3м"), (180, "6м"), (365, "12м"), (730, "24м"), (1095, "36м")]


def add_days_since(
    df: pd.DataFrame, event_col: str, base_col: str, name: str = "tx_days"
) -> pd.DataFrame:
    """Добавляет колонку ``name`` = число дней от ``base_col`` до ``event_col`` (по датам)."""
    out = df.copy()
    event = pd.to_datetime(out[event_col], errors="coerce", format="mixed").dt.normalize()
    base = pd.to_datetime(out[base_col], errors="coerce", format="mixed").dt.normalize()
    out[name] = (event - base).dt.days
    return out


def maturation_transactions(
    df: pd.DataFrame,
    sample: str,
    *,
    client_id: str = "contract_number",
    tx_days_col: str = "tx_days",
    tx_amount_col: str = "transaction_amt",
    balance_col: str = "rtk_balance",
    retro_dt_col: str = "rtk_send_date",
    horizon_days: int = 365,
    step: int = 1,
    processed_dt=None,
    population: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Накопительная динамика транзакций по дням от ретро-даты (винтаж).

    Горизонт урезается, если с последней ретро-даты до ``processed_dt`` прошло меньше
    ``horizon_days`` дней (чтобы не рисовать «недозревший» хвост).

    Parameters
    ----------
    sample : str
        Метка выборки (идёт в колонку ``sample`` — для сравнения нескольких винтажей).
    processed_dt : date-like, optional
        Дата актуальности данных; по умолчанию — сегодня.
    population : pd.DataFrame, optional
        База ``[client_id, balance_col]`` (если клиентов без транзакций нет в ``df``).

    Returns
    -------
    pd.DataFrame
        ``day, sample, N_total, total_balance, cum_tx_cnt, cum_tx_sum, cum_clients_with_tx,
        cum_share_of_balance, cum_share_clients``.
    """
    df = df.copy()
    df[retro_dt_col] = pd.to_datetime(df[retro_dt_col], format="mixed").dt.normalize()
    processed = pd.Timestamp(processed_dt if processed_dt is not None else "today").normalize()
    horizon_days = max(0, min(horizon_days, int((processed - df[retro_dt_col].max()).days)))
    for c in (tx_days_col, tx_amount_col, balance_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    base = population if population is not None else df
    clients = base.drop_duplicates(subset=[client_id])[[client_id, balance_col]].copy()
    clients[balance_col] = pd.to_numeric(clients[balance_col], errors="coerce")
    n_clients = clients[client_id].nunique()
    total_balance = clients[balance_col].fillna(0).sum()

    tx = df.dropna(subset=[tx_days_col, tx_amount_col])
    tx = tx[(tx[tx_days_col] >= 0) & (tx[tx_days_col] <= horizon_days)].copy()
    tx[tx_days_col] = tx[tx_days_col].astype(int)
    if step > 1:
        tx[tx_days_col] = (tx[tx_days_col] // step) * step
    grid = pd.Index(np.arange(0, horizon_days + 1, step), name="day")

    agg = (tx.groupby(tx_days_col)[tx_amount_col].agg(tx_sum="sum", tx_cnt="size")
           .reindex(grid, fill_value=0).sort_index())
    cum_sum, cum_cnt = agg["tx_sum"].cumsum(), agg["tx_cnt"].cumsum()
    first_tx = tx.groupby(client_id)[tx_days_col].min()
    cum_clients = first_tx.value_counts().reindex(grid, fill_value=0).sort_index().cumsum()

    return pd.DataFrame({
        "day": grid.to_numpy(),
        "sample": sample,
        "N_total": n_clients,
        "total_balance": total_balance,
        "cum_tx_cnt": cum_cnt.to_numpy(),
        "cum_tx_sum": cum_sum.to_numpy(),
        "cum_clients_with_tx": cum_clients.to_numpy(),
        "cum_share_of_balance": (cum_sum / total_balance).to_numpy() if total_balance > 0 else 0.0,
        "cum_share_clients": (cum_clients / n_clients).to_numpy() if n_clients > 0 else 0.0,
    })


def plot_vintage(
    res: pd.DataFrame,
    y: str = "cum_share_of_balance",
    *,
    title: str | None = None,
    x_title: str = "Дней от ретро-даты",
    legend_title: str | None = None,
) -> go.Figure:
    """Винтажные кривые по колонке ``sample`` результата :func:`maturation_transactions`.

    ``y``: ``cum_share_of_balance``, ``cum_share_clients``, ``cum_tx_sum``, ``cum_tx_cnt``,
    ``cum_clients_with_tx``.
    """
    fig = go.Figure()
    for i, (smp, g) in enumerate(res.groupby("sample", sort=False)):
        g = g.sort_values("day")
        label = f"{smp} (N={int(g['N_total'].iloc[0]):,}, tx={int(g['cum_tx_cnt'].iloc[-1]):,})"
        fig.add_scatter(x=g["day"], y=g[y], mode="lines", name=label, line={"color": color(i)})
    max_day = res["day"].max()
    for d, lbl in MILESTONES:
        if d <= max_day:
            fig.add_vline(x=d, line_dash="dash", line_color="grey", opacity=0.5,
                          annotation_text=lbl, annotation_position="top")
    if y.startswith("cum_share"):
        fig.update_yaxes(tickformat=".1%")
    fig.update_layout(legend_title_text=legend_title)
    return style(fig, title or f"Vintage: {y}", height=500, xaxis_title=x_title, yaxis_title=y)


def vintage_by_type(
    df: pd.DataFrame,
    tx_type_col: str,
    *,
    y: str = "cum_share_of_balance",
    types: list | None = None,
    top_k: int | None = None,
    min_tx: int = 0,
    abs_amount: bool = False,
    title: str | None = None,
    tx_amount_col: str = "transaction_amt",
    **maturation_kwargs: Any,
) -> tuple[pd.DataFrame, go.Figure]:
    """Винтажи в разрезе типа транзакции на одном графике (бывший ``plot_vintage_by_tx_type``).

    Parameters
    ----------
    types : list, optional
        Явный список типов; иначе все (или ``top_k`` по |обороту|, не реже ``min_tx``).
    abs_amount : bool
        Считать по ``|amount|`` (когда есть отрицательные суммы).
    maturation_kwargs
        Параметры :func:`maturation_transactions` (``client_id``, ``horizon_days``, ...).

    Returns
    -------
    (pd.DataFrame, go.Figure)
    """
    df = df.copy()
    amount_col = tx_amount_col
    if abs_amount:
        df["_amt"] = pd.to_numeric(df[tx_amount_col], errors="coerce").abs()
        amount_col = "_amt"
    df[tx_type_col] = df[tx_type_col].astype(object).fillna("NA")
    if types is None:
        turnover = (df.groupby(tx_type_col)[amount_col]
                    .apply(lambda s: pd.to_numeric(s, errors="coerce").abs().sum())
                    .sort_values(ascending=False))
        if min_tx:
            counts = df.groupby(tx_type_col).size()
            turnover = turnover[counts.reindex(turnover.index).fillna(0) >= min_tx]
        if top_k:
            turnover = turnover.head(top_k)
        types = turnover.index.tolist()

    parts = []
    for t in types:
        sub = df[df[tx_type_col] == t]
        if len(sub):
            parts.append(maturation_transactions(sub, sample=str(t), tx_amount_col=amount_col,
                                                 **maturation_kwargs))
    if not parts:
        raise ValueError("Нет данных для винтажей.")
    data = pd.concat(parts, ignore_index=True)
    retro = maturation_kwargs.get("retro_dt_col", "rtk_send_date")
    fig = plot_vintage(data, y, legend_title=tx_type_col, x_title=f"Дней от {retro}",
                       title=title or f"Vintage по «{tx_type_col}» — {y}"
                       + (" (|amount|)" if abs_amount else ""))
    return data, fig


def eda_transactions(
    df: pd.DataFrame,
    *,
    client_id: str = "contract_number",
    tx_amount_col: str = "transaction_amt",
    balance_col: str = "rtk_balance",
    retro_dt_col: str = "rtk_send_date",
    tx_dt_col: str | None = None,
    tx_days_col: str = "tx_days",
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
    top_n: int = 10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Быстрый EDA транзакционной таблицы.

    Returns
    -------
    dict
        ``overview`` (Series), ``nulls``, ``dist_table``, ``per_client``, ``sign_stats``,
        ``top_clients`` (DataFrame) и ``figures`` — словарь plotly-графиков. При
        ``verbose=True`` таблицы печатаются.
    """
    df = df.copy()
    df[retro_dt_col] = pd.to_datetime(df[retro_dt_col], format="mixed",
                                      errors="coerce").dt.normalize()
    if tx_dt_col and tx_dt_col in df.columns:
        df[tx_dt_col] = pd.to_datetime(df[tx_dt_col], format="mixed",
                                       errors="coerce").dt.normalize()
    for c in (tx_amount_col, balance_col, tx_days_col):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    amt = df[tx_amount_col]
    balance = df.drop_duplicates(client_id).set_index(client_id)[balance_col]
    n_clients = df[client_id].nunique()
    overview = pd.Series({
        "rows": len(df), "clients": n_clients, "retro_dates": df[retro_dt_col].nunique(),
        "retro_dt_min": df[retro_dt_col].min(), "retro_dt_max": df[retro_dt_col].max(),
        "tx_sum_net": amt.sum(), "tx_sum_abs": amt.abs().sum(),
        "total_balance": balance.sum(),
        "tx_per_client": amt.notna().sum() / max(n_clients, 1),
        "abs_tx_per_client": amt.abs().sum() / max(n_clients, 1),
        "balance_mean": balance.mean(), "balance_median": balance.median(),
    })
    null_cols = [c for c in (client_id, tx_amount_col, balance_col, retro_dt_col, tx_days_col,
                             tx_dt_col) if c and c in df.columns]
    nulls = df[null_cols].isna().sum()

    def qtable(s: pd.Series, name: str) -> pd.DataFrame:
        s = s.dropna()
        q = s.quantile(list(quantiles))
        q.index = [f"q{x * 100:g}" for x in quantiles]
        stats = pd.Series({"count": s.size, "mean": s.mean(), "std": s.std(), "min": s.min(),
                           "max": s.max(), "sum": s.sum()})
        return pd.concat([stats, q]).to_frame(name)

    dist_table = pd.concat([qtable(balance, "balance (per client)"),
                            qtable(amt, "tx_amount"), qtable(amt.abs(), "|tx_amount|")], axis=1)
    per_client = df.groupby(client_id).agg(
        tx_cnt=(tx_amount_col, "count"), tx_sum=(tx_amount_col, "sum"),
        tx_abs_sum=(tx_amount_col, lambda s: s.abs().sum()), tx_mean=(tx_amount_col, "mean"))
    per_client["balance"] = balance
    per_client["abs_turnover_to_balance"] = (per_client["tx_abs_sum"]
                                             / per_client["balance"].replace(0, np.nan))
    pos, neg = amt[amt > 0], amt[amt < 0]
    sign_stats = pd.DataFrame({
        "count": [len(pos), len(neg)], "sum": [pos.sum(), neg.sum()],
        "mean": [pos.mean(), neg.mean()], "median": [pos.median(), neg.median()],
    }, index=["positive", "negative"])
    top = per_client.sort_values("tx_abs_sum", ascending=False).head(top_n)

    figures: dict[str, go.Figure] = {}
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Баланс клиентов", "Баланс клиентов (log10, > 0)"))
    fig.add_histogram(x=balance.dropna(), nbinsx=60, row=1, col=1)
    fig.add_histogram(x=np.log10(balance[balance > 0]), nbinsx=60, row=1, col=2)
    figures["balance"] = style(fig, "Распределение балансов", height=380).update_layout(
        showlegend=False)

    t = amt.dropna()
    lo, hi = t.quantile([0.01, 0.99])
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Сумма транзакции (1–99%)", "|Сумма транзакции| (log10)"))
    fig.add_histogram(x=t.clip(lo, hi), nbinsx=80, row=1, col=1)
    fig.add_histogram(x=np.log10(t.abs()[t.abs() > 0]), nbinsx=80, row=1, col=2)
    figures["tx_amount"] = style(fig, "Распределение сумм транзакций", height=380).update_layout(
        showlegend=False)

    cnt = per_client["tx_cnt"]
    figures["tx_per_client"] = style(
        go.Figure(go.Histogram(x=cnt.clip(upper=cnt.quantile(0.99)), nbinsx=60)),
        "Транзакций на клиента (до 99%)", height=350)

    if tx_days_col in df.columns:
        d = df[tx_days_col].dropna()
        d = d[(d >= 0) & (d <= d.quantile(0.99))]
        by_day = df[df[tx_days_col] >= 0].groupby(tx_days_col)[tx_amount_col].apply(
            lambda s: s.abs().sum())
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Транзакций по дням от ретро-даты", "|Оборот| по дням от ретро-даты"))
        fig.add_histogram(x=d, nbinsx=60, row=1, col=1)
        fig.add_scatter(x=by_day.index, y=by_day.values, mode="lines", row=1, col=2)
        figures["activity_by_day"] = style(fig, "Активность после ретро-даты",
                                           height=380).update_layout(showlegend=False)
    if tx_dt_col and tx_dt_col in df.columns:
        m = df.dropna(subset=[tx_dt_col])
        gm = m.groupby(m[tx_dt_col].dt.to_period("M").dt.to_timestamp())[tx_amount_col].agg(
            cnt="count", abs_sum=lambda s: s.abs().sum())
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Транзакций по месяцам",
                                                            "|Оборот| по месяцам"))
        fig.add_bar(x=gm.index, y=gm["cnt"], row=1, col=1)
        fig.add_bar(x=gm.index, y=gm["abs_sum"], row=1, col=2)
        figures["by_month"] = style(fig, "Календарная динамика", height=380).update_layout(
            showlegend=False)

    report = {"overview": overview, "nulls": nulls, "dist_table": dist_table,
              "per_client": per_client, "sign_stats": sign_stats, "top_clients": top,
              "figures": figures}
    if verbose:
        for key in ("overview", "nulls", "dist_table", "sign_stats", "top_clients"):
            print(f"\n=== {key} ===")
            try:
                print(report[key].round(2).to_string())
            except TypeError:  # смешанные типы (даты + числа)
                print(report[key].to_string())
        if tx_days_col in df.columns:
            print(f"\nОтрицательных {tx_days_col}: {int((df[tx_days_col] < 0).sum())}")
    return report
