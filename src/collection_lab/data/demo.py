"""Синтетическая выборка для примеров: запускаются без рабочих данных."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from collection_lab.config import RANDOM_STATE

DEMO_DOMAINS: dict[str, list[str]] = {
    "анкета": ["age", "income", "education", "region", "children_cnt"],
    "поведение": ["dpd_max_12m", "dpd_cnt_12m", "utilization", "months_on_book"],
    "бюро": ["bureau_score", "bureau_inquiries_6m", "bureau_active_loans",
             "bureau_dpd_max_12m", "bureau_legal_flg"],
    "транзакции": ["tx_cnt_3m", "tx_sum_3m", "tx_salary_flg", "tx_cash_share"],
}


def make_demo_data(n: int = 20_000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Таблица «клиент на дату» с бинарным таргетом и признаками из четырёх доменов
    (:data:`DEMO_DOMAINS`).

    Служебные колонки: ``client_id``, ``report_date`` (2023-01 … 2024-06), ``segment``
    (``новый клиент`` / ``действующий клиент``), ``target``.

    Что заложено в данные — чтобы примеры было на чём показывать:

    - таргет зависит от всех четырёх доменов, от транзакций — только у действующих клиентов;
    - бюро нет у ~10% клиентов; ``bureau_legal_flg`` заполнен у ~1%;
    - транзакции появляются только с 2023-07-01;
    - ``bureau_dpd_max_12m`` — более точная версия ``dpd_max_12m`` (сильно коррелируют);
    - ``region`` и ``bureau_active_loans`` — шум.
    """
    rng = np.random.default_rng(random_state)

    def noise(scale: float = 1.0) -> np.ndarray:
        return rng.normal(scale=scale, size=n)

    date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D")
    segment = rng.choice(["новый клиент", "действующий клиент"], n, p=[0.4, 0.6])
    active = segment == "действующий клиент"
    pay, income, bureau, tx = rng.normal(size=(4, n))  # скрытые факторы риска
    logit = -2.0 + 0.8 * pay - 0.5 * income - 0.6 * bureau - 0.7 * tx * active
    education = pd.cut(income + noise(), [-np.inf, -0.7, 0.7, np.inf],
                       labels=["среднее", "среднее специальное", "высшее"]).astype(object)

    df = pd.DataFrame({
        "client_id": np.arange(n),
        "report_date": date,
        "segment": segment,
        "target": rng.binomial(1, expit(logit)),
        # анкета
        "age": rng.integers(18, 76, n).astype(float),
        "income": np.round(np.exp(10.8 + 0.5 * income + noise(0.4)), -2),
        "education": education,
        "region": rng.choice(["север", "юг", "запад", "восток", "центр"], n),
        "children_cnt": rng.poisson(0.8, n).astype(float),
        # поведение по продукту
        "dpd_max_12m": np.clip(np.round(10 * (pay + noise(0.8))), 0, None),
        "dpd_cnt_12m": rng.poisson(np.exp(0.5 * pay)).astype(float),
        "utilization": np.round(expit(0.8 * pay + noise()), 3),
        "months_on_book": rng.integers(1, 120, n).astype(float),
        # бюро кредитных историй
        "bureau_score": np.round(650 + 60 * bureau + noise(40)),
        "bureau_inquiries_6m": rng.poisson(np.exp(-0.4 * bureau)).astype(float),
        "bureau_active_loans": rng.poisson(2.0, n).astype(float),
        "bureau_dpd_max_12m": np.clip(np.round(10 * (pay + noise(0.3))), 0, None),
        "bureau_legal_flg": np.where(rng.random(n) < 0.01, 1.0, np.nan),
        # транзакции
        "tx_cnt_3m": rng.poisson(np.exp(2.0 + 0.3 * tx)).astype(float),
        "tx_sum_3m": np.round(np.exp(10.0 + 0.5 * tx + noise(0.5)), -2),
        "tx_salary_flg": (tx + noise() > 0).astype(float),
        "tx_cash_share": np.round(expit(-tx + noise()), 3),
    })
    df.loc[rng.random(n) < 0.1, DEMO_DOMAINS["бюро"]] = np.nan
    df.loc[df["report_date"] < "2023-07-01", DEMO_DOMAINS["транзакции"]] = np.nan
    return df


DEMO_PROCESSED_DT = "2026-01-01"  # дата актуальности демо-транзакций


def make_demo_transactions(n_contracts: int = 6_000,
                           random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Таблица транзакций по договорам — вход для винтажей и ``eda_transactions``.

    Одна строка — либо одна транзакция, либо договор без транзакций (поля транзакции пустые).

    Колонки уровня договора (повторяются в каждой его строке): ``contract_id``, ``product``
    (три продукта разного размера), ``start_date`` (контрольная дата, 2022-01 … 2024-06),
    ``balance``, ``flag`` (бинарный признак договора; у договоров с ``flag = 1`` платежей меньше).

    Колонки транзакции: ``tx_date``, ``tx_amount`` (около 3% — отрицательные возвраты),
    ``tx_type``, ``tx_channel``.

    Данные актуальны на :data:`DEMO_PROCESSED_DT`.
    """
    rng = np.random.default_rng(random_state)
    n = n_contracts
    products = np.array(["кредит наличными", "кредитная карта", "автокредит"])
    product = rng.choice(products, n, p=[0.6, 0.3, 0.1])
    pay_prob = pd.Series(product).map(dict(zip(products, [0.5, 0.35, 0.25], strict=True)))
    flag = rng.binomial(1, 0.2, n)
    contracts = pd.DataFrame({
        "contract_id": np.arange(100_000, 100_000 + n),
        "product": product,
        "start_date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n),
                                                                   unit="D"),
        "balance": np.round(np.exp(rng.normal(11.5, 0.8, n)), 2),
        "flag": flag,
    })
    pays = rng.random(n) < pay_prob.to_numpy() * np.where(flag == 1, 0.5, 1.0)
    n_tx = np.where(pays, 1 + rng.poisson(3, n), 0)

    owner = np.repeat(np.arange(n), n_tx)
    m = len(owner)
    tx_type = rng.choice(["регулярный платёж", "частичное погашение", "полное погашение"], m,
                         p=[0.7, 0.25, 0.05])
    share = np.exp(rng.normal(-4.0, 1.0, m)) * np.where(tx_type == "полное погашение", 8.0, 1.0)
    amount = np.round(contracts["balance"].to_numpy()[owner] * share, 2)
    refund = rng.random(m) < 0.03
    tx = pd.DataFrame({
        "contract_id": contracts["contract_id"].to_numpy()[owner],
        "tx_date": contracts["start_date"].to_numpy()[owner]
        + pd.to_timedelta(np.clip(rng.exponential(220, m), 1, 1200).astype(int), unit="D"),
        "tx_amount": np.where(refund, -amount, amount),
        "tx_type": np.where(refund, "возврат платежа", tx_type),
        "tx_channel": rng.choice(["онлайн", "офис", "автоплатёж"], m, p=[0.5, 0.2, 0.3]),
    })
    tx = tx[tx["tx_date"] <= DEMO_PROCESSED_DT]
    return contracts.merge(tx, on="contract_id", how="left").sort_values(
        ["contract_id", "tx_date"], ignore_index=True)
