"""Собирает examples/01_eda_overview.ipynb.

Запуск: python examples/builders/build_eda_overview.py

Каждый вызов ниже прописывает **все** параметры функции явно (включая значения по умолчанию) с
комментарием, что в них передаётся — это справочник по `eda`, а не быстрый рецепт.

Выполнить с сохранением графиков:
    jupyter nbconvert --to notebook --execute --inplace examples/01_eda_overview.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# `collection_lab.eda` — обзор всех функций

Один прогон каждой функции модуля `eda` на транзакционных данных RTK (`df_1.csv`):
таблица со сделками (party, contract, транзакция, даты, суммы, баланс) — типичный вход для
`vintage` / `eda_transactions`, а не для моделирования, поэтому пример отдельный от
`03_rtk_pipeline.ipynb`.

**Каждый вызов ниже прописывает все параметры функции явно**, даже те, что равны значению по
умолчанию, — с комментарием, что в них передаётся. Так видно полный набор настроек, а не только
то, что понадобилось для примера.

1. данные;
2. `overview`, `target_summary` — сводка по колонкам и по флагу `ispol_doc_flg`;
3. `plot_distribution`, `plot_target_rate_by_bins` — распределения и связь с флагом;
4. `wilson_ci`, `target_dynamics` — доля флага во времени с доверительным интервалом;
5. `add_days_since`, `vintage`, `vintage_by_type`, `vintage_by_segment` — винтажи
   (расчёт + график одним вызовом);
6. `eda_transactions` — сводный отчёт по транзакциям.
""")

code("""
import pandas as pd
import plotly.io as pio

import collection_lab as cl

pio.renderers.default = "notebook_connected+plotly_mimetype"
DATA = "D:/ML_lib/RTK_model/RTK_model/data/df_1.csv"
""")

md("""
## 1. Данные

Одна строка — либо договор без транзакций (в этом случае поля транзакции пустые), либо одна
транзакция по договору. `ispol_doc_flg` — флаг исполнительного производства, задан на уровне
договора (не меняется от строки к строке) и играет здесь роль «таргета» для примеров.
""")
code("""
df = pd.read_csv(DATA, low_memory=False)
df["rtk_send_date"] = pd.to_datetime(df["rtk_send_date"], format="ISO8601").dt.normalize()
print(df.shape, "| договоров:", df["contract_number"].nunique())
df.head(3)
""")

md("""
## 2. `overview`, `target_summary`

`overview` — типы, пропуски, уникальные значения и базовые статистики по каждой колонке;
`target_summary` — размер выборки и доля флага, целиком и в разрезе.
""")
code("""
cl.eda.overview(
    df=df,
    columns=None,  # None = все колонки df; можно передать список конкретных
)
""")
code("""
# ispol_doc_flg не меняется внутри договора — берём по одной строке на договор
clients = df.drop_duplicates("contract_number")

cl.eda.target_summary(
    df=clients,
    target="ispol_doc_flg",  # колонка-флаг, по которой считаем n / n_target / target_rate
    by=None,  # None = сводка по всей выборке целиком (не по группам)
)
""")
code("""
cl.eda.target_summary(
    df=clients,
    target="ispol_doc_flg",
    by="financial_account_subtype_cd",  # разрез: отдельная строка на каждое значение колонки
)
""")

md("""
## 3. Распределения

`plot_distribution` — гистограмма с обрезкой выбросов и авто-лог-шкалой;
`plot_target_rate_by_bins` — доля флага по квантильным бинам признака (+ объём в столбцах).
""")
code("""
cl.eda.plot_distribution(
    data=clients,
    col="rtk_balance",  # какую колонку data строим (data — DataFrame, поэтому col обязателен)
    clip_low=0.01,  # нижний квантиль обрезки перед отрисовкой
    clip_high=0.99,  # верхний квантиль обрезки
    force_log=None,  # None = лог-шкала включается сама, если skew > 2 и все значения > 0
    bins=100,  # число бинов гистограммы
    title=None,  # None = заголовок собирается автоматически (имя колонки + mean/median/std/skew)
)
""")
code("""
cl.eda.plot_distribution(
    data=df,
    col="transaction_amt",
    clip_low=0.01,
    clip_high=0.99,
    force_log=None,  # здесь останется линейная шкала: в сумме транзакции есть отрицательные
    bins=100,
    title=None,
)
""")
code("""
cl.eda.plot_target_rate_by_bins(
    df=clients,
    col="rtk_balance",  # числовой признак, который бьём на бины
    target="ispol_doc_flg",  # доля этой колонки считается в каждом бине
    bins=8,  # число квантильных бинов
    title=None,  # None = заголовок «Доля таргета по бинам <col>»
)
""")

md("""
## 4. Доля флага во времени: `wilson_ci`, `target_dynamics`

`wilson_ci` — доверительный интервал доли (используется внутри `target_dynamics`, но пригождается
и отдельно, например для одной ручной сводки). `target_dynamics` считает долю флага по периодам
и сегментам сама и сразу возвращает объект с `.plot()`.
""")
code("""
# доверительный интервал вручную: 30 «плохих» из 100 наблюдений против 3 из 50
lo, hi = cl.eda.wilson_ci(
    k=[30, 3],  # число «положительных» наблюдений в каждой группе
    n=[100, 50],  # размер каждой группы
    alpha=0.05,  # уровень значимости -> 95% доверительный интервал
)
list(zip(lo.round(3), hi.round(3), strict=True))
""")
code("""
top_types = clients["financial_account_subtype_cd"].value_counts().head(4).index
dyn = cl.eda.target_dynamics(
    df=clients[clients["financial_account_subtype_cd"].isin(top_types)],
    date_col="rtk_send_date",  # по какой дате группируем в периоды
    target_col="ispol_doc_flg",  # доля этой колонки считается в каждом периоде
    segment_col="financial_account_subtype_cd",  # None = без разреза, одна линия
    freq="Q",  # период агрегации: "D"/"W"/"M"/"Q"
    alpha=0.05,  # уровень значимости доверительного интервала Уилсона
    segment_order=None,  # None = сегменты идут в алфавитном порядке
    verbose=True,  # печатать предупреждение о строках с пустой датой/таргетом
)
dyn.plot()
""")
code("""
dyn.table.head()
""")

md("""
## 5. Винтажи

Работаем с основным продуктом (`PHX`) и договорами не позже 2024-06-01 — как в исходном
ноутбуке `sample_overview`. `add_days_since` считает число дней от отправки в РТК до транзакции.

`vintage` объединяет расчёт (`maturation_transactions`) и график (`plot_vintage`) в один вызов —
для одной выборки за раз. Обе функции остаются доступны и по отдельности: `vintage_by_type`
внутри устроен так же — считает `maturation_transactions` для каждого среза и сводит их одним
`plot_vintage`, поэтому явно нужен, когда своих групп больше одной.

У `vintage_by_type` **база одна для всех типов** — все договоры выборки и их баланс; по типам
делятся только транзакции. Поэтому доли от баланса по типам в сумме дают ровно общий винтаж
(проверка — в конце раздела). Договоры без транзакций отдельным типом не считаются.
""")
code("""
phx = df[(df["financial_account_subtype_cd"] == "PHX") & (df["rtk_send_date"] <= "2024-06-01")]
phx = cl.eda.add_days_since(
    df=phx,
    event_col="real_transaction_dttm",  # дата события (транзакции)
    base_col="rtk_send_date",  # дата отсчёта (отправка в РТК)
    name="tx_days",  # имя новой колонки: event - base, в днях
)
phx[["contract_number", "rtk_send_date", "real_transaction_dttm", "tx_days"]].head(3)
""")
code("""
data, fig = cl.eda.vintage(
    df=phx,
    sample="PHX",  # метка выборки — попадёт в легенду графика и колонку sample таблицы
    client_id="contract_number",  # колонка с идентификатором клиента/договора
    tx_days_col="tx_days",  # дни от ретро-даты (посчитаны add_days_since)
    tx_amount_col="transaction_amt",  # сумма транзакции
    balance_col="rtk_balance",  # баланс клиента на ретро-дату
    retro_dt_col="rtk_send_date",  # сама ретро-дата
    horizon_days=730,  # горизонт винтажа в днях (урезается, если данные ещё не «дозрели»)
    step=7,  # шаг агрегации по дням (7 = понедельно)
    processed_dt="2026-09-01",  # дата актуальности данных — от неё считается урезание горизонта
    population=None,  # None = база клиентов берётся из df, а не из отдельной таблицы
    y="cum_share_of_balance",  # что откладываем по Y: накопленная доля транзакций от баланса
    title=None,  # None = заголовок «Винтаж: <человекочитаемое имя y>»
    x_title=None,  # None = «Дней от <retro_dt_col>»
    legend_title=None,  # None = без заголовка легенды (тут только одна кривая)
)
fig
""")
code("""
data.tail(3)
""")
code("""
# та же таблица, но по оси Y — доля клиентов, у которых уже была хотя бы одна транзакция
_, fig = cl.eda.vintage(
    df=phx,
    sample="PHX",
    client_id="contract_number",
    tx_days_col="tx_days",
    tx_amount_col="transaction_amt",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt="2026-09-01",
    population=None,
    y="cum_share_clients",  # конверсия в транзакцию, а не доля от баланса
    title=None,
    x_title=None,
    legend_title=None,
)
fig
""")
code("""
data, fig = cl.eda.vintage_by_type(
    df=phx,
    tx_type_col="tr_direction",  # колонка-разрез: своя кривая на каждое значение
    y="cum_share_of_balance",  # что откладываем по Y (см. vintage выше)
    types=None,  # None = все значения tx_type_col (либо top_k, либо не реже min_tx)
    top_k=None,  # None = без ограничения по числу типов
    min_tx=0,  # 0 = не отсеивать редкие типы по числу транзакций
    abs_amount=False,  # False = суммировать со знаком (не по модулю)
    title=None,  # None = заголовок «Винтаж по «<tx_type_col>»: <y>»
    tx_amount_col="transaction_amt",  # сумма транзакции
    # дальше — те же параметры maturation_transactions, что и в vintage() выше
    client_id="contract_number",
    tx_days_col="tx_days",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt="2026-09-01",
    population=None,
)
fig
""")
code("""
data_types, fig = cl.eda.vintage_by_type(
    df=phx,
    tx_type_col="transaction_type_cd",  # разрез по типу операции (PAY, DDJ, ...)
    y="cum_share_of_balance",
    types=None,
    top_k=None,
    min_tx=0,
    abs_amount=False,
    title=None,
    tx_amount_col="transaction_amt",
    client_id="contract_number",
    tx_days_col="tx_days",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt="2026-09-01",
    population=None,  # None = база — все договоры phx, одна и та же для каждого типа
)
fig
""")
code("""
# проверка: у всех типов одна база, и сумма долей по типам = общий винтаж (vintage выше)
total, _ = cl.eda.vintage(
    df=phx, sample="PHX", client_id="contract_number", tx_days_col="tx_days",
    tx_amount_col="transaction_amt", balance_col="rtk_balance", retro_dt_col="rtk_send_date",
    horizon_days=730, step=7, processed_dt="2026-09-01", population=None,
    y="cum_share_of_balance", title=None, x_title=None, legend_title=None,
)
check = pd.DataFrame({
    "по типам (сумма)": data_types.groupby("day")["cum_share_of_balance"].sum(),
    "общий винтаж": total.set_index("day")["cum_share_of_balance"],
})
print("база по типам:", data_types["N_total"].unique(), "| общая:", total["N_total"].iloc[0])
check.tail(3)
""")

md("""
### Разные продукты на одном графике: `vintage_by_segment`

Продукты — разные популяции, поэтому **у каждого своя база** (его договоры и баланс), в отличие
от `vintage_by_type`, где типы транзакций делят одну общую базу. Кривая продукта совпадает с
`vintage` по его строкам (для PHX — те же 3,80 %, что выше). Горизонт у каждого продукта свой.

| задача | функция | база |
|---|---|---|
| один продукт | `vintage` | его договоры |
| разные продукты / выборки | `vintage_by_segment` | у каждого своя |
| типы транзакций внутри продукта | `vintage_by_type` | общая; сумма по типам = общий винтаж |
""")
code("""
products = df[df["rtk_send_date"] <= "2024-06-01"]  # все продукты, не только PHX
products = cl.eda.add_days_since(
    df=products,
    event_col="real_transaction_dttm",
    base_col="rtk_send_date",
    name="tx_days",
)
data_seg, fig = cl.eda.vintage_by_segment(
    df=products,
    segment_col="financial_account_subtype_cd",  # продукт; своя кривая и своя база у каждого
    y="cum_share_of_balance",  # что откладываем по Y
    segments=None,  # None = все продукты (по убыванию числа договоров); или явный список
    top_k=4,  # показать 4 крупнейших продукта
    min_contracts=50,  # не показывать продукты, где договоров меньше (шумные кривые)
    title=None,  # None = «Винтаж по «<segment_col>»: <y>»
    client_id="contract_number",  # идентификатор договора
    population=None,  # None = база каждого продукта — его строки df
    # дальше — параметры maturation_transactions, общие для всех продуктов
    tx_days_col="tx_days",
    tx_amount_col="transaction_amt",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt="2026-09-01",
)
fig
""")
code("""
data_seg.groupby("sample").tail(1)[["sample", "N_total", "day", "cum_share_of_balance",
                                    "cum_share_clients"]]
""")

md("""
## 6. `eda_transactions` — сводный отчёт

Одна функция считает и печатает (при `verbose=True`) обзор, пропуски, распределения, агрегаты
по клиенту, статистику по знаку транзакции и топ клиентов; графики лежат в `report["figures"]`.
""")
code("""
report = cl.eda.eda_transactions(
    df=phx,
    client_id="contract_number",  # колонка с идентификатором клиента/договора
    tx_amount_col="transaction_amt",  # сумма транзакции
    balance_col="rtk_balance",  # баланс клиента
    retro_dt_col="rtk_send_date",  # ретро-дата (дата отправки в РТК)
    tx_dt_col="real_transaction_dttm",  # дата самой транзакции -> добавляет разбивку по месяцам
    tx_days_col="tx_days",  # дни от ретро-даты (уже посчитаны add_days_since)
    quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),  # квантили в таблице распределений
    top_n=10,  # сколько клиентов показать в топе по |обороту|
    verbose=True,  # печатать таблицы сразу (иначе только report["figures"] и сами таблицы)
)
""")
code("""
list(report["figures"])
""")
code("""
report["figures"]["balance"]
""")
code("""
report["figures"]["tx_amount"]
""")
code("""
report["top_clients"]
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parents[1] / "01_eda_overview.ipynb"
nbf.write(nb, out)
print("записан", out)
