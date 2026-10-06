"""Собирает examples/01_eda_overview.ipynb.

Запуск: python examples/builders/build_eda_overview.py

Каждый вызов ниже прописывает **все** параметры функции явно (включая значения по умолчанию) с
комментарием, что в них передаётся — это справочник по `eda`, а не быстрый рецепт.

Выполнить с сохранением графиков:
    jupyter nbconvert --to notebook --execute examples/01_eda_overview.ipynb \
        --output 01_eda_overview_executed.ipynb
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

Один прогон каждой функции модуля `eda` на таблице транзакций по договорам: одна строка — либо
одна транзакция, либо договор без транзакций (поля транзакции пустые).

Ноутбук работает на встроенных демо-данных. **Чтобы запустить на своих — поменяйте только
ячейку «Настройки».**

**Каждый вызов ниже прописывает все параметры функции явно**, даже те, что равны значению по
умолчанию, — с комментарием, что в них передаётся. Так видно полный набор настроек, а не только
то, что понадобилось для примера.

1. данные;
2. `overview`, `target_summary` — сводка по колонкам и по флагу договора;
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
""")

md("""
## Настройки

Единственная ячейка, которую нужно менять под свои данные.
""")
code("""
# --- данные ---------------------------------------------------------------------------
df = cl.data.make_demo_transactions()   # свои данные: df = pd.read_csv("путь/к/таблице.csv")
PROCESSED_DT = cl.data.DEMO_PROCESSED_DT   # дата актуальности выгрузки, например "2026-09-01"

# --- колонки уровня договора (одинаковы во всех строках договора) -----------------------
CONTRACT = "contract_id"     # идентификатор договора / клиента
START_DATE = "start_date"    # контрольная дата договора: от неё считаются дни до транзакции
BALANCE = "balance"          # баланс договора на контрольную дату
SEGMENT = "product"          # разрез договоров: продукт, канал, ...
FLAG = "flag"                # бинарный признак договора 0 / 1 — «таргет» в разделах 2–4

# --- колонки транзакции (пустые у договора без транзакций) -------------------------------
TX_DATE = "tx_date"          # дата транзакции
TX_AMOUNT = "tx_amount"      # сумма транзакции
TX_TYPE = "tx_type"          # разрез транзакций: тип операции, направление, ...

# --- расчёт винтажей ------------------------------------------------------------------
HORIZON_DAYS = 540           # горизонт в днях от контрольной даты
STEP = 7                     # шаг по дням: 1 — по дням, 7 — по неделям
""")

md("""
## 1. Данные

Колонки уровня договора повторяются в каждой его строке. Флаг `FLAG` задан на уровне договора и
играет здесь роль «таргета» для примеров.
""")
code("""
df[START_DATE] = pd.to_datetime(df[START_DATE], format="mixed").dt.normalize()
print(df.shape, "| договоров:", df[CONTRACT].nunique())
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
# флаг не меняется внутри договора — берём по одной строке на договор
clients = df.drop_duplicates(CONTRACT)

cl.eda.target_summary(
    df=clients,
    target=FLAG,  # колонка-флаг, по которой считаем n / n_target / target_rate
    by=None,  # None = сводка по всей выборке целиком (не по группам)
)
""")
code("""
cl.eda.target_summary(
    df=clients,
    target=FLAG,
    by=SEGMENT,  # разрез: отдельная строка на каждое значение колонки
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
    col=BALANCE,  # какую колонку data строим (data — DataFrame, поэтому col обязателен)
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
    col=TX_AMOUNT,
    clip_low=0.01,
    clip_high=0.99,
    force_log=None,  # останется линейная шкала, если среди сумм есть отрицательные
    bins=100,
    title=None,
)
""")
code("""
cl.eda.plot_target_rate_by_bins(
    df=clients,
    col=BALANCE,  # числовой признак, который бьём на бины
    target=FLAG,  # доля этой колонки считается в каждом бине
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
# доверительный интервал вручную: 30 «положительных» из 100 наблюдений против 3 из 50
lo, hi = cl.eda.wilson_ci(
    k=[30, 3],  # число «положительных» наблюдений в каждой группе
    n=[100, 50],  # размер каждой группы
    alpha=0.05,  # уровень значимости -> 95% доверительный интервал
)
list(zip(lo.round(3), hi.round(3), strict=True))
""")
code("""
top_segments = clients[SEGMENT].value_counts().head(4).index
dyn = cl.eda.target_dynamics(
    df=clients[clients[SEGMENT].isin(top_segments)],
    date_col=START_DATE,  # по какой дате группируем в периоды
    target_col=FLAG,  # доля этой колонки считается в каждом периоде
    segment_col=SEGMENT,  # None = без разреза, одна линия
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

Работаем с самым крупным сегментом. `add_days_since` считает число дней от контрольной даты до
транзакции.

`vintage` объединяет расчёт (`maturation_transactions`) и график (`plot_vintage`) в один вызов —
для одной выборки за раз. Обе функции остаются доступны и по отдельности: `vintage_by_type`
внутри устроен так же — считает `maturation_transactions` для каждого среза и сводит их одним
`plot_vintage`, поэтому явно нужен, когда своих групп больше одной.

У `vintage_by_type` **база одна для всех типов** — все договоры выборки и их баланс; по типам
делятся только транзакции. Поэтому доли от баланса по типам в сумме дают ровно общий винтаж
(проверка — в конце раздела). Договоры без транзакций отдельным типом не считаются.
""")
code("""
df = cl.eda.add_days_since(
    df=df,
    event_col=TX_DATE,  # дата события (транзакции)
    base_col=START_DATE,  # дата отсчёта (контрольная дата)
    name="tx_days",  # имя новой колонки: event - base, в днях
)
MAIN = clients[SEGMENT].value_counts().index[0]  # самый крупный сегмент
main = df[df[SEGMENT] == MAIN]
print("сегмент:", MAIN, "| договоров:", main[CONTRACT].nunique())
main[[CONTRACT, START_DATE, TX_DATE, "tx_days"]].dropna().head(3)
""")
code("""
data, fig = cl.eda.vintage(
    df=main,
    sample=str(MAIN),  # метка выборки — попадёт в легенду графика и колонку sample таблицы
    client_id=CONTRACT,  # колонка с идентификатором клиента/договора
    tx_days_col="tx_days",  # дни от контрольной даты (посчитаны add_days_since)
    tx_amount_col=TX_AMOUNT,  # сумма транзакции
    balance_col=BALANCE,  # баланс договора на контрольную дату
    retro_dt_col=START_DATE,  # сама контрольная дата
    horizon_days=HORIZON_DAYS,  # горизонт винтажа (урезается, если данные ещё не «дозрели»)
    step=STEP,  # шаг агрегации по дням (7 = понедельно)
    processed_dt=PROCESSED_DT,  # дата актуальности данных — от неё считается урезание горизонта
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
# та же таблица, но по оси Y — доля договоров, у которых уже была хотя бы одна транзакция
_, fig = cl.eda.vintage(
    df=main,
    sample=str(MAIN),
    client_id=CONTRACT,
    tx_days_col="tx_days",
    tx_amount_col=TX_AMOUNT,
    balance_col=BALANCE,
    retro_dt_col=START_DATE,
    horizon_days=HORIZON_DAYS,
    step=STEP,
    processed_dt=PROCESSED_DT,
    population=None,
    y="cum_share_clients",  # конверсия в транзакцию, а не доля от баланса
    title=None,
    x_title=None,
    legend_title=None,
)
fig
""")
code("""
data_types, fig = cl.eda.vintage_by_type(
    df=main,
    tx_type_col=TX_TYPE,  # колонка-разрез: своя кривая на каждое значение
    y="cum_share_of_balance",  # что откладываем по Y (см. vintage выше)
    types=None,  # None = все значения tx_type_col (либо top_k, либо не реже min_tx)
    top_k=None,  # None = без ограничения по числу типов
    min_tx=0,  # 0 = не отсеивать редкие типы по числу транзакций
    abs_amount=False,  # False = суммировать со знаком (не по модулю)
    title=None,  # None = заголовок «Винтаж по «<tx_type_col>»: <y>»
    tx_amount_col=TX_AMOUNT,  # сумма транзакции
    # дальше — те же параметры maturation_transactions, что и в vintage() выше
    client_id=CONTRACT,
    tx_days_col="tx_days",
    balance_col=BALANCE,
    retro_dt_col=START_DATE,
    horizon_days=HORIZON_DAYS,
    step=STEP,
    processed_dt=PROCESSED_DT,
    population=None,  # None = база — все договоры main, одна и та же для каждого типа
)
fig
""")
code("""
# проверка: у всех типов одна база, и сумма долей по типам = общий винтаж (vintage выше)
check = pd.DataFrame({
    "по типам (сумма)": data_types.groupby("day")["cum_share_of_balance"].sum(),
    "общий винтаж": data.set_index("day")["cum_share_of_balance"],
})
print("база по типам:", data_types["N_total"].unique(), "| общая:", data["N_total"].iloc[0])
check.tail(3)
""")

md("""
### Разные сегменты на одном графике: `vintage_by_segment`

Сегменты (продукты, каналы) — разные популяции, поэтому **у каждого своя база** (его договоры и
баланс), в отличие от `vintage_by_type`, где типы транзакций делят одну общую базу. Кривая
сегмента совпадает с `vintage` по его строкам. Горизонт у каждого сегмента свой.

| задача | функция | база |
|---|---|---|
| одна выборка | `vintage` | её договоры |
| разные сегменты / выборки | `vintage_by_segment` | у каждого своя |
| типы транзакций внутри выборки | `vintage_by_type` | общая; сумма по типам = общий винтаж |
""")
code("""
data_seg, fig = cl.eda.vintage_by_segment(
    df=df,
    segment_col=SEGMENT,  # своя кривая и своя база у каждого значения
    y="cum_share_of_balance",  # что откладываем по Y
    segments=None,  # None = все сегменты (по убыванию числа договоров); или явный список
    top_k=4,  # показать 4 крупнейших сегмента
    min_contracts=50,  # не показывать сегменты, где договоров меньше (шумные кривые)
    title=None,  # None = «Винтаж по «<segment_col>»: <y>»
    client_id=CONTRACT,  # идентификатор договора
    population=None,  # None = база каждого сегмента — его строки df
    # дальше — параметры maturation_transactions, общие для всех сегментов
    tx_days_col="tx_days",
    tx_amount_col=TX_AMOUNT,
    balance_col=BALANCE,
    retro_dt_col=START_DATE,
    horizon_days=HORIZON_DAYS,
    step=STEP,
    processed_dt=PROCESSED_DT,
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
    df=main,
    client_id=CONTRACT,  # колонка с идентификатором клиента/договора
    tx_amount_col=TX_AMOUNT,  # сумма транзакции
    balance_col=BALANCE,  # баланс договора
    retro_dt_col=START_DATE,  # контрольная дата
    tx_dt_col=TX_DATE,  # дата самой транзакции -> добавляет разбивку по месяцам
    tx_days_col="tx_days",  # дни от контрольной даты (уже посчитаны add_days_since)
    quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),  # квантили в таблице распределений
    top_n=10,  # сколько клиентов показать в топе по |обороту|
    verbose=True,  # печатать таблицы сразу (иначе только report["figures"] и сами таблицы)
)
""")
code("""
list(report["figures"])
""")
code("""
for fig in report["figures"].values():
    fig.show()
""")
code("""
report["top_clients"]
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parents[1] / "01_eda_overview.ipynb"
nbf.write(nb, out)
print("записан", out)
