"""Собирает examples/02_vintage_overview.ipynb.

Запуск: python examples/builders/build_vintage_overview.py

Каждый вызов прописывает **все** параметры явно, с комментарием.

Выполнить с сохранением графиков:
    jupyter nbconvert --to notebook --execute examples/02_vintage_overview.ipynb \
        --output 02_vintage_overview_executed.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Винтажи транзакций: `collection_lab.eda`

Сколько и когда поступает по договорам после контрольной даты: общий винтаж, разрезы по типам
транзакций, сравнение сегментов и когорт. Графики — plotly.

Ноутбук работает на встроенных демо-данных. **Чтобы запустить на своих — поменяйте только
ячейку «Настройки».**

**Винтаж** — накопленные поступления по дням от контрольной даты:
- `cum_share_of_balance` — **накопленная доля от баланса**: Σ транзакций / Σ балансов базы;
- `cum_share_clients` — **конверсия в транзакцию**: доля договоров базы, у которых уже была
  хотя бы одна транзакция.

Горизонт урезается сам, если последние контрольные даты ещё не «дозрели» к `PROCESSED_DT`.

| задача | раздел | функция | база (знаменатель) |
|---|---|---|---|
| одна выборка | 2 | `vintage` | её договоры |
| типы транзакций внутри выборки | 3, 6 | `vintage_by_type` | **общая** для всех типов |
| только крупные транзакции | 4 | `vintage_by_type` + `population` | все договоры |
| разные сегменты / когорты | 5 | `vintage_by_segment` | **у каждого своя** |
| свой набор кривых | 7 | `maturation_transactions` + `plot_vintage` | как посчитаете |
""")

code("""
import pandas as pd
import plotly.io as pio

import collection_lab as cl

pio.renderers.default = "notebook_connected+plotly_mimetype"
""")

md("""
## Настройки

Единственная ячейка, которую нужно менять под свои данные. Таблица: одна строка — либо одна
транзакция, либо договор без транзакций (поля транзакции пустые).
""")
code("""
# --- данные ---------------------------------------------------------------------------
df = cl.data.make_demo_transactions()   # свои данные: df = pd.read_csv("путь/к/таблице.csv")
PROCESSED_DT = cl.data.DEMO_PROCESSED_DT   # дата актуальности выгрузки, например "2026-09-01"
START_TO = None              # брать договоры с контрольной датой не позже этой; None — все

# --- колонки уровня договора (одинаковы во всех строках договора) -----------------------
CONTRACT = "contract_id"     # идентификатор договора / клиента
START_DATE = "start_date"    # контрольная дата договора: от неё считаются дни до транзакции
BALANCE = "balance"          # баланс договора на контрольную дату
SEGMENT = "product"          # разрез договоров: продукт, канал, ...
MAIN_SEGMENT = None          # сегмент для разделов 2–3 и 7; None — самый крупный

# --- колонки транзакции (пустые у договора без транзакций) -------------------------------
TX_DATE = "tx_date"          # дата транзакции
TX_AMOUNT = "tx_amount"      # сумма транзакции
TX_TYPES = ["tx_type", "tx_channel"]   # разрезы транзакций; первый — основной
POSITIVE_ONLY = True         # убрать отрицательные транзакции (возвраты, сторно)
LARGE_TX = [5_000, 20_000]   # пороги «крупной» транзакции для разделов 4, 5 и 7

# --- расчёт ---------------------------------------------------------------------------
HORIZON_DAYS = 540           # горизонт в днях от контрольной даты
STEP = 7                     # шаг по дням для разрезов: 7 — по неделям, кривые глаже
KEY_DAYS = [30, 90, 180, 365, 540, 730]   # сроки для таблицы итогов
MIN_CONTRACTS = 100          # не показывать сегменты, где договоров меньше
""")

md("""
## 1. Данные

`add_days_since` добавляет `tx_days` — дни от контрольной даты до транзакции.

**База винтажа.** Знаменатель — все договоры выборки и их баланс. Если отфильтровать *строки*
(например, оставить только крупные транзакции), договоры, у которых таких транзакций нет,
выпадут из базы, и доли завысятся. Поэтому базу держим отдельной таблицей `contracts`
(одна строка на договор) и передаём её в `population`, когда фильтруем транзакции.
""")
code("""
df[START_DATE] = pd.to_datetime(df[START_DATE], format="mixed").dt.normalize()
if START_TO is not None:
    df = df[df[START_DATE] <= START_TO]
df = cl.eda.add_days_since(
    df=df,
    event_col=TX_DATE,  # дата события — транзакции
    base_col=START_DATE,  # дата отсчёта — контрольная дата
    name="tx_days",  # имя новой колонки: event - base, в днях
)

# база: одна строка на договор — идентификатор, баланс, сегмент и контрольная дата
contracts = df.drop_duplicates(CONTRACT)[[CONTRACT, BALANCE, SEGMENT, START_DATE]].copy()

# строки для расчётов: договоры без транзакций остаются (fillna — чтобы пустая сумма не
# отсекалась фильтром), отрицательные транзакции убираются
rows = df[df[TX_AMOUNT].fillna(1) > 0] if POSITIVE_ONLY else df

MAIN = MAIN_SEGMENT if MAIN_SEGMENT is not None else contracts[SEGMENT].value_counts().index[0]
TX_TYPE = TX_TYPES[0]
print(f"строк: {len(df):,} | договоров: {len(contracts):,} | основной сегмент: {MAIN}")
contracts[SEGMENT].value_counts()
""")

md("""
## 2. Основной сегмент: общий винтаж
""")
code("""
main = rows[rows[SEGMENT] == MAIN]
main_base = contracts[contracts[SEGMENT] == MAIN]
print(f"договоров в строках: {main[CONTRACT].nunique():,} | в базе: {len(main_base):,}")
""")
code("""
data_main, fig = cl.eda.vintage(
    df=main,
    sample=str(MAIN),  # метка выборки — в легенду и в колонку sample таблицы
    client_id=CONTRACT,  # идентификатор договора
    tx_days_col="tx_days",  # дни от контрольной даты (посчитаны add_days_since)
    tx_amount_col=TX_AMOUNT,  # сумма транзакции
    balance_col=BALANCE,  # баланс договора на контрольную дату
    retro_dt_col=START_DATE,  # контрольная дата
    horizon_days=HORIZON_DAYS,  # горизонт в днях
    step=1,  # шаг агрегации по дням: 1 = по дням, 7 = по неделям
    processed_dt=PROCESSED_DT,  # дата актуальности данных
    population=main_base,  # база: все договоры сегмента
    y="cum_share_of_balance",  # по Y — накопленная доля от баланса
    title=None,  # None = «Винтаж: накопленная доля от баланса»
    x_title=None,  # None = «Дней от <контрольная дата>»
    legend_title=None,  # None = без заголовка легенды
)
fig
""")
code("""
_, fig = cl.eda.vintage(
    df=main,
    sample=str(MAIN),
    client_id=CONTRACT,
    tx_days_col="tx_days",
    tx_amount_col=TX_AMOUNT,
    balance_col=BALANCE,
    retro_dt_col=START_DATE,
    horizon_days=HORIZON_DAYS,
    step=1,
    processed_dt=PROCESSED_DT,
    population=main_base,
    y="cum_share_clients",  # по Y — конверсия в транзакцию
    title=None,
    x_title=None,
    legend_title=None,
)
fig
""")
code("""
# таблица винтажа: значения на ключевых сроках
data_main[data_main["day"].isin(KEY_DAYS)][
    ["day", "N_total", "cum_tx_sum", "cum_share_of_balance", "cum_share_clients"]]
""")

md("""
## 3. Основной сегмент: разрезы по транзакциям — `vintage_by_type`

База у всех типов одна (все договоры сегмента), по типам делятся только транзакции: доли от
баланса по типам в сумме дают общий винтаж из раздела 2. Конверсии по типам в сумме могут быть
больше общей — у договора бывают транзакции нескольких типов. Транзакция без типа попадает в
`NA`.
""")
code("""
by_type = {}
for tx_col in TX_TYPES:  # каждый разрез транзакций из настроек
    for y in ("cum_share_of_balance", "cum_share_clients"):
        by_type[tx_col, y], fig = cl.eda.vintage_by_type(
            df=main,
            tx_type_col=tx_col,  # разрез: своя кривая на каждое значение колонки
            y=y,  # по Y — доля от баланса или конверсия в транзакцию
            types=None,  # None = все значения (с учётом top_k и min_tx); или явный список
            top_k=None,  # None = без ограничения числа типов
            min_tx=0,  # 0 = не отсеивать редкие типы
            abs_amount=False,  # False = суммы со знаком
            title=None,  # None = «Винтаж по «<tx_type_col>»: <подпись y>»
            tx_amount_col=TX_AMOUNT,  # сумма транзакции
            # дальше — параметры maturation_transactions
            client_id=CONTRACT,
            tx_days_col="tx_days",
            balance_col=BALANCE,
            retro_dt_col=START_DATE,
            horizon_days=HORIZON_DAYS,
            step=STEP,
            processed_dt=PROCESSED_DT,
            population=main_base,  # общая база для всех типов
        )
        fig.show()
""")
code("""
# проверка общей базы: сумма долей по типам = общий винтаж с тем же шагом
total = cl.eda.maturation_transactions(
    df=main, sample=str(MAIN), client_id=CONTRACT, tx_days_col="tx_days",
    tx_amount_col=TX_AMOUNT, balance_col=BALANCE, retro_dt_col=START_DATE,
    horizon_days=HORIZON_DAYS, step=STEP, processed_dt=PROCESSED_DT, population=main_base,
)
check = pd.DataFrame({
    "сумма по типам": by_type[TX_TYPE, "cum_share_of_balance"].groupby("day")[
        "cum_share_of_balance"].sum(),
    "общий винтаж": total.set_index("day")["cum_share_of_balance"],
})
check.iloc[[len(check) // 4, len(check) // 2, -1]]
""")

md("""
## 4. Остальные сегменты, только крупные транзакции

Фильтр по сумме убирает строки. Без `population` из базы выпали бы договоры, у которых не было
ни одной крупной транзакции, и доли посчитались бы от меньшей базы. С `population` база — все
договоры, а в числитель идут только крупные транзакции.
""")
code("""
large = LARGE_TX[-1]
other = rows[(rows[SEGMENT] != MAIN)
             & (rows[TX_AMOUNT].fillna(float("inf")) > large)]  # fillna: строки без транзакций
other_base = contracts[contracts[SEGMENT] != MAIN]

for y, label in (("cum_share_of_balance", "накопленная доля от баланса"),
                 ("cum_share_clients", "конверсия в транзакцию")):
    _, fig = cl.eda.vintage_by_type(
        df=other,
        tx_type_col=TX_TYPE,
        y=y,
        types=None,
        top_k=None,
        min_tx=0,
        abs_amount=False,
        title=f"Кроме «{MAIN}», транзакции > {large:,}: {label}",  # свой заголовок
        tx_amount_col=TX_AMOUNT,
        client_id=CONTRACT,
        tx_days_col="tx_days",
        balance_col=BALANCE,
        retro_dt_col=START_DATE,
        horizon_days=HORIZON_DAYS,
        step=STEP,
        processed_dt=PROCESSED_DT,
        population=other_base,  # база — все договоры, а не только те, что прошли фильтр
    )
    fig.show()
""")

md("""
## 5. Сегменты на одном графике — `vintage_by_segment`

Вместо цикла «один сегмент — один график»: все сегменты на одной картинке, **у каждого своя
база** (его договоры и баланс). Кривая сегмента совпадает с `vintage` по его строкам. Малые
сегменты дают шумные кривые — отсекаем их `MIN_CONTRACTS`. Сегмент, по которому в выгрузке нет
ни одной транзакции, остаётся на графике нулевой линией — это не ошибка расчёта.
""")
code("""
for y, title in (("cum_share_of_balance", "Сегменты: накопленная доля от баланса"),
                 ("cum_share_clients", "Сегменты: конверсия в транзакцию")):
    data_seg, fig = cl.eda.vintage_by_segment(
        df=rows,
        segment_col=SEGMENT,  # своя кривая и своя база у каждого значения
        y=y,
        segments=None,  # None = все сегменты по убыванию числа договоров; или явный список
        top_k=None,  # None = без ограничения числа сегментов
        min_contracts=MIN_CONTRACTS,  # не показывать сегменты, где договоров меньше
        title=title,
        client_id=CONTRACT,
        population=contracts,  # база; делится по сегменту так же, как df
        # дальше — параметры maturation_transactions, общие для всех сегментов
        tx_days_col="tx_days",
        tx_amount_col=TX_AMOUNT,
        balance_col=BALANCE,
        retro_dt_col=START_DATE,
        horizon_days=HORIZON_DAYS,
        step=STEP,
        processed_dt=PROCESSED_DT,
    )
    fig.show()
""")
code("""
# сводка: база и итог на последнем дне горизонта каждого сегмента
data_seg.groupby("sample").tail(1)[
    ["sample", "N_total", "day", "cum_share_of_balance", "cum_share_clients"]]
""")
code("""
# то же только по крупным транзакциям; база прежняя — через population
_, fig = cl.eda.vintage_by_segment(
    df=rows[rows[TX_AMOUNT].fillna(float("inf")) > LARGE_TX[0]],
    segment_col=SEGMENT,
    y="cum_share_of_balance",
    segments=None,
    top_k=None,
    min_contracts=MIN_CONTRACTS,
    title=f"Сегменты, транзакции > {LARGE_TX[0]:,}: накопленная доля от баланса",
    client_id=CONTRACT,
    population=contracts,
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

md("""
### Когорты: тот же `vintage_by_segment`, разрез по дате

Сегмент — любая колонка уровня договора. Квартал контрольной даты показывает, меняется ли
скорость и уровень поступлений у новых когорт (дрейф). Горизонт у каждой когорты свой: поздние
когорты короче.
""")
code("""
contracts["cohort"] = contracts[START_DATE].dt.to_period("Q").astype(str)
main_c = main.merge(contracts[[CONTRACT, "cohort"]], on=CONTRACT)

_, fig = cl.eda.vintage_by_segment(
    df=main_c,
    segment_col="cohort",  # квартал контрольной даты
    y="cum_share_of_balance",
    segments=sorted(main_c["cohort"].unique()),  # явный список — в хронологическом порядке
    top_k=None,
    min_contracts=0,
    title=f"{MAIN} по кварталам контрольной даты: накопленная доля от баланса",
    client_id=CONTRACT,
    population=contracts[contracts[SEGMENT] == MAIN],  # в базе тоже есть cohort
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

md("""
## 6. Разрез по типам транзакций внутри каждого сегмента

Один график на сегмент с достаточной базой: типы транзакций делят базу этого сегмента.
""")
code("""
sizes = contracts[SEGMENT].value_counts()
for segment in sizes[sizes >= MIN_CONTRACTS].index:
    part = rows[rows[SEGMENT] == segment]
    if part[TX_AMOUNT].notna().sum() == 0:  # vintage_by_type нужна хоть одна транзакция
        print(f"{segment}: нет транзакций — пропускаем")
        continue
    _, fig = cl.eda.vintage_by_type(
        df=part,
        tx_type_col=TX_TYPE,
        y="cum_share_of_balance",
        types=None,
        top_k=None,
        min_tx=0,
        abs_amount=False,
        title=f"{segment}: накопленная доля от баланса по типам транзакций",
        tx_amount_col=TX_AMOUNT,
        client_id=CONTRACT,
        tx_days_col="tx_days",
        balance_col=BALANCE,
        retro_dt_col=START_DATE,
        horizon_days=HORIZON_DAYS,
        step=STEP,
        processed_dt=PROCESSED_DT,
        population=contracts[contracts[SEGMENT] == segment],
    )
    fig.show()
""")

md("""
## 7. Свой набор кривых: `maturation_transactions` + `plot_vintage`

Когда нужно сравнить на одном графике выборки, которые не укладываются в одну колонку-разрез,
например «все транзакции» против «только крупные». `maturation_transactions` считает таблицу
для одной выборки, `plot_vintage` рисует все выборки из объединённой таблицы (по колонке
`sample`).
""")
code("""
samples = [(f"{MAIN}: все транзакции", main)] + [
    (f"{MAIN}: транзакции > {limit:,}", main[main[TX_AMOUNT].fillna(float("inf")) > limit])
    for limit in LARGE_TX]
parts = [
    cl.eda.maturation_transactions(
        df=sub,
        sample=label,  # имя кривой в легенде
        client_id=CONTRACT,
        tx_days_col="tx_days",
        tx_amount_col=TX_AMOUNT,
        balance_col=BALANCE,
        retro_dt_col=START_DATE,
        horizon_days=HORIZON_DAYS,
        step=STEP,
        processed_dt=PROCESSED_DT,
        population=main_base,  # одна база — кривые сравнимы
    )
    for label, sub in samples
]
cl.eda.plot_vintage(
    res=pd.concat(parts, ignore_index=True),  # таблицы maturation_transactions подряд
    y="cum_share_of_balance",
    title=f"{MAIN}: вклад крупных транзакций",  # None = «Винтаж: <подпись y>»
    x_title="Дней от контрольной даты",
    legend_title="выборка",  # заголовок легенды
)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parents[1] / "02_vintage_overview.ipynb"
nbf.write(nb, out)
print("записан", out)
