"""Собирает examples/02_vintage_overview.ipynb.

Запуск: python examples/builders/build_vintage_overview.py

Винтажи на `df_1.csv` по сценарию исходного `sample_overview.ipynb` (RTK_model), но на функциях
`collection_lab.eda`. Каждый вызов прописывает **все** параметры явно, с комментарием.

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

Тот же анализ, что в `sample_overview.ipynb`, на функциях библиотеки: винтаж PHX, разрезы по
направлению и типу транзакции, другие продукты, сравнение продуктов и когорт. Графики — plotly,
подписи на русском.

**Винтаж** — накопленные поступления по дням от ретро-даты (`rtk_send_date`):
- `cum_share_of_balance` — **накопленная доля от баланса**: Σ транзакций / Σ балансов базы;
- `cum_share_clients` — **конверсия в транзакцию**: доля договоров базы, у которых уже была
  хотя бы одна транзакция.

Горизонт урезается сам, если последние ретро-даты ещё не «дозрели» к `processed_dt`.

| задача | функция | база (знаменатель) |
|---|---|---|
| одна выборка | `vintage` | её договоры |
| типы транзакций внутри выборки | `vintage_by_type` | **общая**; сумма по типам = общий винтаж |
| разные продукты / когорты / каналы | `vintage_by_segment` | **у каждого своя** |
| свой набор кривых на одном графике | `maturation_transactions` + `plot_vintage` | как посчитаете |

Соответствие ячейкам `sample_overview`:

| `sample_overview` | раздел | функция |
|---|---|---|
| `maturation_transactions` + `plot_vintage(res)` / `y='cum_share_clients'` | 2 | `vintage` |
| `plot_vintage_by_tx_type` (`tr_direction`, `transaction_type_cd`) | 3 | `vintage_by_type` |
| не-PHX, транзакции > 10 000 | 4 | `vintage_by_type` + `population` |
| цикл по продуктам с `plot_vintage` | 5 | `vintage_by_segment` — всё на одном графике |
| цикл по продуктам с `plot_vintage_by_tx_type` | 6 | `vintage_by_type` в цикле |
| — (сравнить произвольные выборки) | 7 | `maturation_transactions` + `plot_vintage` |
""")

code("""
import pandas as pd
import plotly.io as pio

import collection_lab as cl

pio.renderers.default = "notebook_connected+plotly_mimetype"
DATA = "D:/ML_lib/RTK_model/RTK_model/data/df_1.csv"
PROCESSED_DT = "2026-09-01"  # дата актуальности выгрузки: от неё урезается «недозревший» хвост
RETRO_TO = "2024-06-01"  # как в sample_overview: договоры, отправленные в РТК не позже этой даты
""")

md("""
## 1. Данные

Одна строка — либо договор без транзакций (поля транзакции пустые), либо одна транзакция.
`add_days_since` добавляет `tx_days` — дни от отправки в РТК до транзакции.

**База винтажа.** Знаменатель — все договоры выборки и их баланс. Если отфильтровать *строки*
(например, оставить только транзакции > 5 000 ₽), договоры, у которых таких транзакций нет,
выпадут из базы, и доли завысятся. Поэтому базу держим отдельной таблицей `contracts`
(одна строка на договор) и передаём её в `population`, когда фильтруем транзакции.
""")
code("""
df = pd.read_csv(DATA, low_memory=False)
df["rtk_send_date"] = pd.to_datetime(df["rtk_send_date"], format="ISO8601").dt.normalize()
df = df[df["rtk_send_date"] <= RETRO_TO]
df = cl.eda.add_days_since(
    df=df,
    event_col="real_transaction_dttm",  # дата события — транзакции
    base_col="rtk_send_date",  # дата отсчёта — отправка в РТК
    name="tx_days",  # имя новой колонки: event - base, в днях
)

# база: одна строка на договор — идентификатор, баланс и продукт (для разрезов по продуктам)
contracts = df.drop_duplicates("contract_number")[
    ["contract_number", "rtk_balance", "financial_account_subtype_cd", "rtk_send_date"]].copy()
print(f"строк: {len(df):,} | договоров: {len(contracts):,}")
contracts["financial_account_subtype_cd"].value_counts()
""")

md("""
## 2. PHX: общий винтаж

Как в `sample_overview`: продукт PHX, отрицательные транзакции убираем, договоры без транзакций
оставляем (`fillna` — чтобы пустая сумма не отсекалась фильтром).
""")
code("""
phx = df[(df["financial_account_subtype_cd"] == "PHX") & (df["transaction_amt"].fillna(1) > 0)]
phx_base = contracts[contracts["financial_account_subtype_cd"] == "PHX"]
print(f"договоров в строках phx: {phx['contract_number'].nunique():,} | в базе: {len(phx_base):,}")
""")
code("""
data_phx, fig = cl.eda.vintage(
    df=phx,
    sample="PHX",  # метка выборки — в легенду и в колонку sample таблицы
    client_id="contract_number",  # идентификатор договора
    tx_days_col="tx_days",  # дни от ретро-даты (посчитаны add_days_since)
    tx_amount_col="transaction_amt",  # сумма транзакции
    balance_col="rtk_balance",  # баланс договора на ретро-дату
    retro_dt_col="rtk_send_date",  # ретро-дата
    horizon_days=730,  # горизонт в днях (2 года)
    step=1,  # шаг агрегации по дням: 1 = по дням, 7 = по неделям
    processed_dt=PROCESSED_DT,  # дата актуальности данных
    population=phx_base,  # база: все договоры PHX (здесь совпадает с phx, но так надёжнее)
    y="cum_share_of_balance",  # по Y — накопленная доля от баланса
    title=None,  # None = «Винтаж: накопленная доля от баланса»
    x_title=None,  # None = «Дней от rtk_send_date»
    legend_title=None,  # None = без заголовка легенды
)
fig
""")
code("""
_, fig = cl.eda.vintage(
    df=phx,
    sample="PHX",
    client_id="contract_number",
    tx_days_col="tx_days",
    tx_amount_col="transaction_amt",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=1,
    processed_dt=PROCESSED_DT,
    population=phx_base,
    y="cum_share_clients",  # по Y — конверсия в транзакцию
    title=None,
    x_title=None,
    legend_title=None,
)
fig
""")
code("""
# таблица винтажа: значения на ключевых сроках
data_phx[data_phx["day"].isin([30, 90, 180, 365, 540, 730])][
    ["day", "N_total", "cum_tx_sum", "cum_share_of_balance", "cum_share_clients"]]
""")

md("""
## 3. PHX: разрезы по транзакциям — `vintage_by_type`

База у всех типов одна (все договоры PHX), по типам делятся только транзакции: доли от баланса
по типам в сумме дают общий винтаж из раздела 2. Конверсии по типам в сумме могут быть больше
общей — у договора бывают транзакции нескольких типов. Транзакция без типа попадает в `NA`.
""")
code("""
data_dir, fig = cl.eda.vintage_by_type(
    df=phx,
    tx_type_col="tr_direction",  # разрез: своя кривая на каждое значение колонки
    y="cum_share_of_balance",  # по Y — накопленная доля от баланса
    types=None,  # None = все значения (с учётом top_k и min_tx); или явный список
    top_k=None,  # None = без ограничения числа типов
    min_tx=0,  # 0 = не отсеивать редкие типы
    abs_amount=False,  # False = суммы со знаком
    title=None,  # None = «Винтаж по «tr_direction»: накопленная доля от баланса»
    tx_amount_col="transaction_amt",  # сумма транзакции
    # дальше — параметры maturation_transactions
    client_id="contract_number",
    tx_days_col="tx_days",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,  # по неделям — кривые глаже
    processed_dt=PROCESSED_DT,
    population=phx_base,  # общая база для всех типов
)
fig
""")
code("""
_, fig = cl.eda.vintage_by_type(
    df=phx,
    tx_type_col="tr_direction",
    y="cum_share_clients",  # конверсия в транзакцию по направлению
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
    processed_dt=PROCESSED_DT,
    population=phx_base,
)
fig
""")
code("""
data_type, fig = cl.eda.vintage_by_type(
    df=phx,
    tx_type_col="transaction_type_cd",  # тип операции (PAY, DDJ, ...)
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
    processed_dt=PROCESSED_DT,
    population=phx_base,
)
fig
""")
code("""
_, fig = cl.eda.vintage_by_type(
    df=phx,
    tx_type_col="transaction_type_cd",
    y="cum_share_clients",
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
    processed_dt=PROCESSED_DT,
    population=phx_base,
)
fig
""")
code("""
# проверка общей базы: сумма долей по типам = общий винтаж с тем же шагом (step=7)
total = cl.eda.maturation_transactions(
    df=phx, sample="PHX", client_id="contract_number", tx_days_col="tx_days",
    tx_amount_col="transaction_amt", balance_col="rtk_balance", retro_dt_col="rtk_send_date",
    horizon_days=730, step=7, processed_dt=PROCESSED_DT, population=phx_base,
)
check = pd.DataFrame({
    "сумма по типам": data_type.groupby("day")["cum_share_of_balance"].sum(),
    "общий винтаж": total.set_index("day")["cum_share_of_balance"],
})
check.iloc[[4, 13, 26, 52, -1]]  # ~1, 3, 6, 12 месяцев и конец горизонта
""")

md("""
## 4. Другие продукты, только крупные транзакции (> 10 000 ₽)

Фильтр по сумме убирает строки. Без `population` из базы выпали бы договоры, у которых не было
ни одной транзакции > 10 000 ₽, и доли посчитались бы от меньшей базы. С `population` база —
все договоры не-PHX, а в числитель идут только крупные транзакции.
""")
code("""
other = df[(df["financial_account_subtype_cd"] != "PHX")
           & (df["transaction_amt"].fillna(1e6) > 10_000)]  # fillna: оставить строки без транзакций
other_base = contracts[contracts["financial_account_subtype_cd"] != "PHX"]

_, fig = cl.eda.vintage_by_type(
    df=other,
    tx_type_col="transaction_type_cd",
    y="cum_share_of_balance",
    types=None,
    top_k=None,
    min_tx=0,
    abs_amount=False,
    title="Не-PHX, транзакции > 10 000 ₽: накопленная доля от баланса",  # свой заголовок
    tx_amount_col="transaction_amt",
    client_id="contract_number",
    tx_days_col="tx_days",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt=PROCESSED_DT,
    population=other_base,  # база — все договоры не-PHX, а не только те, что прошли фильтр
)
fig
""")
code("""
_, fig = cl.eda.vintage_by_type(
    df=other,
    tx_type_col="transaction_type_cd",
    y="cum_share_clients",
    types=None,
    top_k=None,
    min_tx=0,
    abs_amount=False,
    title="Не-PHX, транзакции > 10 000 ₽: конверсия в транзакцию",
    tx_amount_col="transaction_amt",
    client_id="contract_number",
    tx_days_col="tx_days",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt=PROCESSED_DT,
    population=other_base,
)
fig
""")

md("""
## 5. Продукты на одном графике — `vintage_by_segment`

Вместо цикла «один продукт — один график»: все продукты на одной картинке, **у каждого своя
база** (его договоры и баланс). Кривая продукта совпадает с `vintage` по его строкам. Малые
продукты дают шумные кривые — отсекаем их `min_contracts`. Продукт, по которому в выгрузке нет
ни одной транзакции, остаётся на графике нулевой линией — это не ошибка расчёта.
""")
code("""
positive = df[df["transaction_amt"].fillna(1) > 0]  # все продукты, без отрицательных транзакций

data_seg, fig = cl.eda.vintage_by_segment(
    df=positive,
    segment_col="financial_account_subtype_cd",  # продукт: своя кривая и своя база
    y="cum_share_of_balance",
    segments=None,  # None = все продукты по убыванию числа договоров; или явный список
    top_k=None,  # None = без ограничения числа продуктов
    min_contracts=100,  # не показывать продукты, где договоров меньше 100
    title="Recovery по продуктам: накопленная доля от баланса",
    client_id="contract_number",
    population=contracts,  # база; делится по продукту так же, как df
    # дальше — параметры maturation_transactions, общие для всех продуктов
    tx_days_col="tx_days",
    tx_amount_col="transaction_amt",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt=PROCESSED_DT,
)
fig
""")
code("""
_, fig = cl.eda.vintage_by_segment(
    df=positive,
    segment_col="financial_account_subtype_cd",
    y="cum_share_clients",
    segments=None,
    top_k=None,
    min_contracts=100,
    title="First pay rate по продуктам: конверсия в транзакцию",
    client_id="contract_number",
    population=contracts,
    tx_days_col="tx_days",
    tx_amount_col="transaction_amt",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt=PROCESSED_DT,
)
fig
""")
code("""
# сводка: база и итог на последнем дне горизонта каждого продукта
data_seg.groupby("sample").tail(1)[
    ["sample", "N_total", "day", "cum_share_of_balance", "cum_share_clients"]]
""")
code("""
# то же только по транзакциям > 5 000 ₽; база прежняя — через population
_, fig = cl.eda.vintage_by_segment(
    df=df[df["transaction_amt"].fillna(1e6) > 5_000],
    segment_col="financial_account_subtype_cd",
    y="cum_share_of_balance",
    segments=None,
    top_k=None,
    min_contracts=100,
    title="Recovery по продуктам, транзакции > 5 000 ₽",
    client_id="contract_number",
    population=contracts,
    tx_days_col="tx_days",
    tx_amount_col="transaction_amt",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt=PROCESSED_DT,
)
fig
""")

md("""
### Когорты: тот же `vintage_by_segment`, разрез по дате

Сегмент — любая колонка уровня договора. Квартал отправки в РТК показывает, меняется ли
скорость и уровень поступлений у новых когорт (дрейф). Горизонт у каждой когорты свой: поздние
когорты короче.
""")
code("""
contracts["cohort"] = contracts["rtk_send_date"].dt.to_period("Q").astype(str)
phx_c = phx.merge(contracts[["contract_number", "cohort"]], on="contract_number")

_, fig = cl.eda.vintage_by_segment(
    df=phx_c,
    segment_col="cohort",  # квартал отправки в РТК
    y="cum_share_of_balance",
    segments=sorted(phx_c["cohort"].unique()),  # явный список — в хронологическом порядке
    top_k=None,
    min_contracts=0,
    title="PHX по кварталам отправки в РТК: накопленная доля от баланса",
    client_id="contract_number",
    population=contracts[contracts["financial_account_subtype_cd"] == "PHX"],  # есть cohort
    tx_days_col="tx_days",
    tx_amount_col="transaction_amt",
    balance_col="rtk_balance",
    retro_dt_col="rtk_send_date",
    horizon_days=730,
    step=7,
    processed_dt=PROCESSED_DT,
)
fig
""")

md("""
## 6. Разрез по типам транзакций внутри каждого продукта

Цикл, как в `sample_overview`, но только по продуктам с достаточной базой. Один график на продукт:
типы транзакций делят базу этого продукта.
""")
code("""
sizes = contracts["financial_account_subtype_cd"].value_counts()
for product in sizes[sizes >= 100].index:
    rows = positive[positive["financial_account_subtype_cd"] == product]
    if rows["transaction_amt"].notna().sum() == 0:  # vintage_by_type нужна хоть одна транзакция
        print(f"{product}: нет транзакций — пропускаем")
        continue
    _, fig = cl.eda.vintage_by_type(
        df=rows,
        tx_type_col="transaction_type_cd",
        y="cum_share_of_balance",
        types=None,
        top_k=None,
        min_tx=0,
        abs_amount=False,
        title=f"{product}: накопленная доля от баланса по типам транзакций",
        tx_amount_col="transaction_amt",
        client_id="contract_number",
        tx_days_col="tx_days",
        balance_col="rtk_balance",
        retro_dt_col="rtk_send_date",
        horizon_days=730,
        step=7,
        processed_dt=PROCESSED_DT,
        population=contracts[contracts["financial_account_subtype_cd"] == product],
    )
    fig.show()
""")

md("""
## 7. Свой набор кривых: `maturation_transactions` + `plot_vintage`

Когда нужно сравнить на одном графике выборки, которые не укладываются в одну колонку-разрез,
например PHX «все транзакции» против «только > 5 000 ₽». `maturation_transactions` считает
таблицу для одной выборки, `plot_vintage` рисует все выборки из объединённой таблицы (по
колонке `sample`).
""")
code("""
parts = [
    cl.eda.maturation_transactions(
        df=sub,
        sample=label,  # имя кривой в легенде
        client_id="contract_number",
        tx_days_col="tx_days",
        tx_amount_col="transaction_amt",
        balance_col="rtk_balance",
        retro_dt_col="rtk_send_date",
        horizon_days=730,
        step=7,
        processed_dt=PROCESSED_DT,
        population=phx_base,  # одна база — кривые сравнимы
    )
    for label, sub in [
        ("PHX: все транзакции", phx),
        ("PHX: транзакции > 5 000 ₽", phx[phx["transaction_amt"].fillna(1e6) > 5_000]),
        ("PHX: транзакции > 10 000 ₽", phx[phx["transaction_amt"].fillna(1e6) > 10_000]),
    ]
]
cl.eda.plot_vintage(
    res=pd.concat(parts, ignore_index=True),  # таблицы maturation_transactions подряд
    y="cum_share_of_balance",
    title="PHX: вклад крупных транзакций",  # None = «Винтаж: <подпись y>»
    x_title="Дней от rtk_send_date",
    legend_title="выборка",  # заголовок легенды
)
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parents[1] / "02_vintage_overview.ipynb"
nbf.write(nb, out)
print("записан", out)
