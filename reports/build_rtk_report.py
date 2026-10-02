"""Собирает reports/rtk_report.ipynb — отчёт «Включение в РТК: поступления, отбор, эффект».

    python reports/build_rtk_report.py     # пересобрать ноутбук после правки этого файла
    python reports/run_report.py           # выполнить и получить reports/out/rtk_report.html

Настройки (источник данных, продукты, период, затраты) — в первой ячейке ноутбука и в
reports/settings_local.py. Все цифры в тексте считаются из данных при выполнении. Выполненный
ноутбук и HTML содержат статистики рабочих данных — в git они не попадают (reports/out/).
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ------------------------------------------------------------------ настройки
md("""
> **Настройки — в следующей ячейке.** Остальной код менять не нужно. Чтобы не править ноутбук,
> те же переменные можно задать в файле `settings_local.py` рядом с ноутбуком — он выполняется
> поверх настроек (пример — `settings_local.example.py`).
""")
code("""
# ======================= НАСТРОЙКИ =======================
SOURCE = "csv"                     # "csv" — данные из файлов; "query" — запросом через load_query()
TX_CSV = "data/df_1.csv"           # транзакции: строка = договор без транзакций либо одна транзакция
FEATURES_CSV = "data/df_agg.csv"   # признаки договоров на дату включения: строка = договор
TX_QUERY = "select * from <схема>.<таблица транзакций>"
FEATURES_QUERY = "select * from <схема>.<таблица признаков>"
CACHE_DIR = "data_cache"           # куда сохранять результат запросов; None — не сохранять
REFRESH = False                    # True — выполнить запросы заново, даже если есть сохранённый результат

PRODUCT = "PHX"                    # продукт для подробного разбора (часть I, заголовок)
MODEL_PRODUCTS = None              # продукты для модели: None — все, по которым есть признаки; или список
RETRO_TO = None                    # договоры, включённые не позже этой даты; None — все с полными 2 годами
OOT_FROM = "2024-01-01"            # с этой даты включения — проверочная выборка (модель её не видит)
MIN_CONTRACTS = 100                # продукты с меньшим числом договоров на графики не выносятся
DUTY_SHARE = 0.5                   # доля имущественной пошлины: 0.5 — включение в реестр
OVERHEAD = 0                       # ₽ на договор сверх пошлины (подготовка и подача заявления)


def load_query(sql):
    # Нужна только при SOURCE = "query": вернуть pandas.DataFrame по тексту запроса.
    # Задайте её в settings_local.py под свою среду (пример — в settings_local.example.py).
    raise NotImplementedError("Задайте load_query(sql) в settings_local.py")


import os
from pathlib import Path

_settings = Path(os.environ.get("RTK_REPORT_SETTINGS", "settings_local.py"))
if _settings.exists():
    exec(_settings.read_text(encoding="utf-8"))
    print("настройки дополнены из", _settings)
""")

# ------------------------------------------------------------------ расчёты (в HTML скрыты)
code("""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML, Markdown, display
from plotly.subplots import make_subplots

import collection_lab as cl

pio.renderers.default = "notebook"      # plotly.js встраивается в файл: отчёт открывается без сети

H = 730                                  # горизонт наблюдения, дней (2 года)
MONTH = 365.25 / 12

# оформление: чернила для текста, один синий для величин, фиксированный порядок для категорий
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e9e8e4", "#c9c8c2"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # светлый → тёмный
FONT = "Segoe UI, Arial, sans-serif"


def seq_colors(n):
    # n оттенков одного синего от светлого к тёмному (для упорядоченных групп: кварталы)
    from plotly.colors import sample_colorscale
    return sample_colorscale([[0, SEQ[0]], [1, SEQ[-1]]], list(np.linspace(0, 1, max(n, 2))))[:n]


def load_table(name, csv_path, query):
    if SOURCE == "csv":
        return pd.read_csv(csv_path, low_memory=False)
    cache = Path(CACHE_DIR) / f"{name}.csv" if CACHE_DIR else None
    if cache is not None and cache.exists() and not REFRESH:
        return pd.read_csv(cache, low_memory=False)
    table = load_query(query)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
    return table


def require(table, columns, name):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"В таблице «{name}» нет колонок: {', '.join(missing)}")


def num(x):
    return f"{x:,.0f}".replace(",", " ")


def dec(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def pct(x, d=1):
    return dec(x * 100, d) + " %"


def mln(x, d=0):
    return dec(x / 1e6, d) + " млн ₽"


def signed(x, d=1):
    return ("+" if x > 0 else "−" if x < 0 else "") + dec(abs(x) / 1e6, d) + " млн ₽"


def layout(fig, title, subtitle=None, height=420, top=84, **kw):
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:13px;color:{INK2}'>{subtitle}</span>"
    fig.update_layout(
        template="plotly_white", height=height, font={"family": FONT, "size": 13, "color": INK2},
        title={"text": text, "x": 0, "xanchor": "left", "font": {"size": 17, "color": INK}},
        margin={"l": 64, "r": 36, "t": top, "b": 56}, plot_bgcolor="white", paper_bgcolor="white",
        separators=", ",
        hoverlabel={"bgcolor": "white", "font": {"family": FONT, "color": INK}}, **kw)
    fig.update_xaxes(showgrid=False, linecolor=MUTED, ticks="outside", tickcolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


MONTH_TICKS = {"tickvals": [round(m * MONTH) for m in range(0, 25, 3)],
               "ticktext": [str(m) for m in range(0, 25, 3)]}


def month_axis(fig, **kw):
    fig.update_xaxes(title_text="месяцев после включения в РТК", range=[0, H + 12], **MONTH_TICKS, **kw)
    return fig
""")

code("""
raw = load_table("transactions", TX_CSV, TX_QUERY)
require(raw, ["contract_number", "financial_account_subtype_cd", "rtk_send_date", "rtk_balance",
              "real_transaction_dttm", "transaction_amt"], "транзакции")
raw["rtk_send_date"] = pd.to_datetime(raw["rtk_send_date"], format="mixed").dt.normalize()
raw["rtk_balance"] = pd.to_numeric(raw["rtk_balance"], errors="coerce")
raw["transaction_amt"] = pd.to_numeric(raw["transaction_amt"], errors="coerce")
raw = cl.eda.add_days_since(raw, "real_transaction_dttm", "rtk_send_date", "tx_days")
last_tx = pd.to_datetime(raw["real_transaction_dttm"], format="mixed", errors="coerce").max().normalize()
RETRO_TO = pd.Timestamp(RETRO_TO) if RETRO_TO is not None else last_tx - pd.Timedelta(days=H)
if PRODUCT not in set(raw["financial_account_subtype_cd"]):
    raise ValueError(f"В транзакциях нет продукта {PRODUCT!r} (настройка PRODUCT)")

all_product = raw[raw["financial_account_subtype_cd"] == PRODUCT]
flow = (all_product.drop_duplicates("contract_number")
        .groupby(all_product["rtk_send_date"].dt.to_period("M")).size())

df = all_product[all_product["rtk_send_date"] <= RETRO_TO]
if (last_tx - df["rtk_send_date"].max()).days < H:
    raise ValueError(f"RETRO_TO = {RETRO_TO:%d.%m.%Y}: не у всех договоров прошло {H} дней до {last_tx:%d.%m.%Y}")

# одна строка на договор
contracts = df.drop_duplicates("contract_number")[
    ["contract_number", "rtk_balance", "rtk_send_date"]].copy()
tx = df[df["transaction_amt"].notna() & df["tx_days"].between(0, H)]
pos = tx[tx["transaction_amt"] > 0]
contracts["money"] = contracts["contract_number"].map(
    tx.groupby("contract_number")["transaction_amt"].sum()).fillna(0.0)
contracts["first_day"] = contracts["contract_number"].map(
    pos.groupby("contract_number")["tx_days"].min())
contracts["paid"] = contracts["first_day"].notna()
contracts["rate"] = (contracts["money"] / contracts["rtk_balance"]).clip(lower=0)
contracts["cohort"] = contracts["rtk_send_date"].dt.to_period("Q").astype(str)
BAL_LABELS = ["до 50 тыс", "50–100 тыс", "100–200 тыс", "200–300 тыс", "свыше 300 тыс"]
contracts["bal_group"] = pd.cut(contracts["rtk_balance"], [0, 50e3, 100e3, 200e3, 300e3, np.inf],
                                labels=BAL_LABELS)

N = len(contracts)
BALANCE = contracts["rtk_balance"].sum()
MONEY = contracts["money"].sum()
RECOVERY = MONEY / BALANCE
PAID = contracts["paid"].mean()
date_from, date_to = contracts["rtk_send_date"].min(), contracts["rtk_send_date"].max()

# кривые: накопленная доля от долга (все транзакции, с учётом возвратов) и доля заплативших
KW = {"client_id": "contract_number", "tx_days_col": "tx_days", "tx_amount_col": "transaction_amt",
      "balance_col": "rtk_balance", "retro_dt_col": "rtk_send_date", "horizon_days": H,
      "processed_dt": last_tx}
curve = cl.eda.maturation_transactions(df, PRODUCT, step=1, population=contracts, **KW)
conv = cl.eda.maturation_transactions(df[df["transaction_amt"].fillna(1) > 0], PRODUCT, step=1,
                                      population=contracts, **KW)
MARKS = [(6, 183), (12, 365), (18, 548), (24, 730)]


def at(table, col, day):
    return float(table.loc[table["day"] == day, col].iloc[0])


half_day = int(curve.loc[curve["cum_tx_sum"] >= 0.5 * curve["cum_tx_sum"].iloc[-1], "day"].iloc[0])
first_year_share = at(curve, "cum_tx_sum", 365) / at(curve, "cum_tx_sum", H)

# поступления по месяцам после включения
by_month = (tx.assign(m=(tx["tx_days"] // MONTH).astype(int).clip(upper=23) + 1)
            .groupby("m")["transaction_amt"].sum().reindex(range(1, 25), fill_value=0.0))
peak_month = int(by_month.idxmax())
start_month = int((by_month >= 0.5 * by_month.max()).idxmax())   # первый «полноценный» месяц
quiet_share = by_month.loc[:start_month - 1].sum() / by_month.sum()
year1_rest_share = by_month.loc[start_month:12].sum() / by_month.sum()
year2_share = by_month.loc[13:].sum() / by_month.sum()
first_pay_median = contracts["first_day"].median() / MONTH

# концентрация: сколько вернул договор
RATE_LABELS = ["ничего", "до 5 % долга", "5–50 %", "50–95 %", "95 % и больше"]
contracts["rate_group"] = pd.cut(contracts["rate"].where(contracts["money"] > 0, -1),
                                 [-np.inf, 0, 0.05, 0.5, 0.95, np.inf], labels=RATE_LABELS)
conc = contracts.groupby("rate_group", observed=False).agg(
    n=("money", "size"), money=("money", lambda s: s.clip(lower=0).sum()))
conc["share_n"] = conc["n"] / conc["n"].sum()
conc["share_money"] = conc["money"] / conc["money"].sum()

# размер долга
by_bal = contracts.groupby("bal_group", observed=False).agg(
    n=("money", "size"), balance=("rtk_balance", "sum"), money=("money", "sum"), paid=("paid", "mean"))
by_bal["recovery"] = by_bal["money"] / by_bal["balance"]
by_bal["money_per_contract"] = by_bal["money"] / by_bal["n"]

# когорты
by_cohort = contracts.groupby("cohort").agg(
    n=("money", "size"), balance=("rtk_balance", "sum"), money=("money", "sum"), paid=("paid", "mean"))
by_cohort["recovery"] = by_cohort["money"] / by_cohort["balance"]
cohorts = by_cohort.index.tolist()
by_year = contracts.groupby(contracts["rtk_send_date"].dt.year).agg(
    balance=("rtk_balance", "sum"), money=("money", "sum"), paid=("paid", "mean"))
by_year["recovery"] = by_year["money"] / by_year["balance"]
y_first, y_last = by_year.iloc[0], by_year.iloc[-1]
year_drop = 1 - y_last["recovery"] / y_first["recovery"]
LESS = "меньше" if year_drop > 0 else "больше"
LOWER = "ниже" if year_drop > 0 else "выше"
df_c = df.merge(contracts[["contract_number", "cohort"]], on="contract_number")
cohort_curves, _ = cl.eda.vintage_by_segment(df_c, "cohort", segments=cohorts, population=contracts,
                                             step=7, **KW)

# размер транзакции
TX_LABELS = ["до 1 тыс ₽", "1–5 тыс", "5–10 тыс", "10–50 тыс", "свыше 50 тыс"]
pos_g = pos.assign(g=pd.cut(pos["transaction_amt"], [0, 1e3, 5e3, 10e3, 50e3, np.inf], labels=TX_LABELS))
by_tx = pos_g.groupby("g", observed=False)["transaction_amt"].agg(n="size", money="sum")
by_tx["share_n"] = by_tx["n"] / by_tx["n"].sum()
by_tx["share_money"] = by_tx["money"] / by_tx["money"].sum()

# продукты (для сравнения)
all_main = raw[raw["rtk_send_date"] <= RETRO_TO]
prod_contracts = all_main.drop_duplicates("contract_number")[
    ["contract_number", "rtk_balance", "financial_account_subtype_cd"]]
prod_curves, _ = cl.eda.vintage_by_segment(
    all_main, "financial_account_subtype_cd", min_contracts=MIN_CONTRACTS, population=prod_contracts,
    step=7, **KW)
prod_last = prod_curves.groupby("sample", sort=False).tail(1).set_index("sample")
prod_last = prod_last[prod_last["cum_tx_cnt"] > 0]
no_tx_products = sorted(set(prod_curves["sample"]) - set(prod_last.index))
""")


code("""
# ---------- экономика включения, модель и правило отбора
RED = "#e34948"
PROD = "financial_account_subtype_cd"


def duty_scale(claim):
    # имущественная пошлина, подп. 1 п. 1 ст. 333.21 НК РФ (ред. 259-ФЗ)
    x = np.asarray(claim, dtype=float)
    d = np.select(
        [x <= 100_000, x <= 1_000_000, x <= 10_000_000, x <= 50_000_000],
        [10_000.0, 10_000 + 0.05 * (x - 100_000), 55_000 + 0.03 * (x - 1_000_000),
         325_000 + 0.01 * (x - 10_000_000)],
        default=725_000 + 0.005 * (x - 50_000_000))
    return np.minimum(d, 10_000_000)


def cost(balance, share=DUTY_SHARE, overhead=OVERHEAD):
    return share * duty_scale(balance) + overhead


# договоры всех продуктов: поступления за 2 года, пошлина, результат
pc = all_main.drop_duplicates("contract_number")[
    ["contract_number", "rtk_send_date", "rtk_balance", PROD]].copy()
tx_all = all_main[all_main["transaction_amt"].notna() & all_main["tx_days"].between(0, H)]
pc["money"] = pc["contract_number"].map(
    tx_all.groupby("contract_number")["transaction_amt"].sum()).fillna(0.0)
pc["rate"] = (pc["money"] / pc["rtk_balance"]).clip(lower=0)
pc["duty"] = cost(pc["rtk_balance"])
pc["profit"] = pc["money"] - pc["duty"]
pc["payoff"] = (pc["profit"] > 0).astype(int)

# признаки: модель строится по продуктам, для которых они есть
agg = load_table("features", FEATURES_CSV, FEATURES_QUERY)
require(agg, ["contract_number"], "признаки")
agg = agg.drop_duplicates("contract_number")
pc["has_features"] = pc["contract_number"].isin(agg["contract_number"])
coverage = pc.groupby(PROD).agg(n=("money", "size"), with_features=("has_features", "sum"))
available = coverage.index[coverage["with_features"] > 0].tolist()
flags = pc.groupby(PROD).agg(money=("money", "sum"), balance=("rtk_balance", "sum"))
flags["tx"] = tx_all.groupby(PROD).size().reindex(flags.index).fillna(0)
suspect = flags.index[(flags["tx"] == 0) | (flags["money"] > flags["balance"])].tolist()
auto_excluded = [] if MODEL_PRODUCTS else [x for x in available if x in suspect and x != PRODUCT]
model_products = [x for x in (MODEL_PRODUCTS or available) if x in available and x not in auto_excluded]
if PRODUCT not in model_products:
    raise ValueError(f"По продукту {PRODUCT!r} нет признаков — модель построить нельзя")
model_products = (pc[pc[PROD].isin(model_products) & pc["has_features"]][PROD]
                  .value_counts().index.tolist())                      # по убыванию числа договоров
MULTI = len(model_products) > 1
no_feature_products = [x for x in coverage.index if x not in model_products]

base = (pc[pc[PROD].isin(model_products)].drop(columns=["has_features"])
        .merge(agg.drop(columns=[c for c in agg.columns if c in pc.columns and c != "contract_number"]),
               on="contract_number"))
base["pay_any"] = (base["money"] >= 1).astype(int)
base["old_target"] = (base["rate"] > 0.05).astype(int)       # текущий таргет: вернули > 5 % долга
SERVICE = {"contract_number", "rtk_send_date", "target", "last_col_end_dt", "job_organization_nm",
           "money", "rate", "duty", "profit", "payoff", "pay_any", "old_target"}
if not MULTI:
    SERVICE.add(PROD)                                           # один продукт — признак бесполезен
FEATURES = [c for c in base.columns if c not in SERVICE]
num_cols, cat_cols = cl.data.split_feature_types(base, FEATURES)
base["product"] = base[PROD].astype(str)                      # копия: PROD может стать категорией
base = cl.data.cast_types(base, num_cols, cat_cols)
base["bal_group"] = pd.cut(base["rtk_balance"], [0, 50e3, 100e3, 200e3, 300e3, np.inf], labels=BAL_LABELS)

split = cl.data.time_split(base, "pay_any", "rtk_send_date", oot_from=OOT_FROM, val_size=0.2)
tr, va, te = split.train.copy(), split.val.copy(), split.test.copy()
if min(len(tr), len(va), len(te)) == 0:
    raise ValueError(f"OOT_FROM = {OOT_FROM}: одна из выборок пуста — выберите другую дату")
OOT = pd.Timestamp(OOT_FROM)
P_TRAIN, P_TEST = f"до {OOT:%d.%m.%Y}", f"с {OOT:%d.%m.%Y}"   # подписи периодов
PARAMS = {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 16,
          "min_child_samples": 200, "colsample_bytree": 0.7}


def fit(y_tr, y_va, rows_tr=None, rows_va=None, features=None):
    features = features or FEATURES
    a = tr if rows_tr is None else tr[rows_tr]
    b = va if rows_va is None else va[rows_va]
    m = cl.core.make_model("lgbm", PARAMS, task="binary", early_stopping_rounds=100)
    return m.fit(a[features], y_tr, eval_set=(b[features], y_va),
                 cat_features=[c for c in cat_cols if c in features])


model_new = fit(tr["payoff"], va["payoff"])
model_old = fit(tr["old_target"], va["old_target"])
for part in (tr, va, te):
    part["score_new"] = model_new.predict(part[FEATURES])
    part["score_old"] = model_old.predict(part[FEATURES])


def policy(score_va, profit_va, score_te):
    # порог = максимум накопленной прибыли на валидации; применяем к проверочной выборке
    order = np.argsort(-score_va, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(profit_va[order])])
    k = int(np.argmax(cum))
    thr = np.inf if k == 0 else score_va[order][k - 1]
    return score_te >= thr, thr, k / len(score_va)


def summary(mask):
    d = te[mask]
    return {"n": int(mask.sum()), "share": float(mask.mean()), "duty": d["duty"].sum(),
            "money": d["money"].sum(), "profit": d["profit"].sum(),
            "payoff": d["payoff"].mean() if len(d) else np.nan}


p_va, p_te = va["profit"].to_numpy(), te["profit"].to_numpy()
mask_new, thr_new, share_val_new = policy(va["score_new"].to_numpy(), p_va, te["score_new"].to_numpy())
mask_old, thr_old, _ = policy(va["score_old"].to_numpy(), p_va, te["score_old"].to_numpy())
res = {"all": summary(np.ones(len(te), bool)), "old": summary(mask_old), "new": summary(mask_new),
       "oracle": summary(p_te > 0)}
T = len(te)
per_1000 = res["new"]["profit"] / T * 1000
per_1000_txt = mln(per_1000, 1) if abs(per_1000) >= 1e6 else f"{num(per_1000 / 1000)} тыс ₽"

# неопределённость: бутстреп по договорам проверочной выборки
rng = np.random.default_rng(0)
idx = rng.integers(0, T, size=(1000, T))
boot_new = (p_te[idx] * mask_new[idx]).sum(axis=1)
boot_diff = boot_new - (p_te[idx] * mask_old[idx]).sum(axis=1)
ci_new = np.percentile(boot_new, [5, 95])
ci_diff = np.percentile(boot_diff, [5, 95])

# качество
auc = {name: cl.metrics.roc_auc(d["payoff"], d["score_new"]) for name, d in
       (("обучение", tr), ("настройка порога", va), ("проверка", te))}
te["decile"] = pd.qcut(te["score_new"].rank(method="first", ascending=False), 10, labels=range(1, 11))
dec_t = te.groupby("decile", observed=False).agg(
    n=("profit", "size"), payoff=("payoff", "mean"), money=("money", "mean"), duty=("duty", "mean"),
    profit=("profit", "mean"), profit_sum=("profit", "sum"))
importance = model_new.feature_importance("gain")
importance = (importance / importance.sum()).sort_values(ascending=False)


def cum_curve(score):
    order = np.argsort(-score, kind="stable")
    return np.arange(1, T + 1) / T, np.cumsum(p_te[order]) / 1e6


# экономика «по всем» и по размеру долга
periods = {f"обучение ({P_TRAIN})": pd.concat([tr, va]), f"проверка ({P_TEST})": te}
econ = pd.DataFrame({k: {"n": len(d), "money": d["money"].sum(), "duty": d["duty"].sum(),
                         "profit": d["profit"].sum(), "payoff": d["payoff"].mean(),
                         "money_avg": d["money"].mean(), "duty_avg": d["duty"].mean()}
                     for k, d in periods.items()}).T
econ_bal = base.groupby("bal_group", observed=False).agg(
    n=("money", "size"), money=("money", "mean"), duty=("duty", "mean"), payoff=("payoff", "mean"))

# если затраты выше: модель переобучается под каждый сценарий
SCENARIOS = [("как в расчёте", DUTY_SHARE, OVERHEAD),
             ("+ 2 000 ₽ расходов на подачу", DUTY_SHARE, OVERHEAD + 2_000),
             ("+ 5 000 ₽ расходов на подачу", DUTY_SHARE, OVERHEAD + 5_000),
             ("полная имущественная пошлина", 1.0, OVERHEAD)]
sens = {}
for name, share, overhead in SCENARIOS:
    if (share, overhead) == (DUTY_SHARE, OVERHEAD):
        sens[name] = {"all": res["all"]["profit"], "new": res["new"]["profit"], "share": res["new"]["share"]}
        continue
    pr = {k: (d["money"] - cost(d["rtk_balance"], share, overhead)).to_numpy()
          for k, d in (("tr", tr), ("va", va), ("te", te))}
    m = fit((pr["tr"] > 0).astype(int), (pr["va"] > 0).astype(int))
    msk, _, _ = policy(m.predict(va[FEATURES]), pr["va"], m.predict(te[FEATURES]))
    sens[name] = {"all": pr["te"].sum(), "new": pr["te"][msk].sum(), "share": msk.mean()}
sens = pd.DataFrame(sens).T


# результат по продуктам на проверочной выборке (общая модель, общий порог)
te["take"] = mask_new
by_product = te.groupby("product").agg(
    n=("profit", "size"), all_profit=("profit", "sum"), take=("take", "sum"),
    payoff=("payoff", "mean")).reindex(model_products).dropna(subset=["n"])
taken = te[te["take"]].groupby("product").agg(
    duty=("duty", "sum"), money=("money", "sum"), profit=("profit", "sum"), payoff_take=("payoff", "mean"))
by_product = by_product.join(taken).fillna({"duty": 0.0, "money": 0.0, "profit": 0.0})
by_product["share"] = by_product["take"] / by_product["n"]

# общая модель против модели только по основному продукту (на его проверочных договорах)
solo = None
if MULTI:
    r_tr, r_va, r_te = [(d["product"] == PRODUCT).to_numpy() for d in (tr, va, te)]
    solo_features = [c for c in FEATURES if c != PROD]
    m_solo = fit(tr["payoff"][r_tr], va["payoff"][r_va], r_tr, r_va, solo_features)
    msk, _, _ = policy(m_solo.predict(va[r_va][solo_features]), p_va[r_va],
                       m_solo.predict(te[r_te][solo_features]))
    solo = {"solo": p_te[r_te][msk].sum(), "joint": p_te[r_te & mask_new].sum(),
            "all": p_te[r_te].sum(), "n": int(r_te.sum())}

# экономика включения по всем продуктам
tx_cnt = tx_all.groupby("financial_account_subtype_cd").size()
prod_econ = pc.groupby("financial_account_subtype_cd").agg(
    n=("money", "size"), balance=("rtk_balance", "sum"), money=("money", "sum"), duty=("duty", "sum"),
    profit=("profit", "sum"), payoff=("payoff", "mean"),
    with_features=("has_features", "sum")).sort_values("n", ascending=False)
prod_econ["tx"] = tx_cnt.reindex(prod_econ.index).fillna(0).astype(int)
ALL_WORD = "убыточно" if res["all"]["profit"] < 0 else "прибыльно"
GAIN = res["new"]["profit"] - res["all"]["profit"]
if res["all"]["profit"] < 0:
    MARGIN_TXT = ("Основной эффект модели — не заработок, а отказ от убыточных включений: "
                  f"{signed(GAIN, 0)} разницы с включением по всем.")
else:
    MARGIN_TXT = (f"Включение по всем и так окупается; отбор добавляет к нему {signed(GAIN)} за счёт "
                  "отказа от договоров, которые пошлину не окупят.")
NO_FEATURES_NOTE = (f"Модель построена по продуктам: {', '.join(model_products)}. По остальным "
                    f"({', '.join(no_feature_products)}) признаков в выгрузке нет."
                    if no_feature_products else f"Модель построена по всем продуктам: {', '.join(model_products)}.")
others_econ = prod_econ.drop(index=PRODUCT)
others_odd = others_econ[others_econ["money"] > others_econ["balance"]]   # поступлений больше, чем долга
others_no_tx = others_econ[others_econ["tx"] == 0]
others_with_tx = others_econ.drop(index=others_odd.index.union(others_no_tx.index))
others_not_paid = 1 - (others_with_tx["payoff"] * others_with_tx["n"]).sum() / others_with_tx["n"].sum()


def show_tiles(tiles):
    cards = "".join(
        f'<div style="flex:1 1 200px;border:1px solid {GRID};border-radius:10px;padding:16px 18px;background:#fff">'
        f'<div style="font-size:30px;font-weight:650;color:{INK};line-height:1.1">{v}</div>'
        f'<div style="font-size:14px;color:{INK};margin-top:6px">{label}</div>'
        f'<div style="font-size:12.5px;color:{INK2};margin-top:4px">{note}</div></div>'
        for v, label, note in tiles)
    display(HTML(f'<div style="display:flex;gap:14px;flex-wrap:wrap;font-family:{FONT};margin:8px 0 4px">{cards}</div>'))

""")

# ------------------------------------------------------------------ шапка и главное
code("""
display(Markdown(f'''
# Включение в реестр требований кредиторов: поступления, отбор договоров, эффект

**Продукт {PRODUCT} (модель: {", ".join(model_products)}) · договоры, включённые в РТК с {date_from:%d.%m.%Y} по {date_to:%d.%m.%Y} ·
наблюдение 2 года по каждому договору · данные по {last_tx:%d.%m.%Y}**

**Вопрос.** Включение в реестр теперь стоит денег (госпошлина). По каким договорам включаться,
чтобы поступления окупали пошлину, и сколько на этом можно заработать?

**Содержание.** Часть I — что происходит после включения: сколько денег приходит, когда и от кого
(разделы 1–5). Часть II — экономика включения с учётом пошлины, модель отбора договоров и её
эффект в деньгах (разделы 6–8). В конце — как считали и что важно помнить.
'''))

tiles = [
    (num(N), "договоров", f"общий долг {dec(BALANCE / 1e9, 2)} млрд ₽"),
    (pct(RECOVERY), "долга возвращается за 2 года", mln(MONEY)),
    (pct(PAID, 0), "договоров дают хотя бы один платёж", f"{pct(1 - PAID, 0)} не платят ничего"),
    (f"{dec(half_day / MONTH, 0)} мес", "проходит до половины всех денег",
     f"за первый год — {pct(first_year_share, 0)} от суммы за 2 года"),
]
cards = "".join(
    f'<div style="flex:1 1 200px;border:1px solid {GRID};border-radius:10px;padding:16px 18px;background:#fff">'
    f'<div style="font-size:30px;font-weight:650;color:{INK};line-height:1.1">{v}</div>'
    f'<div style="font-size:14px;color:{INK};margin-top:6px">{label}</div>'
    f'<div style="font-size:12.5px;color:{INK2};margin-top:4px">{note}</div></div>'
    for v, label, note in tiles)
display(HTML(f'<div style="display:flex;gap:14px;flex-wrap:wrap;font-family:{FONT};margin:8px 0 4px">{cards}</div>'))
show_tiles([
    (pct(base["payoff"].mean(), 0), "договоров окупают пошлину",
     f"пошлина в среднем {num(base['duty'].mean())} ₽, поступления — {num(base['money'].mean())} ₽"),
    (signed(res["all"]["profit"], 0), "если включаться по всем", f"{num(T)} договоров, включённых {P_TEST}"),
    (signed(res["new"]["profit"]), "если отбирать моделью",
     f"включаемся по {pct(res['new']['share'], 0)} договоров"),
    (signed(res["new"]["profit"] - res["all"]["profit"], 0), "эффект отбора",
     "разница между «по всем» и «по модели»"),
])
""")

code("""
top = conc.loc["95 % и больше"]
zero = conc.loc["ничего"]
first_c, last_c = by_cohort.iloc[0], by_cohort.iloc[-1]
small, big = by_bal.iloc[0], by_bal.iloc[-1]
display(Markdown(f'''
## Главное

**Что происходит после включения**

1. **Возвращается немного и небыстро.** За 2 года после включения в реестр приходит
   **{pct(RECOVERY)}** от суммы долга — {mln(MONEY)} из {dec(BALANCE / 1e9, 2)} млрд ₽. Первые
   {start_month - 1} месяца денег почти нет, поток начинается с {start_month}-го месяца; половина
   суммы за 2 года набирается к {dec(half_day / MONTH, 0)}-му месяцу.
2. **Платит меньшинство.** Хотя бы один платёж дают **{pct(PAID, 0)}** договоров,
   {pct(1 - PAID, 0)} за 2 года не платят ничего.
3. **Деньги сосредоточены в малой части договоров.** {pct(top["share_n"])} договоров гасят долг
   практически полностью и дают **{pct(top["share_money"], 0)}** всех поступлений.
4. **Мелкие долги возвращают бóльшую долю, но мало денег.** У долгов до 50 тыс ₽ возвращается
   {pct(small["recovery"])} долга — это {num(small["money_per_contract"])} ₽ на договор; у долгов
   свыше 300 тыс ₽ — {pct(big["recovery"])}, но уже {num(big["money_per_contract"])} ₽ на договор.
5. **Договоры {y_last.name} года возвращают {LESS}, чем {y_first.name}-го.** Доля возврата за 2 года —
   {pct(y_last["recovery"])} против {pct(y_first["recovery"])} (на {pct(abs(year_drop), 0)} {LOWER}), доля
   платящих — {pct(y_last["paid"], 0)} против {pct(y_first["paid"], 0)}.

**Отбор договоров и эффект**

6. **Включаться по всем договорам при нынешней пошлине {ALL_WORD}.** Пошлина — от 5 000 ₽ за договор,
   в среднем {num(base["duty"].mean())} ₽; поступления за 2 года — в среднем
   {num(base["money"].mean())} ₽. Пошлину окупают {pct(econ["payoff"].iloc[0], 0)} договоров, включённых
   {P_TRAIN}, и {pct(econ["payoff"].iloc[-1], 0)} — включённых {P_TEST}. По {num(T)} договорам
   проверочного периода включение по всем дало бы **{signed(res["all"]["profit"], 0)}**.
7. **Модель отбирает договоры, по которым включение окупается.** Она оценивает вероятность того,
   что поступления за 2 года превысят пошлину. Включаемся по {pct(res["new"]["share"], 0)} договоров с
   наибольшей оценкой — среди них пошлину окупают {pct(res["new"]["payoff"], 0)}, против
   {pct(res["all"]["payoff"], 0)} в среднем.
8. **Результат на договорах проверочного периода: {signed(res["new"]["profit"])}** вместо
   {signed(res["all"]["profit"], 0)} — это {per_1000_txt} на каждую 1 000 рассмотренных договоров.
   Текущая модель (таргет «вернули больше 5 % долга») дала бы {signed(res["old"]["profit"])}.
9. **Откуда эффект и насколько он устойчив.** {MARGIN_TXT} При дополнительных 2 000 ₽ расходов
   на подачу результат отбора — {signed(sens["new"].iloc[1])} (вместо {signed(res["new"]["profit"])}).
   Оценка — сверху: все поступления считаются результатом включения.
10. **Остальные продукты: результат включения по всем.** На {num(others_with_tx["n"].sum())}
    договорах ({", ".join(others_with_tx.index)}) включение по всем дало бы
    **{signed(others_with_tx["profit"].sum(), 0)}**. {NO_FEATURES_NOTE}
'''))
""")

# ------------------------------------------------------------------ 1. сколько и когда
md("""
# Часть I. Что происходит после включения в реестр

## 1. Сколько возвращается и когда

Кривая показывает, какая доля суммарного долга вернулась к каждому месяцу после включения в реестр.
""")
code("""
fig = go.Figure()
fig.add_scatter(x=curve["day"], y=curve["cum_share_of_balance"], mode="lines",
                line={"color": BLUE, "width": 2.5}, name="доля долга",
                hovertemplate="день %{x}: %{y:.2%}<extra></extra>")
mx = [d for _, d in MARKS]
my = [at(curve, "cum_share_of_balance", d) for d in mx]
fig.add_scatter(x=mx, y=my, mode="markers+text", text=[pct(v) for v in my],
                textposition="top left", textfont={"color": INK, "size": 13},
                marker={"color": BLUE, "size": 9, "line": {"color": "white", "width": 2}},
                hoverinfo="skip")
layout(fig, f"За 2 года возвращается {pct(RECOVERY)} долга",
       "накопленная доля от суммы долга на дату включения в РТК", showlegend=False)
fig.update_yaxes(tickformat=".1%", rangemode="tozero")
month_axis(fig)
""")
code("""
colors = [BLUE if m >= start_month else "#b7d3f6" for m in by_month.index]
fig = go.Figure(go.Bar(x=by_month.index, y=by_month / 1e6, marker={"color": colors, "cornerradius": 4},
                       hovertemplate="месяц %{x}: %{y:.1f} млн ₽<extra></extra>"))
fig.add_annotation(x=peak_month, y=by_month.max() / 1e6, text=f"пик: {mln(by_month.max(), 1)}",
                   showarrow=False, yshift=14, font={"color": INK, "size": 13})
layout(fig, f"Деньги начинают приходить с {start_month}-го месяца",
       "поступления по месяцам после включения в РТК, млн ₽", bargap=0.18)
fig.update_xaxes(title_text="месяц после включения в РТК", dtick=1)
fig.update_yaxes(title_text="млн ₽")
fig
""")
code("""
display(Markdown(f'''
> **Вывод.** Первые {start_month - 1} месяца — пауза: приходит {pct(quiet_share, 0)} от суммы за
> 2 года. С {start_month}-го месяца поток выходит на
> {dec(by_month.loc[start_month:12].mean() / 1e6, 0)} млн ₽ в месяц и держится до конца первого года
> ({pct(year1_rest_share, 0)} всех денег). Во втором году поступления постепенно снижаются до
> {dec(by_month.loc[22:].mean() / 1e6, 0)} млн ₽ в месяц, но в сумме дают ещё {pct(year2_share, 0)} —
> оценивать результат включения раньше чем через год преждевременно.
'''))
""")

# ------------------------------------------------------------------ 2. кто платит
md("""
## 2. Сколько договоров платят

Доля договоров, по которым к данному месяцу был хотя бы один платёж.
""")
code("""
fig = go.Figure()
fig.add_scatter(x=conv["day"], y=conv["cum_share_clients"], mode="lines",
                line={"color": BLUE, "width": 2.5},
                hovertemplate="день %{x}: %{y:.1%}<extra></extra>")
my = [at(conv, "cum_share_clients", d) for d in mx]
fig.add_scatter(x=mx, y=my, mode="markers+text", text=[pct(v) for v in my],
                textposition="top left", textfont={"color": INK, "size": 13},
                marker={"color": BLUE, "size": 9, "line": {"color": "white", "width": 2}},
                hoverinfo="skip")
layout(fig, f"Хотя бы один платёж за 2 года — у {pct(PAID, 0)} договоров",
       "накопленная доля договоров с платежом", showlegend=False)
fig.update_yaxes(tickformat=".0%", rangemode="tozero")
month_axis(fig)
""")
code("""
def share_bars(table, labels, title, subtitle, rows):
    palette = [MUTED] + SEQ[1:1 + len(labels) - 1] if labels[0] == "ничего" else SEQ[:len(labels)]
    fig = go.Figure()
    for label, colr in zip(labels, palette, strict=True):
        vals = [table.loc[label, col] for col, _ in rows]
        fig.add_bar(y=[name for _, name in rows], x=vals, orientation="h", name=label,
                    marker={"color": colr, "line": {"color": "white", "width": 2}},
                    text=[pct(v, 0) if v >= 0.045 else "" for v in vals], textposition="inside",
                    insidetextanchor="middle", textfont={"color": "white" if colr not in (MUTED, SEQ[0]) else INK},
                    hovertemplate=label + ": %{x:.1%}<extra></extra>")
    layout(fig, title, subtitle, height=300, barmode="stack",
           legend={"orientation": "h", "y": -0.18, "x": 0, "traceorder": "normal"})
    fig.update_xaxes(tickformat=".0%", range=[0, 1], showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


share_bars(conc, RATE_LABELS,
           f"{pct(top['share_n'])} договоров дают {pct(top['share_money'], 0)} всех денег",
           "договоры по доле возвращённого долга за 2 года",
           [("share_n", "доля договоров"), ("share_money", "доля денег")])
""")
code("""
display(Markdown(f'''
> **Вывод.** {pct(zero["share_n"], 0)} договоров не приносят ничего. Среди платящих половина делает
> первый платёж в первые {dec(first_pay_median, 0)} месяцев. Результат портфеля определяют
> немногие договоры с почти полным погашением: {num(top["n"])} договоров ({pct(top["share_n"])})
> дают {mln(top["money"])} — {pct(top["share_money"], 0)} поступлений. Умение заранее находить
> такие договоры — основной источник выгоды от скоринга.
'''))
""")

# ------------------------------------------------------------------ 3. размер долга и платежа
md("""
## 3. Размер долга и размер платежа
""")
code("""
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.09, subplot_titles=[
    "доля возврата за 2 года", "доля договоров с платежом", "поступления на договор, ₽"])
for i, (col, fmt) in enumerate([("recovery", lambda v: pct(v)), ("paid", lambda v: pct(v, 0)),
                                ("money_per_contract", num)], start=1):
    fig.add_bar(x=by_bal.index.astype(str), y=by_bal[col], row=1, col=i,
                marker={"color": BLUE, "cornerradius": 4}, text=[fmt(v) for v in by_bal[col]],
                textposition="outside", textfont={"color": INK, "size": 12}, cliponaxis=False,
                customdata=by_bal["n"], hovertemplate="%{x}: %{text}<br>договоров: %{customdata:,}<extra></extra>")
layout(fig, f"Чем крупнее долг, тем ниже доля возврата, но больше денег: "
            f"{num(small['money_per_contract'])} → {num(big['money_per_contract'])} ₽ на договор",
       "договоры по размеру долга на дату включения в РТК, ₽", height=430, top=120, showlegend=False,
       bargap=0.3)
fig.update_yaxes(showticklabels=False, showgrid=False)
fig.update_xaxes(tickangle=-30)
fig.update_annotations(font={"size": 13, "color": INK2})
fig
""")
code("""
t = by_bal.assign(share_n=by_bal["n"] / N, share_money=by_bal["money"] / MONEY)
display(pd.DataFrame({
    "договоров": t["n"].map(num), "доля договоров": t["share_n"].map(lambda v: pct(v, 0)),
    "долг, млн ₽": (t["balance"] / 1e6).map(lambda v: dec(v, 0)),
    "поступления, млн ₽": (t["money"] / 1e6).map(lambda v: dec(v, 1)),
    "доля поступлений": t["share_money"].map(lambda v: pct(v, 0)),
}).rename_axis("размер долга, ₽"))
""")
code("""
big_tx = by_tx.loc[["10–50 тыс", "свыше 50 тыс"]].sum()
share_bars(by_tx, TX_LABELS,
           f"Платежи от 10 тыс ₽ — {pct(big_tx['share_n'], 0)} по числу и {pct(big_tx['share_money'], 0)} по деньгам",
           "поступившие платежи по размеру",
           [("share_n", "доля платежей"), ("share_money", "доля денег")])
""")
code("""
small2 = by_bal.iloc[:2]
display(Markdown(f'''
> **Вывод.** В процентах мелкие долги возвращаются даже лучше крупных, а платят по всем размерам
> долга примерно одинаково часто. Но в деньгах разница кратная: долги до 100 тыс ₽ — это
> {pct(small2["n"].sum() / N, 0)} договоров и лишь {pct(small2["money"].sum() / MONEY, 0)}
> поступлений, в среднем {num(small2["money"].sum() / small2["n"].sum())} ₽ на договор за 2 года.
> Деньги приходят в основном крупными платежами: платежи от 10 тыс ₽ — это
> {pct(big_tx["share_n"], 0)} платежей и {pct(big_tx["share_money"], 0)} суммы.
'''))
""")

# ------------------------------------------------------------------ 4. когорты
md("""
## 4. Меняется ли картина со временем

Договоры сгруппированы по кварталу включения в реестр; у всех кварталов прошло полных 2 года.
""")
code("""
fig = go.Figure()
for colr, c in zip(seq_colors(len(cohorts)), cohorts, strict=True):
    g = cohort_curves[cohort_curves["sample"] == c]
    fig.add_scatter(x=g["day"], y=g["cum_share_of_balance"], mode="lines", name=c,
                    line={"color": colr, "width": 2}, hovertemplate=c + ": %{y:.2%}<extra></extra>")
for c in (cohorts[0], cohorts[-1]):
    g = cohort_curves[cohort_curves["sample"] == c]
    fig.add_annotation(x=g["day"].iloc[-1], y=g["cum_share_of_balance"].iloc[-1], text=c, showarrow=False,
                       xanchor="left", xshift=6, font={"color": INK, "size": 12})
layout(fig, f"Договоры {y_last.name} года возвращают на {pct(abs(year_drop), 0)} {LESS}, чем {y_first.name}-го",
       "накопленная доля от долга по кварталам включения в РТК (светлее — раньше)",
       hovermode="x unified", legend={"title": {"text": "квартал"}})
fig.update_yaxes(tickformat=".1%", rangemode="tozero")
month_axis(fig)
fig.update_xaxes(range=[0, H + 60])
fig
""")
code("""
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                    subplot_titles=["доля возврата за 2 года", "доля договоров с платежом"])
for i, (col, d) in enumerate([("recovery", 1), ("paid", 0)], start=1):
    fig.add_bar(x=by_cohort.index, y=by_cohort[col], row=1, col=i,
                marker={"color": seq_colors(len(by_cohort)), "cornerradius": 4},
                text=[pct(v, d) for v in by_cohort[col]], textposition="outside", cliponaxis=False,
                textfont={"color": INK, "size": 12}, customdata=by_cohort["n"],
                hovertemplate="%{x}: %{text}<br>договоров: %{customdata:,}<extra></extra>")
layout(fig, f"С {cohorts[0]} по {cohorts[-1]}: возврат {pct(first_c['recovery'])} → {pct(last_c['recovery'])}, "
            f"платящих {pct(first_c['paid'], 0)} → {pct(last_c['paid'], 0)}",
       "итог за 2 года по кварталам включения в РТК", height=410, top=120, showlegend=False, bargap=0.3)
fig.update_yaxes(showticklabels=False, showgrid=False)
fig.update_annotations(font={"size": 13, "color": INK2})
fig
""")
code("""
bal_med = contracts.groupby("cohort")["rtk_balance"].median()
display(Markdown(f'''
> **Вывод.** Договоры {y_first.name} года возвращали стабильно около {pct(y_first["recovery"])} долга;
> у договоров {y_last.name} года — {pct(y_last["recovery"])}. Доля платящих снижается плавно, от
> квартала к кварталу. С размером долга это не связано: медианный долг не менялся
> ({num(bal_med.iloc[0])} ₽ в {cohorts[0]} и {num(bal_med.iloc[-1])} ₽ в {cohorts[-1]}). Прогнозы и пороги решений, настроенные на договорах 2023 года, будут завышать
> ожидаемые поступления по новым договорам — их нужно сверять по самым свежим кварталам.
'''))
""")

# ------------------------------------------------------------------ 5. контекст
md("""
## 5. Контекст: поток договоров и другие продукты
""")
code("""
x = flow.index.to_timestamp()
in_scope = flow.index <= pd.Period(RETRO_TO, "M")
fig = go.Figure(go.Bar(x=x, y=flow.values, marker={"color": np.where(in_scope, BLUE, MUTED).tolist()},
                       hovertemplate="%{x|%m.%Y}: %{y:,} договоров<extra></extra>"))
fig.add_annotation(x=x[in_scope][len(x[in_scope]) // 2], y=flow[in_scope].max(), yshift=16, showarrow=False,
                   text="период отчёта: прошло 2 года", font={"color": INK, "size": 12})
later = flow[~in_scope]
if len(later):
    fig.add_annotation(x=x[~in_scope][len(later) // 2], y=later.max(), yshift=16, showarrow=False,
                       text="позже: 2 года ещё не прошло", font={"color": INK2, "size": 12})
layout(fig, f"В отчёт вошли договоры до {date_to:%m.%Y}: по ним результат уже известен",
       f"договоров {PRODUCT}, включённых в РТК, по месяцам", height=360, bargap=0.15)
fig.update_yaxes(title_text="договоров")
fig.update_xaxes(tickformat="%m.%Y", dtick="M6")
fig
""")
code("""
fig = go.Figure()
order = prod_last.sort_values("N_total", ascending=False).index.tolist()   # цвет закреплён за продуктом
by_level = prod_last["cum_share_of_balance"].sort_values(ascending=False).index.tolist()
for colr, p in zip(CATEGORICAL, order, strict=False):
    g = prod_curves[prod_curves["sample"] == p]
    last = g["cum_share_of_balance"].iloc[-1]
    fig.add_scatter(x=g["day"], y=g["cum_share_of_balance"], mode="lines", name=f"{p} — {pct(last, 0)}",
                    line={"color": colr, "width": 2.5 if p == PRODUCT else 2}, legendrank=by_level.index(p),
                    hovertemplate=p + ": %{y:.1%}<extra></extra>")
layout(fig, f"{PRODUCT} возвращает в разы меньше других продуктов",
       "накопленная доля от долга по продуктам, договоры того же периода",
       hovermode="x unified", legend={"title": {"text": "продукт — возврат за 2 года"}})
fig.update_yaxes(tickformat=".0%", rangemode="tozero")
month_axis(fig)
fig
""")
code("""
t = prod_last.loc[order]
display(pd.DataFrame({
    "договоров": t["N_total"].map(num), "долг, млн ₽": (t["total_balance"] / 1e6).map(lambda v: dec(v, 0)),
    "поступления за 2 года, млн ₽": (t["cum_tx_sum"] / 1e6).map(lambda v: dec(v, 1)),
    "доля возврата": t["cum_share_of_balance"].map(pct),
    "доля договоров с платежом": t["cum_share_clients"].map(lambda v: pct(v, 0)),
}).rename_axis("продукт"))
note = (f" По продуктам {', '.join(no_tx_products)} в выгрузке нет ни одной транзакции — на график они не вынесены."
        if no_tx_products else "")
others = prod_last.drop(index=PRODUCT)
display(Markdown(f'''
> **Вывод.** {PRODUCT} — {pct(t.loc[PRODUCT, "N_total"] / t["N_total"].sum(), 0)} договоров, но самая
> низкая отдача: {pct(t.loc[PRODUCT, "cum_share_of_balance"])} против
> {dec(others["cum_share_of_balance"].min() * 100, 0)}–{pct(others["cum_share_of_balance"].max(), 0)} у остальных
> продуктов. Остальные продукты малочисленны (от {num(others["N_total"].min())} до
> {num(others["N_total"].max())} договоров), их кривые — ориентир, а не точная оценка.{note}
'''))
""")


# ================================================================== ЧАСТЬ II
md("""
# Часть II. По каким договорам включаться и сколько это даёт

## 6. Экономика включения: пошлина против поступлений

С 09.09.2024 заявление о включении в реестр оплачивается госпошлиной — половина имущественной
пошлины по ст. 333.21 НК РФ:

| сумма требования | пошлина за включение в реестр |
|---|---|
| до 100 тыс ₽ | 5 000 ₽ |
| 300 тыс ₽ | 10 000 ₽ |
| 500 тыс ₽ | 15 000 ₽ |
| 1 млн ₽ | 27 500 ₽ |

Между этими точками — 5 000 ₽ плюс 2,5 % с суммы свыше 100 тыс ₽. Договоры отчёта включались в
реестр, когда пошлины ещё не было; ниже посчитано, **что было бы при нынешнем тарифе**.
""")
code("""
fig = go.Figure()
x = econ_bal.index.astype(str)
fig.add_bar(x=x, y=econ_bal["money"], name="поступления за 2 года", marker={"color": BLUE, "cornerradius": 4},
            text=[num(v) for v in econ_bal["money"]], textposition="outside", textfont={"color": INK, "size": 12},
            customdata=econ_bal["n"], hovertemplate="%{x}: %{y:,.0f} ₽<br>договоров: %{customdata:,}<extra>поступления</extra>")
fig.add_bar(x=x, y=econ_bal["duty"], name="пошлина", marker={"color": ORANGE, "cornerradius": 4},
            text=[num(v) for v in econ_bal["duty"]], textposition="outside", textfont={"color": INK, "size": 12},
            hovertemplate="%{x}: %{y:,.0f} ₽<extra>пошлина</extra>")
layout(fig, f"В среднем на договор: пошлина {num(base['duty'].mean())} ₽, поступления {num(base['money'].mean())} ₽",
       "средние поступления за 2 года и пошлина на договор по размеру долга, ₽", height=430,
       barmode="group", bargap=0.28, bargroupgap=0.08, legend={"orientation": "h", "y": 1.02, "x": 0})
fig.update_yaxes(title_text="₽ на договор", tickformat=",.0f")
fig.update_xaxes(title_text="размер долга, ₽")
fig
""")
code("""
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, column_widths=[0.42, 0.58],
                    subplot_titles=["доля договоров, окупивших пошлину", "итог включения по всем договорам, млн ₽"])
fig.add_bar(x=econ_bal.index.astype(str), y=econ_bal["payoff"], row=1, col=1,
            marker={"color": BLUE, "cornerradius": 4}, text=[pct(v, 0) for v in econ_bal["payoff"]],
            textposition="outside", textfont={"color": INK, "size": 12}, cliponaxis=False,
            hovertemplate="%{x}: %{text}<extra></extra>")
for i, (label, col, colr) in enumerate([("поступления", "money", BLUE), ("пошлина", "duty", ORANGE),
                                         ("результат", "profit", None)]):
    vals = econ[col].astype(float) / 1e6
    fig.add_bar(x=econ.index, y=vals, name=label, row=1, col=2,
                marker={"color": colr or [BLUE if v >= 0 else RED for v in vals], "cornerradius": 4},
                text=[dec(v, 0) for v in vals], textposition="outside", textfont={"color": INK, "size": 12},
                cliponaxis=False, hovertemplate="%{x}: %{y:.1f} млн ₽<extra>" + label + "</extra>")
layout(fig, f"Пошлину окупает {pct(base['payoff'].mean(), 0)} договоров; включение по всем {ALL_WORD}",
       "по размеру долга (слева) и по периодам включения: поступления, пошлина и их разница (справа)",
       height=430, top=120, barmode="group", bargap=0.3, showlegend=False)
fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
fig.update_yaxes(zeroline=True, zerolinecolor=MUTED, row=1, col=2)
fig.update_xaxes(tickangle=-30, row=1, col=1)
fig.update_annotations(font={"size": 13, "color": INK2})
fig
""")
code("""
e23, e24 = econ.iloc[0], econ.iloc[-1]
display(Markdown(f\'\'\'
> **Вывод.** При нынешнем тарифе включение по всем договорам даёт {signed(e23["profit"], 0)} по
> договорам, включённым {P_TRAIN} ({num(e23["n"])} шт.), и {signed(e24["profit"], 0)} — включённым
> {P_TEST} ({num(e24["n"])} шт.). Минимальная пошлина 5 000 ₽ больше типичных поступлений по
> мелким долгам, а {pct(1 - PAID, 0)} договоров {PRODUCT} не платят ничего. При этом по
> {pct(base["payoff"].mean(), 0)} договоров поступления пошлину перекрывают — задача в том, чтобы
> находить их заранее.
\'\'\'))
""")

md("""
### Другие продукты: тот же расчёт

Посчитать, окупается ли включение **по всем** договорам продукта, можно и без модели — для этого
достаточно транзакций и суммы долга.
""")
code("""
t = others_with_tx.sort_values("profit")
fig = go.Figure(go.Bar(y=[f"{i}  " for i in t.index], x=t["profit"] / 1e6, orientation="h",
                       marker={"color": [BLUE if v >= 0 else RED for v in t["profit"]], "cornerradius": 4},
                       text=[signed(v) for v in t["profit"]], textposition="outside", cliponaxis=False,
                       textfont={"color": INK, "size": 12}, customdata=t["n"],
                       hovertemplate="%{y}: %{x:.1f} млн ₽<br>договоров: %{customdata:,}<extra></extra>"))
layout(fig, f"По остальным продуктам включение по всем окупается: {signed(others_with_tx['profit'].sum(), 0)} "
            f"на {num(others_with_tx['n'].sum())} договорах",
       "результат включения по всем договорам продукта (поступления за 2 года − пошлина), млн ₽", height=380)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showgrid=False)
fig
""")
code("""
t = prod_econ
display(pd.DataFrame({
    "договоров": t["n"].map(num), "долг, млн ₽": (t["balance"] / 1e6).map(lambda v: dec(v, 0)),
    "поступления, млн ₽": (t["money"] / 1e6).map(lambda v: dec(v, 1)),
    "пошлина, млн ₽": (t["duty"] / 1e6).map(lambda v: dec(v, 1)),
    "результат «по всем», млн ₽": t["profit"].map(lambda v: signed(v).replace(" млн ₽", "")),
    "окупили пошлину": t["payoff"].map(lambda v: pct(v, 0)),
    "транзакций в выгрузке": t["tx"].map(num),
    "договоров с признаками": t["with_features"].map(num),
}).rename_axis("продукт"))
no_tx_note = (f" По продуктам {', '.join(others_no_tx.index)} ({num(others_no_tx['n'].sum())} договоров) в выгрузке "
              f"нет ни одной транзакции — это нужно проверить в данных, прежде чем считать их убыточными."
              if len(others_no_tx) else "")
odd_note = (f" По продукту {', '.join(others_odd.index)} поступления больше самого долга "
            f"({mln(others_odd['money'].sum(), 1)} при долге {mln(others_odd['balance'].sum(), 1)}) — похоже, в "
            f"транзакции попали не только погашения; в итог он не включён." if len(others_odd) else "")
display(Markdown(f\'\'\'
> **Вывод.** Включение по всем {num(others_with_tx["n"].sum())} договорам (продукты
> {", ".join(others_with_tx.index)}) дало бы {signed(others_with_tx["profit"].sum(), 0)}: поступления
> {mln(others_with_tx["money"].sum())} при пошлине {mln(others_with_tx["duty"].sum())}. Пошлину не окупают
> {pct(others_not_paid, 0)} этих договоров, поэтому выигрыш от отбора по ним не больше всей пошлины,
> {mln(others_with_tx["duty"].sum())}. Малочисленные продукты оцениваются грубо. {NO_FEATURES_NOTE}{no_tx_note}{odd_note}
\'\'\'))
""")

md("""
## 7. Модель: какие договоры окупят пошлину

**Что предсказывает.** Вероятность того, что поступления по договору за 2 года после включения
превысят пошлину по нему. Это прямо условие решения «включаться или нет», поэтому отбор по такой
оценке — отбор по окупаемости. Текущая модель предсказывает другое событие — «вернули больше 5 %
долга» — и пошлину не учитывает.

**На чём обучена и как проверена.** Модель учится на более ранних договорах и проверяется на более
поздних, которых не видела (периоды — в таблице ниже), — так же, как будет работать в жизни:
решение по новым договорам на опыте прошлых.
""")
code("""
parts = [("обучение", tr), ("настройка порога", va), ("проверка", te)]
display(pd.DataFrame({
    "договоров": [num(len(d)) for _, d in parts],
    "период включения в РТК": [f"{d['rtk_send_date'].min():%d.%m.%Y} – {d['rtk_send_date'].max():%d.%m.%Y}" for _, d in parts],
    "окупили пошлину": [pct(d["payoff"].mean()) for _, d in parts],
    "качество ранжирования (AUC)": [dec(auc[k], 3) for k, _ in parts],
}, index=pd.Index([k for k, _ in parts], name="выборка")))
display(Markdown(f\'\'\'
Признаков — {len(FEATURES)}: сумма долга, просрочки по договору, имущество (автомобили,
недвижимость), кредиты в других банках, анкетные данные. Алгоритм — градиентный бустинг (LightGBM).
AUC — доля пар «окупился / не окупился», которые модель упорядочила правильно: 0,5 — случайно,
1,0 — безошибочно.
\'\'\'))
""")
code("""
lift = dec_t["payoff"].iloc[0] / te["payoff"].mean()
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                    subplot_titles=["доля договоров, окупивших пошлину", "результат на договор, ₽ (поступления − пошлина)"])
fig.add_bar(x=dec_t.index.astype(int), y=dec_t["payoff"], row=1, col=1, marker={"color": BLUE, "cornerradius": 4},
            text=[pct(v, 0) for v in dec_t["payoff"]], textposition="outside", cliponaxis=False,
            textfont={"color": INK, "size": 12}, hovertemplate="группа %{x}: %{text}<extra></extra>")
fig.add_hline(y=te["payoff"].mean(), line={"color": INK2, "width": 1, "dash": "dot"}, row=1, col=1,
              annotation_text=f"в среднем {pct(te['payoff'].mean(), 0)}", annotation_position="top right",
              annotation_font={"color": INK2, "size": 12})
fig.add_bar(x=dec_t.index.astype(int), y=dec_t["profit"], row=1, col=2,
            marker={"color": [BLUE if v >= 0 else RED for v in dec_t["profit"]], "cornerradius": 4},
            text=[("+" if v > 0 else "−") + num(abs(v)) for v in dec_t["profit"]], textposition="outside",
            cliponaxis=False, textfont={"color": INK, "size": 12},
            hovertemplate="группа %{x}: %{y:,.0f} ₽ на договор<extra></extra>")
layout(fig, f"В лучшей десятой части пошлину окупают {pct(dec_t['payoff'].iloc[0], 0)} договоров — "
            f"в {dec(lift, 1)} раза чаще среднего",
       "договоры проверочного периода, разбитые на 10 равных групп по оценке модели: 1 — самая высокая оценка",
       height=430, top=120, showlegend=False, bargap=0.25)
fig.update_xaxes(dtick=1, title_text="группа по оценке модели")
fig.update_yaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(zeroline=True, zerolinecolor=MUTED, row=1, col=2)
fig.update_annotations(font={"size": 13, "color": INK2})
fig
""")
code("""
RU = {"rtk_balance": "сумма долга", "age": "возраст", "job_position_cd": "должность",
      "dpd_act": "текущая просрочка, дней", "car_price_sum": "стоимость автомобилей",
      "car_count": "число автомобилей", "realty_price_sum": "стоимость недвижимости",
      "realty_count": "число объектов недвижимости", "education_level_cd": "образование",
      "marital_status_cd": "семейное положение", "children_cnt": "число детей",
      "pensioner_flg": "пенсионер", "parent_financial_account_subtype_cd": "тип исходного продукта"}
top = importance.head(12).iloc[::-1]
top.index = [f"{RU[c]} ({c})  " if c in RU else f"{c}  " for c in top.index]
fig = go.Figure(go.Bar(y=top.index, x=top.values, orientation="h", marker={"color": BLUE, "cornerradius": 4},
                       text=[pct(v, 0) for v in top.values], textposition="outside",
                       textfont={"color": INK, "size": 12}, cliponaxis=False,
                       hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
layout(fig, f"На что опирается модель: {top.index[-1].split(' (')[0].strip()} — {pct(top.values[-1], 0)} вклада",
       "12 самых важных признаков, доля в суммарном вкладе; в скобках — название поля", height=440)
fig.update_layout(margin={"l": 330})
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showgrid=False)
fig
""")
code("""
pos_groups = int((dec_t["profit"] > 0).sum())
display(Markdown(f\'\'\'
> **Вывод.** Модель упорядочивает договоры по окупаемости: в лучшей группе пошлину окупают
> {pct(dec_t["payoff"].iloc[0], 0)} договоров, в худшей — {pct(dec_t["payoff"].iloc[-1], 0)}.
> В среднем в плюс выходит {pos_groups} из 10 групп — включаться стоит по верхней части списка, а не
> по большинству. Качество на проверке (AUC {dec(auc["проверка"], 2)}) ниже, чем на
> обучении ({dec(auc["обучение"], 2)}): модель умеренной силы, и главный резерв роста — новые признаки.
\'\'\'))
""")

md("""
## 8. Как отбираем договоры и сколько это даёт

**Правило.** Для каждого нового договора модель считает оценку. Договоры упорядочиваются по оценке,
включаемся по тем, у кого она выше порога. Порог выбран на договорах периода обучения — в точке,
где суммарный результат (поступления минус пошлина) максимален, — и затем проверен на более поздних
договорах, которых модель не видела.
""")
code("""
fig = go.Figure()
for name, col, colr, mask in [("новая модель: «окупит пошлину»", "score_new", BLUE, mask_new),
                              ("текущая модель: «вернёт > 5 % долга»", "score_old", ORANGE, mask_old)]:
    xs, ys = cum_curve(te[col].to_numpy())
    fig.add_scatter(x=xs, y=ys, mode="lines", name=name, line={"color": colr, "width": 2.5},
                    hovertemplate="%{x:.0%} договоров: %{y:.1f} млн ₽<extra>" + name + "</extra>")
    k = int(mask.sum())
    fig.add_scatter(x=[xs[k - 1]], y=[ys[k - 1]], mode="markers", showlegend=False,
                    marker={"color": colr, "size": 11, "line": {"color": "white", "width": 2}}, hoverinfo="skip")
    up = col == "score_new"
    fig.add_annotation(x=xs[k - 1], y=ys[k - 1], ax=50 if up else 70, ay=-42 if up else 58,
                       text=f"порог: {pct(mask.mean(), 0)} договоров, {signed(ys[k - 1] * 1e6)}",
                       showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=INK2, standoff=8,
                       font={"color": INK, "size": 12}, bgcolor="white")
fig.add_hline(y=0, line={"color": INK2, "width": 1})
ZOOM = 0.6
xs_new, ys_new = cum_curve(te["score_new"].to_numpy())
y_lo = ys_new[int(ZOOM * T) - 1]
fig.add_annotation(x=0, y=y_lo, xanchor="left", yanchor="bottom", xshift=14, yshift=6, showarrow=False,
                   align="left", font={"color": INK2, "size": 12},
                   text=f"дальше линии идут вниз:<br>при включении по всем — {signed(res['all']['profit'], 0)}")
layout(fig, "Накопленный результат растёт до порога и падает после: включаться стоит по верхней части списка",
       "накопленный результат (поступления − пошлина) на проверочных договорах; показаны первые 60 % списка",
       height=470, hovermode="x unified", legend={"orientation": "h", "y": -0.2, "x": 0})
fig.update_xaxes(title_text="доля договоров, по которым включаемся (от самой высокой оценки)", tickformat=".0%",
                 range=[0, ZOOM])
fig.update_yaxes(title_text="млн ₽", range=[y_lo - 1, ys_new.max() + 3])
fig
""")
code("""
names = {"all": "включаться по всем", "old": "отбор текущей моделью", "new": "отбор новой моделью"}
vals = [res[k]["profit"] / 1e6 for k in names]
fig = go.Figure(go.Bar(x=list(names.values()), y=vals, width=0.5,
                       marker={"color": [BLUE if v >= 0 else RED for v in vals], "cornerradius": 4},
                       text=[signed(v * 1e6) for v in vals], textposition="outside", cliponaxis=False,
                       textfont={"color": INK, "size": 14}, hovertemplate="%{x}: %{y:.1f} млн ₽<extra></extra>"))
fig.add_hline(y=0, line={"color": INK2, "width": 1})
layout(fig, f"Отбор новой моделью: {signed(res['new']['profit'])} вместо {signed(res['all']['profit'], 0)}",
       f"результат (поступления − пошлина) на {num(T)} договорах, включённых {P_TEST}, млн ₽", height=400)
fig.update_yaxes(showticklabels=False, showgrid=False)
fig
""")
code("""
display(pd.DataFrame({
    names[k]: {"включаемся по договорам": num(res[k]["n"]), "доля договоров": pct(res[k]["share"], 0),
               "из них окупают пошлину": pct(res[k]["payoff"], 0),
               "уплачено пошлины, млн ₽": dec(res[k]["duty"] / 1e6, 1),
               "поступления за 2 года, млн ₽": dec(res[k]["money"] / 1e6, 1),
               "результат, млн ₽": signed(res[k]["profit"]).replace(" млн ₽", ""),
               "на 1 ₽ пошлины возвращается, ₽": dec(res[k]["money"] / res[k]["duty"], 2)}
    for k in names}))
show_tiles([
    (signed(res["new"]["profit"]), "результат отбора новой моделью",
     f"с вероятностью 90 % — от {signed(ci_new[0])} до {signed(ci_new[1])}"),
    (per_1000_txt, "на 1 000 рассмотренных договоров", "при потоке и качестве договоров как в проверочном периоде"),
    (signed(res["new"]["profit"] - res["old"]["profit"]), "лучше текущей модели",
     f"с вероятностью 90 % — от {signed(ci_diff[0])} до {signed(ci_diff[1])}"),
    (signed(res["oracle"]["profit"], 0), "потолок: безошибочный отбор",
     f"недостижим; показывает запас ({pct(res['oracle']['share'], 0)} договоров)"),
])
""")
md("""
### Результат по продуктам

Одна общая модель и один порог на все продукты; ниже — что получается по каждому продукту на
проверочных договорах. Блок появляется, когда модель построена больше чем по одному продукту.
""")
code("""
if MULTI:
    t = by_product
    fig = go.Figure()
    fig.add_bar(x=t.index, y=t["all_profit"] / 1e6, name="включаться по всем",
                marker={"color": MUTED, "cornerradius": 4}, text=[signed(v) for v in t["all_profit"]],
                textposition="outside", textfont={"color": INK, "size": 12}, cliponaxis=False,
                hovertemplate="%{x}: %{y:.1f} млн ₽<extra>по всем</extra>")
    fig.add_bar(x=t.index, y=t["profit"] / 1e6, name="отбор моделью",
                marker={"color": BLUE, "cornerradius": 4}, text=[signed(v) for v in t["profit"]],
                textposition="outside", textfont={"color": INK, "size": 12}, cliponaxis=False,
                hovertemplate="%{x}: %{y:.1f} млн ₽<extra>отбор моделью</extra>")
    fig.add_hline(y=0, line={"color": INK2, "width": 1})
    layout(fig, "Результат по продуктам: включение по всем и отбор моделью",
           f"поступления − пошлина на договорах, включённых {P_TEST}, млн ₽", height=420, barmode="group",
           bargap=0.3, legend={"orientation": "h", "y": 1.02, "x": 0})
    fig.update_yaxes(title_text="млн ₽")
    fig.show()
    display(pd.DataFrame({
        "договоров на проверке": t["n"].map(num),
        "окупают пошлину": t["payoff"].map(lambda v: pct(v, 0)),
        "включаться по всем, млн ₽": t["all_profit"].map(lambda v: signed(v).replace(" млн ₽", "")),
        "включаемся по модели": t["share"].map(lambda v: pct(v, 0)),
        "из них окупают": t["payoff_take"].map(lambda v: "—" if pd.isna(v) else pct(v, 0)),
        "отбор моделью, млн ₽": t["profit"].map(lambda v: signed(v).replace(" млн ₽", "")),
        "эффект отбора, млн ₽": (t["profit"] - t["all_profit"]).map(lambda v: signed(v).replace(" млн ₽", "")),
    }).rename_axis("продукт"))
    small = t.index[t["n"] < MIN_CONTRACTS].tolist()
    small_note = (f" По продуктам {', '.join(small)} на проверке меньше {MIN_CONTRACTS} договоров — "
                  f"цифры по ним ориентировочные." if small else "")
    better = "общая модель" if solo["joint"] >= solo["solo"] else f"модель только по {PRODUCT}"
    display(Markdown(f\'\'\'
> **Вывод.** По всем продуктам вместе отбор даёт {signed(res["new"]["profit"])} против
> {signed(res["all"]["profit"])} при включении по всем. На {num(solo["n"])} проверочных договорах
> {PRODUCT}: общая модель — {signed(solo["joint"])}, модель, обученная только на {PRODUCT}, —
> {signed(solo["solo"])}, включение по всем — {signed(solo["all"])}; лучше {better}.{small_note}
\'\'\'))
else:
    display(Markdown(f"Модель построена по одному продукту ({PRODUCT}) — разбивки по продуктам нет."))
""")
md("""
### Если затраты окажутся выше

Пошлина — не единственный расход: возможны затраты на подготовку и подачу заявления. Под каждый
сценарий модель обучена заново («окупит ли договор эти затраты»), порог подобран тем же способом.
""")
code("""
display(pd.DataFrame({
    "включаться по всем, млн ₽": sens["all"].map(lambda v: signed(v, 0).replace(" млн ₽", "")),
    "отбор моделью, млн ₽": sens["new"].map(lambda v: signed(v).replace(" млн ₽", "")),
    "включаемся по доле договоров": sens["share"].map(lambda v: pct(v, 0)),
}).rename_axis("затраты на включение"))
worst = sens["new"].iloc[1:]
display(Markdown(f\'\'\'
> **Вывод.** На договорах проверочного периода отбор новой моделью даёт {signed(res["new"]["profit"])}
> ({per_1000_txt} на 1 000 рассмотренных договоров) против {signed(res["all"]["profit"], 0)} при
> включении по всем и {signed(res["old"]["profit"])} у текущей модели. {MARGIN_TXT}
> Среди отобранных пошлину окупают {pct(res["new"]["payoff"], 0)} договоров: результат делают
> немногие договоры с крупными поступлениями, поэтому он колеблется (интервал выше).
> При более высоких затратах результат отбора — от {signed(worst.min())} до {signed(worst.max())}.
> Безошибочный отбор дал бы {signed(res["oracle"]["profit"], 0)} — рост возможен за счёт новых
> признаков.
\'\'\'))
""")

# ------------------------------------------------------------------ методика
code("""
neg = tx.loc[tx["transaction_amt"] < 0, "transaction_amt"].sum()
late = df.loc[df["transaction_amt"].notna() & (df["tx_days"] > H), "transaction_amt"].sum()
display(Markdown(f'''
## Как считали

- **Данные.** Договоры продукта {PRODUCT}, включённые в реестр требований кредиторов с
  {date_from:%d.%m.%Y} по {date_to:%d.%m.%Y}: {num(N)} договоров, долг {dec(BALANCE / 1e9, 2)} млрд ₽.
  Транзакции — по {last_tx:%d.%m.%Y}.
- **Срок наблюдения.** По каждому договору — ровно {H} дней (2 года) от даты включения в РТК; у всех
  договоров этот срок истёк, поэтому кварталы сравнимы между собой.
- **Доля возврата** — сумма поступлений за 2 года, делённая на сумму долга на дату включения
  (`rtk_balance`). Считается по всем договорам, включая те, что ничего не заплатили.
- **Поступления** — все транзакции по договору со знаком: возвраты (отрицательные суммы,
  {mln(-neg, 1)}) вычитаются.
- **Договор с платежом** — была хотя бы одна положительная транзакция за 2 года.
- **Пошлина** — по действующему тарифу (50 % имущественной пошлины, ст. 333.21 НК РФ) от суммы
  долга на дату включения. **Результат** — поступления за 2 года минус пошлина.
- **Другие продукты** — тот же расчёт поступлений и пошлины. {NO_FEATURES_NOTE}
- **Модель** — LightGBM, {len(FEATURES)} признаков на дату включения; обучение — договоры до
  {tr["rtk_send_date"].max():%d.%m.%Y}, проверка — с {te["rtk_send_date"].min():%d.%m.%Y}. Порог
  отбора выбран на отложенной части договоров периода обучения, проверочная выборка в выборе не
  участвовала.
- **Интервалы «с вероятностью 90 %»** — бутстреп по договорам проверочной выборки (1 000 повторов).

**Что важно помнить**

- В выборке только договоры, по которым банк включался в реестр. Сколько пришло бы без включения,
  по этим данным оценить нельзя: все поступления считаются результатом, это оценка сверху.
- После 2 лет деньги продолжают приходить: по этим же договорам позже поступило ещё {mln(late, 1)}
  (+{pct(late / MONEY, 0)} к сумме за 2 года).
- Деньги не приведены к текущей стоимости: пошлина платится сразу, поступления растянуты на 2 года.
- Требования, подтверждённые вступившим в силу судебным актом, пошлиной не облагаются. Если по части
  договоров такие акты есть, включение по ним бесплатно и выгодно почти всегда — отбор нужен только
  для остальных. В расчёте это не учтено (признака в данных нет).
- Новые договоры возвращают меньше прежних (раздел 4), поэтому порог нужно регулярно сверять по
  свежим созревшим договорам.
- Модель в отчёте — рабочий прототип: параметры не подбирались, отбор признаков не проводился.
  Когда решение будет приниматься по модели, данные для переобучения появятся только по отобранным
  договорам — для честной оценки нужна небольшая случайная контрольная группа.
'''))
""")

# ------------------------------------------------------------------ приложение
code("""
checks = prod_econ.copy()
checks["note"] = ""
checks.loc[checks["tx"] == 0, "note"] += "нет транзакций; "
checks.loc[checks["money"] > checks["balance"], "note"] += "поступлений больше долга; "
checks.loc[checks["with_features"] == 0, "note"] += "нет признаков; "
checks.loc[(checks["with_features"] > 0) & (checks["with_features"] < checks["n"]), "note"] += "признаки не по всем договорам; "
checks.loc[checks["n"] < MIN_CONTRACTS, "note"] += f"меньше {MIN_CONTRACTS} договоров; "
checks.loc[checks.index.isin(auto_excluded), "note"] += "исключён из модели автоматически; "
display(Markdown(f\'\'\'
## Приложение: проверка данных

Договоры, включённые в РТК по {RETRO_TO:%d.%m.%Y} (у них прошло полных 2 года); транзакции — по
{last_tx:%d.%m.%Y}. Проверочный период — {P_TEST}: {num(len(te))} договоров, обучение —
{num(len(tr) + len(va))}. Продукты без транзакций и с поступлениями больше долга в модель по
умолчанию не идут; чтобы включить их, перечислите продукты в настройке `MODEL_PRODUCTS`.
\'\'\'))
display(pd.DataFrame({
    "договоров": checks["n"].map(num), "с признаками": checks["with_features"].map(num),
    "транзакций": checks["tx"].map(num), "в модели": ["да" if i in model_products else "нет" for i in checks.index],
    "на что обратить внимание": checks["note"].str.rstrip("; ").replace("", "—"),
}).rename_axis("продукт"))
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["title"] = "Включение в РТК: поступления, модель отбора, эффект"
out = Path(__file__).with_name("rtk_report.ipynb")
nbf.write(nb, out)
print("written", out.name)
