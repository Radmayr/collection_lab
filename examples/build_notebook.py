"""Собирает examples/rtk_pipeline.ipynb (запуск: python examples/build_notebook.py).

Выполнить с сохранением графиков:
    jupyter nbconvert --to notebook --execute --inplace examples/rtk_pipeline.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# RTK-модель на `collection_lab`

Тот же путь, что в ноутбуках `one_factor_analisys` и `prepare_rtk_model`,
но на библиотеке и без утечек:

1. данные и типы признаков;
2. OOT-сплит: **отбор признаков и подбор параметров — только на train/val**, test — для отчёта;
3. EDA: динамика таргета, распределения;
4. отбор признаков одним пайплайном с общим логом;
5. обучение, Optuna (выбор по val), финальная модель;
6. отчёт: метрики, gain chart, стабильность признаков, PSI;
7. сохранение эксперимента (локально, при желании — в ClearML).
""")

code("""
import pandas as pd
import plotly.io as pio

import collection_lab as cl

pio.renderers.default = "notebook_connected+plotly_mimetype"
DATA = "D:/ML_lib/df_agg.csv"   # на ML Core — выгрузка из DAL
TARGET, DATE = "target", "rtk_send_date"
""")

md("## 1. Данные и типы признаков")
code("""
df = pd.read_csv(DATA, low_memory=False)
df[DATE] = pd.to_datetime(df[DATE], format="ISO8601").dt.normalize()

features = [c for c in df.columns
            if not c.endswith(("_dt", "_dttm", "_rk", "_date")) and "target" not in c.lower()
            and c not in ["sample", "contract_number", DATE, "financial_account_subtype_cd",
                          "parent_financial_account_subtype_cd"]]
num_cols, cat_cols = cl.data.split_feature_types(df, features)
df = cl.data.cast_types(df, num_cols, cat_cols, target_col=TARGET)
print(df.shape, "| признаков:", len(features), "| категориальных:", cat_cols)
cl.eda.overview(df, features).sort_values("missing_share", ascending=False).head(10)
""")

md("""
## 2. Сплит без утечек

`test` — всё с 2024-03-01 (out-of-time), `val` — случайные 20% периода разработки.
""")
code("""
split = cl.data.time_split(df, TARGET, DATE, oot_from="2024-03-01", val_size=0.2)
split.summary(DATE)
""")

md("## 3. EDA")
code("""
cl.eda.target_dynamics(split.labeled(), DATE, TARGET, "sample", freq="M", verbose=False).plot()
""")
code("""
cl.eda.plot_distribution(df, "dpd_act")
""")

md("""
## 4. Отбор признаков

Четыре шага из `one_factor_analisys`:
фильтр качества → корреляции (гибрид) → кумулятивная важность → RFE.
Всё на **train**; у каждого шага свой подробный лог в `pipe.results_`.
""")
code("""
pipe = cl.selection.SelectionPipeline([
    cl.selection.QualityFilter(num_missing_threshold=0.999, num_zero_threshold=0.999,
                               num_zero_nan_threshold=0.999, cat_unique_max=300,
                               cat_missing_threshold=0.999),
    cl.selection.CorrelationFilter(threshold=0.8, strategy="hybrid", n_splits=5),
    cl.selection.CumulativeImportance(threshold=0.9),
    cl.selection.RFE(tol=0.01, n_splits=3, min_features=5, params={"n_estimators": 200},
                     verbose=False),
]).fit(split.train, TARGET, features, cat_features=cat_cols)
selected = pipe.selected_
cats = [c for c in cat_cols if c in selected]
print("Отобрано:", selected)
pipe.summary()
""")
code("pipe.plot()")
code("""
# почему выбыл признак — общий лог по всем шагам
pipe.log_.head(15)
""")
code("pipe.results_['4_rfe'].plot()")

md("""
### Порядок признаков: forward-оценка на train (CV), val и test

Отбор идёт только по CV на train. Val и test лишь оцениваются на каждом шаге: если CV растёт,
а test падает, добавленный признак приводит к переобучению.
""")
code("""
fwd = cl.selection.incremental_feature_eval(
    split.train, split.train[TARGET], selected, cat_features=cats,
    eval_sets=split.eval_sets(),        # val и test; можно любые: {"имя": (X, y)}
    n_splits=5, verbose=False)
fwd.plot()
""")
code("""
fwd.table[["n_features", "feature_changed", "auc_mean", "auc_val", "auc_test", "auc_delta_test"]]
""")

md("""
## 5. Обучение и подбор параметров

Optuna выбирает параметры **по val**; AUC на test только записывается.
""")
code("""
base = cl.modeling.train_model(split, selected, cat_features=cats,
                               params={"n_estimators": 300, "learning_rate": 0.01,
                                       "max_depth": 3, "num_leaves": 8})
print("base:", {k: round(v, 4) for k, v in base.scores_.items()})

tune = cl.modeling.tune_hyperparams(split, selected, cat_features=cats, n_trials=30,
                                    fixed_params={"n_estimators": 1000}, verbose=False)
print("лучший val:", round(tune.info["best_val"], 4),
      "| его test:", round(tune.info["best_test"], 4))
tune.plot()
""")
code("""
final = cl.modeling.train_model(split, selected, cat_features=cats,
                                params=tune.info["best_params"], early_stopping_rounds=None)
{k: round(v, 4) for k, v in final.scores_.items()}
""")

md("## 6. Отчёт по модели")
code("""
report = cl.validation.model_report(final, split, date_col=DATE, n_buckets=20)
report.metrics.round(4)
""")
code("report.calibration")
code("report.figures['gain_charts']")
code("report.figures['auc_dynamics']")
code("report.feature_psi")

md("### Стабильность признаков во времени (`plot_stab`)")
code("""
for f in [c for c in selected if c not in cats][:3]:
    for fig in cl.validation.plot_stab(df[f], df[TARGET], df[DATE], 5, feature_nm=f,
                                       period="M", return_plotly_fig=True):
        fig.show()
""")

md("""
## 7. Эксперимент

`clearml=True` дублирует всё в ClearML. **В ClearML уходит только то, что отправлено через
`exp`** (`save_pipeline`, `save_report`, `save_result`, `save_figure`, `save_table`, `log`);
`pipe.save()` и `report.save()` пишут лишь файлы на диск.
""")
code("""
with cl.tracking.Experiment("RTK_model", root="experiments") as exp:   # clearml=True — в ClearML
    exp.log_params(tune.info["best_params"], name="model")
    exp.log_metrics({f"auc_{k}": v for k, v in final.scores_.items()})
    exp.save_features(selected, cat_features=cats, target=TARGET)
    exp.save_model(final)
    exp.save_pipeline(pipe)      # сводка, общий лог, воронка, результаты шагов
    exp.save_report(report)      # метрики, калибровка, gain chart, важности, динамика
    exp.save_result(fwd, "incremental")
    print(exp.plots_summary()[["title", "kind", "sent", "on_server"]])
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("rtk_pipeline.ipynb")
nbf.write(nb, out)
print("записан", out)
