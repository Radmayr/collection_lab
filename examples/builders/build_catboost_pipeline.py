"""Собирает examples/05_catboost_pipeline.ipynb.

Запуск: python examples/builders/build_catboost_pipeline.py

Выполнить с сохранением графиков:
    jupyter nbconvert --to notebook --execute examples/05_catboost_pipeline.ipynb \
        --output 05_catboost_pipeline_executed.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Пайплайн с финальной моделью CatBoost

Путь от таблицы до сохранённой модели, где итоговая модель — **CatBoost**. Во всех функциях
библиотеки алгоритм выбирается одним аргументом `model="catboost"`: подготовка признаков,
метрики, отчёт и выгрузка остаются теми же.

Ноутбук работает на встроенных демо-данных. **Чтобы запустить на своих — поменяйте только
ячейку «Настройки».**

1. типы признаков и сплит без утечек;
2. отбор признаков;
3. LightGBM или CatBoost — сравнение на одних и тех же фолдах;
4. подбор параметров CatBoost (Optuna, выбор по val);
5. финальная модель;
6. отчёт: метрики, калибровка, важности, стабильность;
7. что у CatBoost устроено иначе;
8. выгрузка для инференса без библиотеки;
9. сохранение эксперимента.

Нужен `catboost`: `pip install "collection_lab[catboost]"` (или `[all]`).
""")

code("""
import pandas as pd
import plotly.io as pio

import collection_lab as cl

pio.renderers.default = "notebook_connected+plotly_mimetype"
pd.set_option("display.max_columns", 40)
""")

md("""
## Настройки

Единственная ячейка, которую нужно менять под свои данные. Требования к таблице: одна строка —
одно наблюдение; в ней лежат таргет, дата и признаки.
""")
code("""
# --- данные ---------------------------------------------------------------------------
df = cl.data.make_demo_data()        # свои данные: df = pd.read_csv("путь/к/таблице.csv")

TARGET = "target"                    # бинарный таргет: 0 / 1
DATE = "report_date"                 # дата наблюдения: разбиение по времени, динамика
OOT_FROM = "2024-03-01"              # с этой даты — отложенная выборка test
SEGMENT = "segment"                  # колонка сегмента для отчёта; None — без сегментов

# --- признаки -------------------------------------------------------------------------
NOT_FEATURES = ["client_id"]         # идентификаторы и прочие служебные колонки
FEATURES = None                      # None — все колонки, кроме служебных; или явный список

# --- расчёт ---------------------------------------------------------------------------
N_TRIALS = 15                        # испытаний Optuna; в работе — 50–100
MAX_ITERATIONS = 500                 # потолок числа деревьев; ранняя остановка выберет меньше
PROJECT = "catboost_demo"            # имя проекта: папка эксперимента и проект в ClearML
""")

md("""
## 1. Типы признаков и сплит

`test` — всё с даты `OOT_FROM` (out-of-time), `val` — случайные 20% периода разработки.
Отбор признаков и подбор параметров идут только на train / val.
""")
code("""
service = {TARGET, DATE, SEGMENT, *NOT_FEATURES}
features = list(FEATURES) if FEATURES else [c for c in df.columns if c not in service]

df[DATE] = pd.to_datetime(df[DATE], format="mixed").dt.normalize()   # дата без времени
num_cols, cat_cols = cl.data.split_feature_types(df, features)
df = cl.data.cast_types(df, num_cols, cat_cols, target_col=TARGET)
print("признаков:", len(features), "| категориальных:", cat_cols)

split = cl.data.time_split(df, TARGET, DATE, oot_from=OOT_FROM, val_size=0.2)
split.summary(DATE)
""")

md("""
## 2. Отбор признаков

Шаги отбора по умолчанию считают на LightGBM — он быстрее, а список признаков получается близким.
Если нужно, чтобы отбор шёл той же моделью, что и финальная, добавьте шагу `model="catboost"`
(есть у `CumulativeImportance` и `RFE`).
""")
code("""
pipe = cl.selection.SelectionPipeline([
    cl.selection.QualityFilter(num_missing_threshold=0.95, cat_unique_max=100),
    cl.selection.CorrelationFilter(threshold=0.8, strategy="hybrid"),
    cl.selection.CumulativeImportance(threshold=0.95, model="catboost",
                                      params={"iterations": 300}),
]).fit(split.train, TARGET, features, cat_features=cat_cols)

selected = pipe.selected_
cats = [c for c in cat_cols if c in selected]
print("отобрано:", len(selected), "| категориальных:", cats)
pipe.summary()
""")
code("pipe.plot()")
code("""
# почему выбыл признак — общий лог по всем шагам
pipe.log_
""")

md("""
## 3. LightGBM или CatBoost

`compare_models` обучает обе модели на одних и тех же фолдах train — разница в метрике
объясняется алгоритмом, а не разбиением.
""")
code("""
summary, oof = cl.modeling.compare_models(
    split.train, split.train[TARGET], models=("lgbm", "catboost"),
    features=selected, cat_features=cats, n_splits=3,
    params={"lgbm": {"n_estimators": MAX_ITERATIONS, "learning_rate": 0.05},
            "catboost": {"iterations": MAX_ITERATIONS, "learning_rate": 0.05}})
summary.round(4)
""")

md("""
## 4. Подбор параметров CatBoost

`model="catboost"` включает пространство поиска CatBoost (`depth`, `learning_rate`,
`l2_leaf_reg`, `random_strength`, `bagging_temperature`); своё задаётся через `search_space`.
Лучший набор выбирается **по val**, метрика на test только записывается.
""")
code("""
tune = cl.modeling.tune_hyperparams(
    split, selected, cat_features=cats, model="catboost", n_trials=N_TRIALS,
    fixed_params={"iterations": MAX_ITERATIONS}, verbose=False)
print("лучший val:", round(tune.info["best_val"], 4),
      "| его test:", round(tune.info["best_test"], 4))
tune.info["best_params"]
""")
code("tune.plot()")

md("""
## 5. Финальная модель

В `best_params` уже записано число деревьев, найденное ранней остановкой, поэтому финальная
модель обучается без неё (`early_stopping_rounds=None`). `refit_on_train_val=True` — обучить на
train + val: val больше не нужен для остановки, а test остаётся независимым.
""")
code("""
model = cl.modeling.train_model(split, selected, cat_features=cats, model="catboost",
                                params=tune.info["best_params"], early_stopping_rounds=None,
                                refit_on_train_val=True)
print(model)
{k: round(v, 4) for k, v in model.scores_.items()}
""")

md("""
## 6. Отчёт по модели

Отчёт от алгоритма не зависит: те же таблицы и графики, что для LightGBM.
""")
code("""
report = cl.validation.model_report(model, split, date_col=DATE, segment=SEGMENT, n_buckets=10)
report.metrics.round(4)
""")
code("report.calibration.round(4)")
code("report.figures['gain_charts']")
code("report.figures['feature_importance']")
code("report.figures['auc_dynamics']")
code("""
if report.segments is not None:
    display(report.segments.round(4))
""")
code("""
# PSI каждого признака по месяцам test относительно train
report.plot_feature_psi()
""")

md("""
## 7. Что у CatBoost устроено иначе

| | LightGBM | CatBoost |
|---|---|---|
| Число деревьев | `n_estimators` | `iterations` |
| Глубина и регуляризация | `num_leaves`, `max_depth`, `reg_lambda` | `depth`, `l2_leaf_reg` |
| Категориальные признаки | dtype `category` | строки; пропуск и новая категория → `"NaN"` |
| Важности `feature_importance()` | прирост качества (gain) | изменение прогноза |
| Файл модели при выгрузке | `model.txt` | `model.cbm` |

Категории в обоих случаях запоминаются по train одним и тем же `FeaturePreparer`, так что
готовить данные руками не нужно: в `fit` и `predict` передаётся исходная таблица.
""")
code("""
print("деревьев в модели:", model.n_iterations_)
print("категориальные:", model.cat_features_)
model.feature_importance().round(3)
""")
code("""
# нативный объект CatBoost — если нужны его собственные методы (SHAP, деревья и т.п.)
type(model.estimator_)
""")

md("""
## 8. Выгрузка для инференса без библиотеки

В папке — `model.cbm`, `preprocessing.json`, `inference.py` и `requirements.txt`. На боевой
стороне нужны только pandas, numpy и catboost: `import inference; inference.predict(df)`.
`X_check` сверяет предсказания скрипта с библиотекой.
""")
code("""
info = model.export(f"export/{PROJECT}", X_check=split.test)
print("максимальное расхождение со скриптом:", info["max_abs_diff"])
sorted(p.name for p in info["directory"].iterdir())
""")

md("""
## 9. Эксперимент

`clearml=True` дублирует всё в ClearML.
""")
code("""
with cl.tracking.Experiment(PROJECT, root="experiments") as exp:   # clearml=True — в ClearML
    exp.log_params(tune.info["best_params"], name="model")
    exp.log_metrics({f"auc_{k}": v for k, v in model.scores_.items()})
    exp.save_features(selected, cat_features=cats, target=TARGET)
    exp.save_model(model)
    exp.save_pipeline(pipe)
    exp.save_report(report)
    exp.save_result(tune, "tuning")
    print("версия:", exp.path)
""")
md("""
Сохранённую модель можно сразу использовать как базу при тестировании новых признаков:
`cl.feature_testing.test_features(split, exp.load_model(), candidates)` — см.
`04_feature_testing`.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parents[1] / "05_catboost_pipeline.ipynb"
nbf.write(nb, out)
print("записан", out)
