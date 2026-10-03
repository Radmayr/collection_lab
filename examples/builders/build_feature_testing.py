"""Собирает examples/04_feature_testing.ipynb.

Запуск: python examples/builders/build_feature_testing.py

Выполнить с сохранением графиков:
    jupyter nbconvert --to notebook --execute examples/04_feature_testing.ipynb \
        --output 04_feature_testing_executed.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Тестирование новых признаков и доменов: `cl.feature_testing`

Вопрос один: **что даст модели добавление этих признаков?** Ответ — прирост метрик с
доверительным интервалом, в целом и по сегментам.

| Что нужно | Как вызвать | Раздел |
|---|---|---|
| Несколько признаков: по одному и все вместе | `test_features(split, base, candidates)` | 2 |
| Прирост по сегментам | `segment="колонка"` | 3 |
| Сравнение с уже разработанной моделью | `base=model` или путь к ней | 4 |
| Есть только скор готовой модели | `base="колонка_скора"` | 5 |
| Кандидатов десятки и сотни | то же самое, быстрый отсев включается сам | 6 |
| Новый признак похож на тот, что уже в модели | `test.replacement()` | 7 |
| Домен данных — группа признаков из одного источника | `test_domains(split, base, domains)` | 8 |
| Свой набор метрик, более узкие интервалы | `metrics=...`, `n_repeats=...` | 9 |
| Сохранить результат | `test.save(dir)`, `exp.log(test)` | 10 |

**Без утечки.** Вердикт ставится по кросс-валидации на train. Val нужен для ранней остановки,
test только показывается.
""")

code("""
import pandas as pd
import plotly.io as pio

import collection_lab as cl

pio.renderers.default = "notebook_connected+plotly_mimetype"
pd.set_option("display.max_columns", 40)

DATA = "D:/ML_lib/df_agg.csv"   # на ML Core — выгрузка из DAL
TARGET, DATE = "target", "rtk_send_date"
SEGMENT = "age_group"
""")

md("""
## 1. Данные, сплит и «разработанная модель»

Признаки разложены по доменам — по источнику данных. Считаем, что модель уже разработана на
доменах **просрочка** и **анкета**, а **другие счета** и **имущество** — новые данные, которые
нужно оценить.
""")
code("""
df = pd.read_csv(DATA, low_memory=False)
df[DATE] = pd.to_datetime(df[DATE], format="ISO8601").dt.normalize()

DOMAINS = {
    "просрочка": [c for c in df.columns if c.startswith("dpd")] + ["last_col_end_days"],
    "анкета": ["age", "education_level_cd", "marital_status_cd", "children_cnt",
               "self_employed_job_org_flg", "job_position_cd", "pensioner_flg", "risk_level_cd"],
    "другие счета": [c for c in df.columns if "oth_" in c or c.endswith("_util_days")],
    "имущество": [c for c in df.columns if c.startswith(("car_", "realty_", "suspected_car"))]
                 + ["pledge_count", "apartment_cnt", "living_house_cnt", "land_cnt",
                    "other_realty_cnt"],
}
features = [c for cols in DOMAINS.values() for c in cols]
num_cols, cat_cols = cl.data.split_feature_types(df, features)
df = cl.data.cast_types(df, num_cols, cat_cols, target_col=TARGET)

# сегмент — отдельная колонка таблицы; здесь это возрастная группа
df[SEGMENT] = pd.cut(df["age"], [0, 35, 50, 200], labels=["до 35", "36–50", "старше 50"]
                     ).astype(object).fillna("возраст не указан")
{name: len(cols) for name, cols in DOMAINS.items()}
""")
code("""
split = cl.data.time_split(df, TARGET, DATE, oot_from="2024-03-01", val_size=0.2)
split.summary(DATE)
""")
code("""
# параметры поменьше, чтобы пример считался быстро; в работе — параметры вашей модели
PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 16}

base_features = DOMAINS["просрочка"] + DOMAINS["анкета"]
base_model = cl.modeling.train_model(split, base_features, params=PARAMS)
{k: round(v, 4) for k, v in base_model.scores_.items()}
""")

md("""
## 2. Несколько новых признаков

`base` — список признаков текущей модели, `candidates` — новые. Считается каждый кандидат по
отдельности и строка `ALL` — все вместе: база и «база + кандидаты» обучаются с одинаковыми
параметрами на одних и тех же фолдах.
""")
code("""
candidates = ["car_count", "car_price_sum", "oth_acc_cnt", "max_util_days", "bad_oth_bal_sum"]

test = cl.feature_testing.test_features(split, base_features, candidates, segment=SEGMENT,
                                        date_col=DATE, params=PARAMS)
test.summary.round(4)
""")
md("""
`bad_oth_bal_sum` взят намеренно: он заполнен меньше чем у 1% договоров — это видно по
`coverage`, а прирост только по заполненным строкам показывает `delta_test_covered`.

Как читать сводку. Все дельты — «насколько стало лучше»: `> 0` — кандидат улучшил метрику.

| Колонка | Что это |
|---|---|
| `coverage`, `filled_from`, `filled_to` | У какой доли строк признак заполнен и за какой период |
| `uni_auc` | AUC признака в одиночку (без модели) |
| `max_corr_base`, `corr_with` | Самая сильная корреляция с признаком базы и с каким именно |
| `psi_max` | Максимальный PSI по месяцам test относительно train |
| `delta_cv`, `cv_ci_low`, `cv_ci_high` | Прирост основной метрики по CV на train, интервал 95% |
| `folds_better` | Доля фолдов, где с кандидатом стало лучше |
| `delta_val`, `delta_test`, `test_ci_*` | Прирост на val и test; интервал на test — бутстрап |
| `delta_test_covered` | Прирост на test только по строкам, где кандидат заполнен |
| `worst_segment`, `worst_segment_delta` | Сегмент с наименьшим приростом на test |
| `verdict` | `better` / `same` / `worse` — по интервалу CV; подсказка, решение за вами |

`better` — интервал по CV целиком выше нуля, `same` — захватывает ноль, `worse` — целиком ниже.
""")
code("""
# точка — прирост по CV на train (цвет — вердикт), ромб — прирост на test
test.plot()
""")
code("""
# все метрики сразу: прирост в % от значения базы на test
test.plot_metrics()
""")
code("""
# то же числами: каждая метрика на CV / val / test
test.metrics.query("name == 'ALL'").round(4)
""")

md("""
## 3. Сегменты

`segment="колонка"` в вызове выше — метка сегмента для каждой строки. Прирост считается ещё и
внутри каждого сегмента: общий плюс может складываться из выигрыша в одном сегменте и проигрыша
в другом. Сегменты меньше 30 строк не показываются.
""")
code("""
test.summary[["name", "delta_cv", "delta_test", "worst_segment", "worst_segment_delta",
              "verdict"]].round(4)
""")
code("test.plot_segments()                      # основная метрика на test")
code("test.plot_segments(metric='ks', part='cv')  # другая метрика, по CV на train")
code("""
test.segments.query("name == 'ALL' and metric == 'auc'").round(4)
""")

md("""
## 4. База — уже разработанная модель

Вместо списка признаков передаётся сама модель: признаки и гиперпараметры берутся из неё.
Подходит всё, что есть на руках:

```python
test_features(split, model, candidates)                           # обученная модель библиотеки
test_features(split, "experiments/RTK_model/v_3/models/model.pkl", candidates)   # из эксперимента
test_features(split, "export/rtk_v3", candidates)                 # папка model.export(...)
test_features(split, lgbm_or_catboost_model, candidates)          # нативный LightGBM / CatBoost
test_features(split, "model.txt", candidates)                     # файл LightGBM (.cbm — CatBoost)
```
""")
code("""
by_model = cl.feature_testing.test_features(split, base_model, candidates[:2], segment=SEGMENT)
print(by_model.info["base"], "|", by_model.info["how"])
by_model.summary[["name", "delta_cv", "cv_ci_low", "cv_ci_high", "delta_test", "verdict"]].round(4)
""")

md("""
## 5. Есть только скор готовой модели

Когда модель нельзя или не хочется переобучать, в `base` передаётся имя колонки со скором
(вероятность). Тогда способ сравнения — `how="on_top"`: поверх скора обучается LightGBM
**только на кандидатах**. Вопрос здесь другой: не «какой станет модель», а «есть ли в кандидатах
информация, которой нет в модели».

> Если скор на train посчитан моделью, обученной на этих же строках, прирост по CV занижен
> (модель «помнит» train). В этом режиме ориентир — `delta_test`.
""")
code("""
scored = df.assign(score=base_model.predict(df))
scored_split = cl.data.time_split(scored, TARGET, DATE, oot_from="2024-03-01", val_size=0.2)

on_top = cl.feature_testing.test_features(scored_split, "score", candidates, segment=SEGMENT,
                                          params=PARAMS)
print(on_top.info["base"], "|", on_top.info["how"])
on_top.summary[["name", "max_corr_base", "delta_cv", "delta_test", "test_ci_low",
                "test_ci_high", "verdict"]].round(4)
""")
code("on_top.plot()")
md("""
Тот же способ можно включить и для модели: `test_features(split, model, candidates,
how="on_top")` — скор посчитает сама модель.
""")

md("""
## 6. Много кандидатов

Вызов тот же. Если кандидатов больше `max_exact` (по умолчанию 20), сначала идёт быстрый отсев:
на каждом кандидате маленькая модель поверх скора базы, прирост на val — колонка
`screen_delta`. Точно оцениваются `max_exact` лучших и строка `ALL`; остальные получают вердикт
`screened out`.
""")
code("""
many = cl.feature_testing.test_features(split, base_features, DOMAINS["другие счета"],
                                        max_exact=5, date_col=DATE, params=PARAMS)
print(many)
many.summary[["name", "screen_delta", "uni_auc", "delta_cv", "cv_ci_low", "cv_ci_high",
              "delta_test", "verdict"]].round(4)
""")
code("many.plot()")

md("""
## 7. Добавить или заменить

Кандидат, похожий на признак базы, сверху почти ничего не добавляет — информация уже в модели.
Но он может быть лучше старого. `test.replacement()` для каждого кандидата с корреляцией с
признаком базы не ниже `threshold` обучает модель «база без похожего признака + кандидат».

Пример: в базе уже есть число автомобилей с известной ценой, кандидаты — общее число
автомобилей и их суммарная цена.
""")
code("""
base_with_cars = base_features + ["car_count_with_price"]

t_cars = cl.feature_testing.test_features(split, base_with_cars, ["car_count", "car_price_sum"],
                                          params=PARAMS)
t_cars.summary[["name", "max_corr_base", "corr_with", "delta_cv", "cv_ci_low", "cv_ci_high",
                "verdict"]].round(4)
""")
code("""
replacement = t_cars.replacement(threshold=0.6)
replacement.table.round(4)
""")
code("""
# синий — добавить кандидата к базе, красный — поставить его вместо похожего признака
replacement.plot()
""")

md("""
## 8. Домены данных

Домен — группа признаков из одного источника; подключается или не подключается целиком,
поэтому и оценивается целиком: база против «база + все признаки домена». Домены оцениваются
независимо друг от друга.

Дополнительно к колонкам `test_features`:

| Колонка | Что это |
|---|---|
| `coverage` | Доля строк, где заполнен хотя бы один признак домена |
| `delta_test_covered` | Прирост только на этих строках |
| `alone_auc_test`, `base_auc_test` | Качество модели на одном домене (без базы) и качество базы |
| `gain_share` | Доля домена в важности модели «база + домен» |
""")
code("""
domains = cl.feature_testing.test_domains(
    split, DOMAINS["анкета"],                       # база — только анкета
    {name: DOMAINS[name] for name in ("просрочка", "другие счета", "имущество")},
    segment=SEGMENT, date_col=DATE, params=PARAMS)
domains.summary[["name", "n_features", "coverage", "delta_cv", "cv_ci_low", "cv_ci_high",
                 "delta_test", "delta_test_covered", "alone_auc_test", "base_auc_test",
                 "gain_share", "verdict"]].round(4)
""")
code("domains.plot()")
code("domains.plot_segments()")
code("domains.plot_metrics()")
md("""
Что внутри домена: `features` — вклад каждого признака домена в модель «база + домен».
Если почти весь вклад дают два-три признака, остальные можно не тащить.
""")
code("""
domains.features.query("domain == 'другие счета'").head(10).round(4)
""")
code("domains.plot_domain('другие счета')")
md("""
`base` у доменов задаётся так же, как у признаков: список, модель или колонка скора.
""")
code("""
domains_on_top = cl.feature_testing.test_domains(
    scored_split, "score", {name: DOMAINS[name] for name in ("другие счета", "имущество")},
    segment=SEGMENT, params=PARAMS)
domains_on_top.summary[["name", "delta_cv", "delta_test", "test_ci_low", "test_ci_high",
                        "worst_segment", "worst_segment_delta", "verdict"]].round(4)
""")

md("""
## 9. Свои метрики и точность интервалов

- `metrics` — любой набор; **первая — основная**: по ней ранжирование, интервалы и вердикт.
  Доступны `auc`, `gini`, `ks`, `logloss`, `pr_auc`, `brier`, `lift@10%` (процент любой) и своя
  функция `f(y_true, y_pred)`.
- `n_repeats` — повторы кросс-валидации: интервал уже, считается дольше.
- `n_boot` — повторы бутстрапа для интервала на test; `alpha` — уровень значимости.
- `max_rows` — на больших данных кросс-валидация идёт на подвыборке train (по умолчанию 200 000
  строк); итог на val и test всегда на всех данных.
""")
code("""
precise = cl.feature_testing.test_features(
    split, base_features, ["oth_acc_cnt", "car_count"],
    metrics=("gini", "ks", "lift@5%"), n_repeats=2, n_boot=500, params=PARAMS)
precise.summary[["name", "base_gini_cv", "delta_cv", "cv_ci_low", "cv_ci_high", "folds_better",
                 "delta_test", "test_ci_low", "test_ci_high", "verdict"]].round(4)
""")

md("""
## 10. Сохранение

`test.save(dir)` — таблицы в csv. Внутри эксперимента `exp.log(test)` кладёт таблицы и графики
в `logs/` версии и, при `clearml=True`, в ClearML.
""")
code("""
with cl.tracking.Experiment("RTK_feature_testing", root="experiments") as exp:   # clearml=True
    exp.log(test, "new_features")
    exp.log(domains, "new_domains")
    exp.log(replacement)
    print(sorted(p.name for p in exp.logs_path.iterdir()))
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parents[1] / "04_feature_testing.ipynb"
nbf.write(nb, out)
print("записан", out)
