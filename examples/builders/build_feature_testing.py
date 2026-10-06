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

Ноутбук работает на встроенных демо-данных. **Чтобы запустить на своих — поменяйте только
ячейку «Настройки»**: всё, что ниже, от конкретных колонок не зависит.

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
""")

md("""
## Настройки

Единственная ячейка, которую нужно менять под свои данные. Требования к таблице: одна строка —
одно наблюдение; в ней лежат таргет, дата, все признаки (и текущей модели, и новые) и, если
нужно, колонка сегмента.
""")
code("""
# --- данные ---------------------------------------------------------------------------
df = cl.data.make_demo_data()        # свои данные: df = pd.read_csv("путь/к/таблице.csv")

TARGET = "target"                    # бинарный таргет: 0 / 1
DATE = "report_date"                 # дата наблюдения: разбиение по времени, PSI по месяцам
OOT_FROM = "2024-03-01"              # с этой даты — отложенная выборка test
SEGMENT = "segment"                  # колонка сегмента; None — без сегментов

# --- признаки -------------------------------------------------------------------------
DOMAINS = {                          # признаки по источникам данных: {"домен": [колонки]}
    "анкета": ["age", "income", "education", "region", "children_cnt"],
    "поведение": ["dpd_max_12m", "dpd_cnt_12m", "utilization", "months_on_book"],
    "бюро": ["bureau_score", "bureau_inquiries_6m", "bureau_active_loans",
             "bureau_dpd_max_12m", "bureau_legal_flg"],
    "транзакции": ["tx_cnt_3m", "tx_sum_3m", "tx_salary_flg", "tx_cash_share"],
}
BASE_DOMAINS = ["анкета", "поведение"]     # на них построена текущая модель
NEW_DOMAINS = ["бюро", "транзакции"]       # новые данные, которые нужно оценить

# несколько новых признаков для точечной проверки (разделы 2–5, 7, 9)
CANDIDATES = ["bureau_score", "bureau_dpd_max_12m", "bureau_legal_flg", "tx_cnt_3m",
              "tx_cash_share"]

# --- расчёт ---------------------------------------------------------------------------
PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 16}
MAX_EXACT = 4                        # сколько кандидатов оценивать точно (раздел 6); в работе — 20
""")

md("""
## 1. Сплит и «разработанная модель»

Из настроек собираются списки признаков, сплит и базовая модель — она играет роль уже
разработанной модели, к которой примеряются новые признаки.
""")
code("""
base_features = [c for name in BASE_DOMAINS for c in DOMAINS[name]]
new_features = [c for name in NEW_DOMAINS for c in DOMAINS[name]]

df[DATE] = pd.to_datetime(df[DATE], format="mixed").dt.normalize()   # дата без времени
num_cols, cat_cols = cl.data.split_feature_types(df, base_features + new_features)
df = cl.data.cast_types(df, num_cols, cat_cols, target_col=TARGET)

split = cl.data.time_split(df, TARGET, DATE, oot_from=OOT_FROM, val_size=0.2)
print("признаков в базе:", len(base_features), "| новых:", len(new_features))
split.summary(DATE)
""")
code("""
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
test = cl.feature_testing.test_features(split, base_features, CANDIDATES, segment=SEGMENT,
                                        date_col=DATE, params=PARAMS)
test.summary.round(4)
""")
md("""
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

На что смотреть в демо-данных: признак, заполненный у 1% строк (`coverage`), признаки,
появившиеся только с середины периода (`filled_from`, высокий `psi_max`), и кандидат, похожий
на признак базы (`max_corr_base`).
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

`segment=SEGMENT` в вызове выше — колонка с меткой сегмента. Прирост считается ещё и внутри
каждого сегмента: общий плюс может складываться из выигрыша в одном сегменте и проигрыша в
другом. Сегменты меньше 30 строк не показываются.
""")
code("""
if SEGMENT:
    test.plot_segments().show()                       # основная метрика на test
    test.plot_segments(metric="ks", part="cv").show()   # другая метрика, по CV на train
""")
code("""
# числами: прирост основной метрики от всех кандидатов вместе в каждом сегменте
if SEGMENT:
    display(test.segments.query("name == 'ALL' and metric == @test.info['metric']").round(4))
""")

md("""
## 4. База — уже разработанная модель

Вместо списка признаков передаётся сама модель: признаки и гиперпараметры берутся из неё.
Подходит всё, что есть на руках:

```python
test_features(split, model, candidates)                      # обученная модель библиотеки
test_features(split, "experiments/my_model/v_3/models/model.pkl", candidates)   # из эксперимента
test_features(split, "export/my_model", candidates)          # папка model.export(...)
test_features(split, lgbm_or_catboost_model, candidates)     # нативный LightGBM / CatBoost
test_features(split, "model.txt", candidates)                # файл LightGBM (.cbm — CatBoost)
```
""")
code("""
by_model = cl.feature_testing.test_features(split, base_model, CANDIDATES[:2], segment=SEGMENT)
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

Здесь колонка `score` считается базовой моделью; в работе это готовый скор из вашей таблицы.
""")
code("""
scored = df.assign(score=base_model.predict(df))
scored_split = cl.data.time_split(scored, TARGET, DATE, oot_from=OOT_FROM, val_size=0.2)

on_top = cl.feature_testing.test_features(scored_split, "score", CANDIDATES, segment=SEGMENT,
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
many = cl.feature_testing.test_features(split, base_features, new_features,
                                        max_exact=MAX_EXACT, date_col=DATE, params=PARAMS)
print(many)
many.summary[["name", "screen_delta", "uni_auc", "delta_cv", "cv_ci_low", "cv_ci_high",
              "delta_test", "verdict"]].round(4)
""")
code("many.plot()")

md("""
## 7. Добавить или заменить

Кандидат, похожий на признак базы, сверху добавляет меньше, чем мог бы: часть информации уже в
модели. Но он может быть лучше старого. `test.replacement()` для каждого кандидата с
корреляцией с признаком базы не ниже `threshold` обучает модель «база без похожего признака +
кандидат» и сравнивает её с базой.

Если таблица пустая — среди кандидатов нет похожих на признаки базы (см. `max_corr_base` в
сводке), можно понизить `threshold`.
""")
code("""
replacement = test.replacement(threshold=0.6)
replacement.table.round(4)
""")
code("""
# синий — добавить кандидата к базе, красный — поставить его вместо похожего признака
if len(replacement.table):
    replacement.plot().show()
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
    split, base_features, {name: DOMAINS[name] for name in NEW_DOMAINS},
    segment=SEGMENT, date_col=DATE, params=PARAMS)
domains.summary.drop(columns=["kind", "uni_auc", "folds_better", "delta_val"]).round(4)
""")
code("domains.plot()")
code("""
if SEGMENT:
    domains.plot_segments().show()
""")
code("domains.plot_metrics()")
md("""
Что внутри домена: `features` — вклад каждого признака домена в модель «база + домен».
Если почти весь вклад дают два-три признака, остальные можно не подключать.
""")
code("""
domains.features.round(4)
""")
code("""
for name in NEW_DOMAINS:
    domains.plot_domain(name).show()
""")
md("""
`base` у доменов задаётся так же, как у признаков: список, модель или колонка скора.
""")
code("""
domains_on_top = cl.feature_testing.test_domains(
    scored_split, "score", {name: DOMAINS[name] for name in NEW_DOMAINS},
    segment=SEGMENT, params=PARAMS)
domains_on_top.summary[["name", "coverage", "delta_cv", "delta_test", "test_ci_low",
                        "test_ci_high", "verdict"]].round(4)
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
    split, base_features, CANDIDATES[:2],
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
with cl.tracking.Experiment("feature_testing_demo", root="experiments") as exp:   # clearml=True
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
