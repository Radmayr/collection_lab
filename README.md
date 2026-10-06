# collection_lab

Библиотека для разработки, отбора признаков и валидации ML-моделей (бинарная классификация,
LightGBM / CatBoost; регрессия поддерживается ядром). Графики — plotly, трекинг — локально
и в ClearML.

**Содержание:** [Установка](#установка) · [Быстрый старт](#быстрый-старт) · [Примеры](#примеры) ·
[Структура репозитория](#структура-репозитория) ·
[Версионирование экспериментов](#версионирование-экспериментов) ·
[Инференс без библиотеки](#подготовка-признаков-и-инференс-без-библиотеки) ·
[Тестирование новых признаков](#тестирование-новых-признаков) ·
[Как устроено](#как-устроено) · [Справочник функций](#справочник-функций)

## Установка

С GitHub (на ML Core или любой машине с `pip`):

```bash
pip install "collection_lab[all] @ git+https://github.com/Radmayr/collection_lab.git"
```

Опциональные зависимости: `catboost`, `optuna`, `clearml`, `all`, `dev`.

Для разработки (правки видны сразу, без переустановки):

```bash
git clone https://github.com/Radmayr/collection_lab.git && cd collection_lab
python -m venv .venv && .venv/Scripts/activate      # Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Быстрый старт

```python
import collection_lab as cl

# типы и сплит без утечек: test — out-of-time, отбор только на train
num_cols, cat_cols = cl.data.split_feature_types(df, features)
df = cl.data.cast_types(df, num_cols, cat_cols, target_col="target")
split = cl.data.time_split(df, "target", "rtk_send_date", oot_from="2024-03-01")
split.summary("rtk_send_date")

# отбор признаков — цепочка шагов с общим логом
pipe = cl.selection.SelectionPipeline([
    cl.selection.QualityFilter(num_missing_threshold=0.999, cat_unique_max=300),
    cl.selection.CorrelationFilter(threshold=0.8, strategy="hybrid"),
    cl.selection.CumulativeImportance(threshold=0.9),
    cl.selection.RFE(tol=0.01, n_splits=3, segments="product"),
]).fit(split.train, "target", features, cat_features=cat_cols)
pipe.summary(); pipe.log_; pipe.plot()

# обучение, подбор параметров (выбор по val), отчёт
tune = cl.modeling.tune_hyperparams(split, pipe.selected_, n_trials=50)
model = cl.modeling.train_model(split, pipe.selected_, params=tune.info["best_params"])
report = cl.validation.model_report(model, split, date_col="rtk_send_date")
report.show()
report.plot_feature_psi()        # PSI каждого признака по месяцам test относительно train

# эксперимент: папки v_N, модель, признаки, метрики (+ ClearML)
with cl.tracking.Experiment("RTK_model", clearml=True) as exp:
    exp.save_model(model); exp.save_features(pipe.selected_, cat_features=cat_cols)
    exp.save_pipeline(pipe)      # сводка, лог, воронка, результаты шагов
    exp.save_report(report)      # метрики, калибровка, gain chart, важности, динамика
```

> **Что уходит в ClearML.** Всё, что отправлено через `exp`: `save_pipeline`, `save_report`,
> `save_result`, `save_figure`, `save_table`, `log` (универсальный, тип определяется сам).
> Внутри активного `Experiment(clearml=True)` то же делают `.save(dir)` у `Result`,
> `SelectionPipeline`, `ModelReport` (`to_clearml=False` — только диск). Что произошло с каждым
> графиком — `exp.plots_summary()` и итоговая строка при `close()`.
>
> Чтобы графики, которые вы строите и просто смотрите в ноутбуке (`pipe.plot()`, `fwd.plot()`,
> `plot_stab(...)`, `gain_chart(...)`), тоже попадали в ClearML, создайте
> `Experiment(..., clearml=True, capture_plots=True)`. Создавайте **один** `Experiment` на запуск и
> не вызывайте `Experiment(...)` повторно внутри `with ... as exp:`.

## Примеры

| № | Ноутбук | На какой вопрос отвечает |
|---|---|---|
| 1 | [01_eda_overview](examples/01_eda_overview.ipynb) | Что лежит в таблице и как пользоваться каждой функцией `eda` |
| 2 | [02_vintage_overview](examples/02_vintage_overview.ipynb) | Сколько и когда поступает после контрольной даты — по продуктам, типам транзакций, когортам |
| 3 | [03_rtk_pipeline](examples/03_rtk_pipeline.ipynb) | Как построить модель от данных до отчёта и сохранённого эксперимента |
| 4 | [04_feature_testing](examples/04_feature_testing.ipynb) | Что даст модели добавление новых признаков и доменов данных |
| 5 | [05_catboost_pipeline](examples/05_catboost_pipeline.ipynb) | Как построить модель, если финальная модель — CatBoost |

Ноутбуки 4 и 5 работают на встроенных демо-данных (`cl.data.make_demo_data()`) и переносятся на
свою таблицу правкой одной ячейки «Настройки». Что считает каждый и как запустить — в
[examples/README.md](examples/README.md).

Пример отдельного проекта на библиотеке — отчёт по включению в РТК — ведётся в своём репозитории
[rtk_recovery](https://github.com/Radmayr/rtk_recovery-): библиотека содержит только функции, расчёты под конкретную задачу — в проектах.

## Структура репозитория

```
collection_lab/
├── src/collection_lab/   библиотека (модули — в таблице «Как устроено» ниже)
├── examples/             ноутбуки-примеры 01–05, их описание и скрипты-сборщики (builders/)
├── tests/                тесты: pytest; медленные помечены slow
├── README.md             этот файл
├── CHANGELOG.md          что менялось по версиям
└── pyproject.toml        зависимости и настройки сборки, ruff, pytest
```

## Версионирование экспериментов

`tracking.Experiment("RTK_model", root=..., clearml=True)` — один запуск = одна версия.

| | Локально | В ClearML |
|---|---|---|
| Что такое версия | папка `<root>/<project>/v_<N>` (`models/`, `logs/`, `meta.json`) | задача с именем `v_<N>` в проекте `<project>` |
| Как выдаётся номер | `version="auto"`: максимальный `N` в папке проекта + 1; номера не переиспользуются | берётся тот же `N`; идентичность задачи — её id, а не имя |
| Явный номер | `version=3`; если `v_3` уже есть — `FileExistsError`, для перезаписи `overwrite=True` | новая задача с тем же именем `v_3` |

- **Перезапуск в том же ноутбуке.** ClearML допускает одну активную задачу на процесс.
  `Experiment` сам закрывает предыдущую и создаёт новую, поэтому ячейку можно запускать
  повторно. Лучше оформлять запуск как `with Experiment(...) as exp:` — задача закроется сама.
- **Ошибка ClearML** при создании задачи не оставляет пустой локальной папки версии и не
  «сжигает» номер.
- **Список версий:** `Experiment.list_versions("RTK_model", root=...)` — таблица версий с
  метриками из `meta.json`.

## Подготовка признаков и инференс без библиотеки

**Как готовятся признаки.** Обе модели (LightGBM и CatBoost) получают данные через один и тот же
`FeaturePreparer` (внутри `model.fit/predict`): числовые → `float64` (даты → наносекунды, `bool` → 0/1,
строки → число, нечисловое → NaN); категориальные → строки, где `nan`/`None`/`null`/`''` и настоящие
пропуски считаются пропуском, а числовые категории приводятся к виду `"1"` (не `"1.0"`).
Категории запоминаются **по train**. LightGBM получает `category` с этими категориями, CatBoost — строки,
причём пропуск и неизвестная на обучении категория превращаются в строку `"NaN"`.
Старый `make_lgbm_ready` остаётся для совместимости, но модели библиотеки им не пользуются.

**Инференс, когда библиотеки нет.** Выгрузите модель — получите папку, которая работает без
`collection_lab`:

```python
info = model.export("export/rtk_v3", X_check=split.test)   # или exp.export_model(model, "final")
# model.txt | model.cbm   — модель в нативном формате
# preprocessing.json      — порядок признаков, категориальные, их категории, обозначения пропусков
# inference.py            — prepare(df) и predict(df); не импортирует collection_lab
# requirements.txt        — версии numpy / pandas / lightgbm|catboost
```

На боевой стороне (нужны только pandas, numpy и lightgbm/catboost):

```python
import inference
scores = inference.predict(df)          # df — таблица с колонками признаков
```

- Код подготовки признаков в `inference.py` **копируется из исходников библиотеки автоматически**, а
  не набирается вручную; `X_check` сверяет предсказания скрипта с библиотекой (при расхождении —
  `RuntimeError`). Тесты проверяют паритет на данных с пропусками, неизвестными категориями,
  числовыми категориями, датами и `bool`.
- Свои преобразования (feature engineering и т.п.), выполняемые до стандартной подготовки,
  прописываются в `custom_preprocess(df)` внутри `inference.py`.

## Тестирование новых признаков

Раздел `feature_testing` отвечает на вопрос «что даст модели добавление этих признаков» —
приростом метрик с доверительным интервалом, в целом и по сегментам. Полный пример —
[04_feature_testing](examples/04_feature_testing.ipynb).

```python
# несколько признаков: каждый отдельно и все вместе (строка ALL)
test = cl.feature_testing.test_features(split, base, candidates, segment="product",
                                        date_col="report_date")
test.summary; test.plot(); test.plot_segments(); test.plot_metrics()
test.replacement().table        # добавить или заменить похожий признак базы

# домен данных — группа признаков из одного источника, оценивается целиком
domains = cl.feature_testing.test_domains(split, base, {"транзакции": [...], "бюро": [...]})
domains.summary; domains.features; domains.plot_domain("транзакции")
```

Что передать в `base` и какой способ сравнения получится:

| Что есть | `base` | Способ по умолчанию |
|---|---|---|
| Список признаков | `base=selected` | `retrain` — база и «база + кандидаты» с одинаковыми параметрами |
| Разработанная модель | `base=model`, путь к `.pkl` эксперимента, папка экспорта, нативный LightGBM / CatBoost или его файл | `retrain` — признаки и гиперпараметры берутся из модели |
| Только скор модели | `base="score_col"` — колонка с вероятностью | `on_top` — LightGBM поверх скора только на кандидатах |

- **`how="retrain"`** отвечает на вопрос «какой станет модель», **`how="on_top"`** — «есть ли в
  кандидатах то, чего нет в модели». Для модели можно задать `how="on_top"` явно. Если скор на
  train посчитан моделью, обученной на этих же строках, прирост по CV в режиме `on_top`
  занижен — ориентир даёт test.
- **Дельты** — «насколько стало лучше»: `> 0` — кандидат улучшил метрику (для `logloss` и
  `brier` знак уже перевёрнут).
- **Вердикт** `better` / `same` / `worse` ставится по интервалу прироста на кросс-валидации
  train (целиком выше нуля, захватывает ноль, целиком ниже). Test только показывается — с
  бутстрап-интервалом. Вердикт — подсказка, решение за вами.
- **Метрики** — `metrics=("auc", "gini", "ks", "logloss", "pr_auc", "brier", "lift@10%")` по
  умолчанию; первая — основная. Список можно сократить или дополнить своей функцией.
- **Много кандидатов** — если их больше `max_exact` (20), сначала идёт быстрый отсев, точно
  оцениваются лучшие; на больших данных кросс-валидация идёт на подвыборке (`max_rows`).
- **Неполное заполнение** — в сводке видно покрытие, период заполнения и прирост только по
  строкам, где кандидат заполнен.

## Как устроено

- **Единый движок.** Все методы обучают модели через `core.cross_validate` и адаптеры
  `core.make_model("lgbm" | "catboost")`: одинаковые фолды, категории, early stopping,
  метрика и сегменты — везде. Метрика задаётся строкой (`"auc"`, `"gini"`, `"ks"`,
  `"logloss"`) или своей функцией; направление учитывается автоматически.
- **Результат — объект `Result`**: `.table` / `.log` (лог решений), `.selected`, `.info`,
  `.plot()` (plotly), `.save(dir)`.
- **Параметры по умолчанию** — только в `collection_lab/config.py`.
- **Без утечек**: `DataSplit` проверяет непересечение частей; `tune_hyperparams` выбирает
  лучшие параметры по val, test только записывается; `FeaturePreparer` фиксирует категории
  по train.

| Модуль | Что внутри |
|---|---|
| `data` | `split_feature_types`, `cast_types`, `time_split`, `random_split`, `DataSplit`, `make_demo_data` |
| `core` | `make_model`, `LGBMModel`, `CatBoostModel`, `FeaturePreparer`, `cross_validate`, `make_folds`, `Result` |
| `feature_testing` | `test_features`, `test_domains`, `FeatureTest` |
| `eda` | `overview`, `target_summary`, `plot_distribution`, `plot_target_rate_by_bins`, `target_dynamics`, `vintage`, `vintage_by_type`, `vintage_by_segment`, `maturation_transactions`, `plot_vintage`, `eda_transactions`, `add_days_since` |
| `metrics` | `roc_auc`, `gini`, `ks`, `metrics_by_segment`, `psi`, `psi_table`, `feature_psi`, `psi_by_period`, `gain_chart`, `gain_chart_metrics`, `hosmer_lemeshow`, `information_value`, `iv_table` |
| `selection` | `quality_filter`, `correlation_filter`, `univariate_scores`, `cumulative_importance_selection`, `drop_column_importance`, `permutation_importance`, `rfe`, `backward_elimination`, `forward_addition`, `incremental_feature_eval`, `SelectionPipeline` |
| `modeling` | `train_model`, `compare_models`, `tune_hyperparams`, `sample_size_curve` |
| `validation` | `plot_stab`, `metric_dynamics`, `feature_metric_dynamics`, `learning_curve`, `model_report` |
| `tracking` | `Experiment`, `clearml.*` |
| `plotting` | `style`, `combine` (сетка графиков) |

Что делает каждая функция — в [справочнике](#справочник-функций) ниже.

## Справочник функций

Все функции вызываются как `cl.<модуль>.<функция>` после `import collection_lab as cl`. Полный список
параметров и значения по умолчанию — в докстринге: `help(cl.selection.rfe)` или `cl.selection.rfe?`
в Jupyter. Общие для большинства функций параметры:

| Параметр | Смысл |
|---|---|
| `features` | Список признаков; по умолчанию — все колонки таблицы |
| `cat_features` | Категориальные признаки; по умолчанию определяются автоматически |
| `model`, `params` | `"lgbm"` / `"catboost"` или готовый адаптер; `params` дополняют дефолты из `config.py` |
| `metric` | `"auc"`, `"gini"`, `"ks"`, `"logloss"`, `"pr_auc"`, `"brier"`, `"lift@10%"` или своя функция `f(y_true, y_pred)` |
| `segments` | Колонка или массив меток — метрика считается ещё и по каждому сегменту |
| `folds` / `n_splits` | Готовые фолды из `core.make_folds` или их число (`<= 1` — holdout) |

### `data` — типы признаков и сплиты

| Функция | Что делает | Возвращает |
|---|---|---|
| `split_feature_types(df, features)` | Делит признаки на числовые и категориальные | `(num_cols, cat_cols)` |
| `detect_categorical(df, exclude)` | Категориальные колонки таблицы | `list[str]` |
| `is_categorical(s)` | Категориальный ли признак: нельзя целиком привести к числу (dtype `category` — всегда да) | `bool` |
| `cast_types(df, num_cols, cat_cols, target_col)` | Числовые → `float64`, категориальные → `object` с NaN вместо строковых пропусков, таргет → `int` | копия `df` |
| `normalize_missing(s)` | Строки `'nan'`, `'None'`, `'null'`, `''` → NaN | `Series` |
| `category_strings(s)` | Категории как строки в каноничном виде (`1` и `1.0` → `"1"`) | `Series` |
| `to_numeric_frame(X, cat_cols)` | Некатегориальные колонки → float (даты → наносекунды, `bool` → 0/1) | `DataFrame` |
| `time_split(df, target, date_col, oot_from, oot_to, val_size, val_mode)` | Out-of-time сплит: test — строки с датой `>= oot_from`, остальное делится на train / val (`val_mode="random"` или `"time"` — последние по дате) | `DataSplit` |
| `random_split(df, target, val_size, test_size, stratify)` | Случайный стратифицированный сплит | `DataSplit` |
| `make_demo_data(n, random_state)` | Синтетическая таблица для примеров: таргет, дата, сегмент и признаки четырёх доменов (`DEMO_DOMAINS`) | `DataFrame` |

`DataSplit` — части `train`, `val`, `test` и имя таргета; при создании проверяется, что части не пересекаются.

| Метод | Что делает |
|---|---|
| `split.summary(date_col)` | Размер, число и доля таргета по частям (+ диапазон дат) |
| `split.xy(part, features)` | `(X, y)` для части `"train"` / `"val"` / `"test"` |
| `split.eval_sets(parts)` | `{"val": (df, y), "test": (df, y)}` — вход `eval_sets=` у `incremental_feature_eval` |
| `split.labeled(col="sample")` | Все части одной таблицей с колонкой-меткой части |
| `split.items()` | Итерация по непустым частям: `("train", df), ...` |

### `core` — модели, кросс-валидация, результат

| Функция / класс | Что делает | Возвращает |
|---|---|---|
| `make_model(model, params, task, early_stopping_rounds, verbose)` | Создаёт адаптер по имени (`"lgbm"`, `"catboost"`) или клонирует готовый | `LGBMModel` / `CatBoostModel` |
| `make_folds(y, n_splits, stratify, test_size)` | Фиксированные фолды, чтобы сравнения шли на одном разбиении | `[(train_idx, valid_idx), ...]` |
| `cross_validate(X, y, features, model, params, folds, metric, segments, ...)` | Обучает модель на фолдах, считает метрику — общую и по сегментам | `CVResult` |
| `FeaturePreparer(cat_features)` | Подготовка признаков для бустингов: `fit` запоминает категории по train, `transform` применяет их к любой выборке | — |
| `make_lgbm_ready(X, cat_cols)` | Совместимость со старым кодом; модели библиотеки им не пользуются | `(X_ready, cat_cols)` |
| `export_model(model, directory, X_check)` | Выгрузка модели для инференса без библиотеки (то же, что `model.export`) | `dict` с путями файлов |
| `iterations_param(model, n)` | Имя параметра числа деревьев: `{"n_estimators": n}` или `{"iterations": n}` | `dict` |

Адаптер модели (`LGBMModel`, `CatBoostModel`):

| Метод | Что делает |
|---|---|
| `model.fit(X, y, eval_set, cat_features, sample_weight)` | Обучение; early stopping по `eval_set`, если он передан |
| `model.predict(X)` | Вероятность класса 1 (binary) или прогноз (regression) |
| `model.feature_importance(kind="gain")` | Важности признаков (`"gain"` / `"split"`) по убыванию |
| `model.clone(params)` | Необученная копия с теми же настройками |
| `model.export(directory, X_check)` | Папка с нативной моделью, `preprocessing.json`, `inference.py`, `requirements.txt` |
| `model.n_iterations_` | Число деревьев, реально используемое при прогнозе |

`CVResult`: `fold_scores`, `mean`, `std`, `segment_scores`, `segment_means`, `oof` (out-of-fold прогноз),
`best_iterations`, `mean_best_iteration`, `summary()`.

`Result` — то, что возвращают функции отбора и анализа:

| Атрибут / метод | Что это |
|---|---|
| `result.table` (он же `result.log`) | Основная таблица: лог решений по признакам или история шагов |
| `result.selected` | Отобранные признаки |
| `result.dropped` | Признаки, помеченные в логе как удалённые |
| `result.info` | Параметры запуска и сводные метрики |
| `result.plot()` | Plotly-график результата |
| `result.save(directory, prefix, to_clearml)` | Таблица (csv), список признаков и `info` (json) в папку |

### `eda` — разведочный анализ

| Функция | Что делает | Возвращает |
|---|---|---|
| `overview(df, columns)` | Сводка по колонкам: тип, пропуски, уникальные, доля нулей, самое частое значение, min / median / mean / max | `DataFrame` |
| `target_summary(df, target, by)` | Размер выборки, число и доля таргета — всего или в разрезе `by` | `DataFrame` |
| `plot_distribution(data, col, clip_low, clip_high, force_log, bins)` | Гистограмма с обрезкой выбросов по квантилям и авто-лог-шкалой | `Figure` |
| `plot_target_rate_by_bins(df, col, target, bins)` | Доля таргета по квантильным бинам признака и размер бинов | `Figure` |
| `target_dynamics(df, date_col, target_col, segment_col, freq, alpha)` | Доля таргета по периодам и сегментам с доверительным интервалом Уилсона | `Result` |
| `wilson_ci(k, n, alpha)` | Доверительный интервал Уилсона для доли `k / n` | `(low, high)` |
| `add_days_since(df, event_col, base_col, name)` | Колонка «дней от `base_col` до `event_col`» | `DataFrame` |
| `maturation_transactions(df, sample, horizon_days, step, population, ...)` | Расчёт винтажа: накопительные суммы, число транзакций и доля клиентов по дням от ретро-даты | `DataFrame` |
| `plot_vintage(res, y)` | Винтажные кривые по таблице `maturation_transactions` | `Figure` |
| `vintage(df, sample, y, ...)` | Винтаж одной выборки: расчёт и график одним вызовом | `(DataFrame, Figure)` |
| `vintage_by_type(df, tx_type_col, types, top_k, min_tx, abs_amount, ...)` | Винтажи по типам транзакций на одном графике; база общая, кривые в сумме дают общий винтаж | `(DataFrame, Figure)` |
| `vintage_by_segment(df, segment_col, segments, top_k, min_contracts, ...)` | Винтажи по продуктам / когортам; у каждого сегмента своя база | `(DataFrame, Figure)` |
| `eda_transactions(df, ...)` | Быстрый EDA транзакционной таблицы: обзор, пропуски, распределение сумм, знаки, топ клиентов | `dict` таблиц и графиков |

`y` у винтажей: `cum_share_of_balance` (по умолчанию), `cum_share_clients`, `cum_tx_sum`, `cum_tx_cnt`,
`cum_clients_with_tx`.

### `metrics` — качество, стабильность, калибровка, WoE / IV

Качество:

| Функция | Что делает | Возвращает |
|---|---|---|
| `roc_auc(y_true, y_pred)` | ROC AUC; NaN, если в `y_true` один класс | `float` |
| `gini(y_true, y_pred)` | `2 · AUC − 1` | `float` |
| `ks(y_true, y_pred)` | Статистика Колмогорова–Смирнова между скорами классов | `float` |
| `logloss(y_true, y_pred)` | Log loss по вероятностям класса 1 | `float` |
| `metrics_by_segment(y_true, y_pred, segments, metrics, min_size, add_total)` | Метрики по сегментам (+ строка `ALL`) | `DataFrame` |
| `pr_auc(y_true, y_pred)` | Площадь под PR-кривой | `float` |
| `brier(y_true, y_pred)` | Средний квадрат ошибки вероятности | `float` |
| `lift(y_true, y_pred, share)` | Доля таргета в верхней доле `share` по скору относительно всей выборки | `float` |
| `get_metric(metric)` | Метрика по имени (включая `"lift@10%"`) или функции вместе с направлением оптимизации | `Metric` |

Стабильность (PSI). Бины числового признака — квантили базовой выборки, пропуски — отдельный бин `NaN`;
`< 0.1` — stable, `0.1–0.25` — moderate, `> 0.25` — significant.

| Функция | Что делает | Возвращает |
|---|---|---|
| `psi(expected, actual, bins, categorical)` | PSI между двумя выборками | `float` |
| `psi_table(expected, actual, bins, categorical)` | Разбивка PSI по бинам: счётчики, доли, вклад бина | `DataFrame` |
| `feature_psi(expected, actual, features, bins, cat_features)` | PSI каждого признака между двумя таблицами, по убыванию | `DataFrame` `feature, psi, status` |
| `psi_by_period(df, column, date_col, freq, mode, reference, bins, categorical)` | PSI признака по периодам: против базы `reference` (или первого периода) либо против предыдущего периода (`mode="adjacent"`) | `DataFrame` `period, n, psi, status` |
| `psi_label(value)` | Текстовая интерпретация значения PSI | `str` |
| `psi_from_counts(expected_counts, actual_counts)` | PSI по готовым счётчикам бинов | `float` |
| `quantile_edges(x, bins)` | Границы квантильных бинов | `ndarray` |

Калибровка:

| Функция | Что делает | Возвращает |
|---|---|---|
| `prob_to_logit(p)` | Логит вероятности | `ndarray` |
| `gain_chart(values, target, groups, n_buckets, calib, calib_full, ...)` | Gain chart по бакетам **логита**: фактический badrate с 99% ДИ против прогноза, опционально линии калибровки | `dict` метрик, фигура или `(fig, dict)` |
| `gain_chart_table(logit, target, n_buckets)` | Расчётная часть gain chart для одной группы | `(DataFrame, dict)` |
| `gain_chart_metrics(metrics, segments)` | Сводная таблица метрик нескольких `gain_chart` | `DataFrame` |
| `hosmer_lemeshow(y_true, y_prob, buckets)` | Статистика Хосмера–Лемешоу и p-value | `(hl, p_value)` |
| `calibration_offset(logit, target)` | Сдвиг `b`, при котором средний прогноз равен среднему таргету | `float` |
| `full_calibration(logit, target)` | Коэффициенты `(k, b)` регрессии `target ~ k·logit + b` | `(k, b)` |

WoE / IV:

| Функция | Что делает | Возвращает |
|---|---|---|
| `information_value(values, target, n_buckets, method, bins_method)` | IV признака (NaN — отдельный бакет) | `float` |
| `iv_table(df, target, features, n_buckets)` | IV числовых признаков по убыванию с оценкой силы | `DataFrame` `feature, iv, strength` |
| `quantile_buckets(values, n_buckets)` | Квантильные бакеты, не разрывающие одинаковые значения | `ndarray` номеров бакетов |
| `woe_iv_table(values, target, buckets)` | WoE и вклад в IV по бакетам | `DataFrame` |

### `selection` — отбор признаков

Все функции возвращают `Result`: `selected` — оставшиеся признаки, `table` — лог решений.

| Функция | Что делает |
|---|---|
| `quality_filter(df, features, num_missing_threshold, num_zero_threshold, cat_unique_max, ...)` | Убирает признаки с долей пропусков / нулей выше порога, константы, категориальные со слишком малым или большим числом значений |
| `correlation_filter(df, y, features, threshold, strategy, ...)` | В каждой группе коррелирующих числовых признаков оставляет один с лучшим AUC. `strategy`: `"direct"` — AUC признака как скора, `"model"` — CV AUC однофакторной модели, `"hybrid"` — модель только там, где лидеры близки |
| `correlation_groups(X, threshold, method)` | Группы признаков, связанных цепочками `\|corr\| > threshold`; возвращает `(groups, corr)` |
| `univariate_scores(X, y, features, n_splits, ...)` | AUC маленькой модели LightGBM на каждом признаке отдельно |
| `univariate_filter(X, y, features, min_auc)` | Оставляет признаки с однофакторным AUC `>= min_auc` |
| `direct_auc(x, y, folds)` | AUC признака как скора без модели: `max(AUC, 1 − AUC)`; возвращает `float` |
| `cumulative_importance_selection(df, target, features, threshold, importance, ...)` | Оставляет признаки, дающие `threshold` суммарной важности, и сравнивает метрику полной и урезанной модели |
| `permutation_importance(model, X, y, features, metric, n_repeats)` | Падение метрики при перемешивании признака; возвращает `DataFrame` |
| `drop_column_importance(train, val, features, target, test, ...)` | Падение метрики при удалении признака и переобучении |
| `rfe(X, y, features, tol, segments, segment_tol, step, min_features, ...)` | Рекурсивное исключение: признак удаляется, если CV-метрика падает не больше чем на `tol` (и в каждом сегменте — не больше `segment_tol`) |
| `backward_elimination(X_train, y_train, X_valid, y_valid, features, ...)` | Удаляет по одному наименее важный признак до `min_features`, на каждом шаге — метрика на train / valid и в сегментах |
| `forward_addition(X_train, y_train, X_valid, y_valid, base_features, candidate_features, mode)` | Эффект добавления кандидатов к базовому набору: по одному, всех разом или группами |
| `incremental_feature_eval(X, y, features, eval_sets, direction, mode, ...)` | Пошагово добавляет или убирает признаки и считает CV-метрику и метрику на внешних наборах (val, test); выбор шага — только по CV |
| `importance_plot(importance, top_k)` | Горизонтальный bar-chart важностей; возвращает `Figure` |

`SelectionPipeline([шаги])` — цепочка, где каждый шаг получает признаки, оставшиеся после предыдущего.
Шаги: `QualityFilter`, `CorrelationFilter`, `UnivariateFilter`, `CumulativeImportance`, `RFE` (параметры —
как у одноимённых функций) и `Custom(func)` для своего шага `func(df, target, features, cat_features)`.

| Атрибут / метод | Что это |
|---|---|
| `pipe.fit(df, target, features, cat_features)` | Запуск всех шагов |
| `pipe.selected_` | Итоговый список признаков |
| `pipe.results_` | `{имя_шага: Result}` с полными логами |
| `pipe.log_` | Общий лог выбывших: `step, feature, action, reason` |
| `pipe.summary()` | Сколько признаков вошло и вышло на каждом шаге |
| `pipe.plot()` | Воронка отбора |
| `pipe.save(directory)` | Результаты шагов и общий лог в папку |

### `modeling` — обучение и подбор параметров

| Функция | Что делает | Возвращает |
|---|---|---|
| `train_model(split, features, model, params, auto_scale_pos_weight, refit_on_train_val, ...)` | Обучение на train с early stopping по val; `refit_on_train_val=True` — финальная модель на train + val с подобранным числом деревьев | адаптер; `model.scores_` — метрика по частям |
| `compare_models(X, y, models, params, n_splits, ...)` | CV-сравнение моделей на одних и тех же фолдах | `(summary, oof)` |
| `tune_hyperparams(split, features, model, search_space, fixed_params, n_trials, ...)` | Optuna-поиск: обучение на train, выбор по val, test только записывается | `Result`; `info["best_params"]`, `info["study"]` |
| `lgbm_search_space(trial)`, `catboost_search_space(trial)` | Пространства поиска по умолчанию | `dict` параметров |
| `sample_size_curve(split, features, sizes, n_repeats, ...)` | Как метрика зависит от объёма обучающей выборки | `Result` |

### `feature_testing` — новые признаки и домены

| Функция | Что делает | Возвращает |
|---|---|---|
| `test_features(split, base, candidates, how, segment, date_col, metrics, params, n_splits, n_repeats, max_exact, max_rows, n_boot, alpha)` | Прирост от каждого кандидата и от всех вместе относительно базы | `FeatureTest` |
| `test_domains(split, base, domains, ...)` | Прирост от каждого домена целиком; покрытие, качество домена в одиночку, вклад признаков внутри домена | `FeatureTest` |

`FeatureTest`:

| Атрибут / метод | Что это |
|---|---|
| `test.summary` | Строка на кандидата (и `ALL`) или на домен: покрытие, однофакторный AUC, корреляция с базой, PSI, прирост по CV с интервалом, прирост на val и test, худший сегмент, вердикт |
| `test.metrics` | Все метрики: `name, part (cv / val / test), metric, base, new, delta, ci_low, ci_high` |
| `test.segments` | То же по сегментам: `name, part, segment, n, metric, base, new, delta` |
| `test.features` | Для доменов: `domain, feature, importance, share, coverage, uni_auc` |
| `test.plot()` | Прирост основной метрики с интервалами: CV на train и test |
| `test.plot_segments(metric, part)` | Прирост по сегментам: кандидаты × сегменты |
| `test.plot_metrics(part)` | Прирост всех метрик в % от базы |
| `test.plot_domain(name)` | Вклад признаков внутри домена |
| `test.replacement(threshold=0.7)` | Для кандидатов, похожих на признак базы: прирост при добавлении и при замене этого признака (`Result` с `table` и `plot()`) |
| `test.save(directory)` | Таблицы в csv, параметры запуска в json |

### `validation` — стабильность и отчёт по модели

| Функция | Что делает | Возвращает |
|---|---|---|
| `plot_stab(values, target, time, n_buckets, feature_nm, period, add_psi, ...)` | Стабильность признака во времени: WoE по бакетам, доли бакетов, badrate и IV по периодам (две фигуры по две панели); `add_psi=True` — PSI между соседними периодами | `None` или `[fig1, fig2]` при `return_plotly_fig=True` |
| `stability_table(values, target, time, n_buckets, ...)` | Расчётная часть `plot_stab` | `(buckets, periods)` |
| `metric_dynamics(df, target, score, date_col, segment, freq, metric)` | Метрика скора по периодам и сегментам | `Result` |
| `feature_metric_dynamics(train, data, features, target, date_col, freq)` | Однофакторная метрика каждого признака по периодам: модель обучается на `train`, оценивается в периодах `data` | `Result` |
| `learning_curve(model)` | Метрика по итерациям обучения LightGBM | `Figure` |
| `model_report(model, split, features, date_col, segment, n_buckets, freq)` | Отчёт по модели на train / val / test; с `date_col` — ещё AUC и PSI признаков по периодам | `ModelReport` |
| `compare_scores(y_true, scores)` | AUC, Gini, KS нескольких скоров на одной выборке | `DataFrame` |

`ModelReport`:

| Атрибут / метод | Что это |
|---|---|
| `report.metrics` | AUC, Gini, KS, logloss, PSI скора по частям выборки |
| `report.calibration` | Метрики gain chart по частям |
| `report.feature_psi` | PSI каждого признака: train → test целиком |
| `report.feature_psi_by_period` | PSI каждого признака по периодам test относительно train: `feature, part, period, n, psi, status` (нужен `date_col`) |
| `report.plot_feature_psi(features, top_k, n_cols, size)` | Графики PSI по периодам — по одному на признак, с порогами 0.1 и 0.25; по умолчанию все признаки, самые нестабильные первыми |
| `report.segments`, `report.dynamics` | Метрики по сегментам; AUC по периодам |
| `report.figures` | Графики: `gain_charts`, `feature_importance`, `auc_dynamics`, `feature_psi_by_period` |
| `report.show()`, `report.to_html(path)`, `report.save(directory)` | Показать в Jupyter; один HTML; csv + `report.html` в папку |

### `tracking` — эксперименты

`Experiment(project, root, version, name, clearml, clearml_offline, tags, overwrite, capture_plots)` —
см. [Версионирование экспериментов](#версионирование-экспериментов).

| Метод | Что делает |
|---|---|
| `exp.log_params(params, name)` | Параметры → `meta.json` и ClearML Configuration |
| `exp.log_metrics(metrics, prefix)` | Скалярные метрики → `meta.json` и ClearML |
| `exp.save_features(features, cat_features, target, **extra)` / `exp.load_features()` | `features.json`: признаки, категориальные, таргет |
| `exp.save_model(model, name)` / `exp.load_model(name)` | Модель → `models/<name>.pkl` |
| `exp.export_model(model, name, X_check)` | Выгрузка для инференса без библиотеки в `export/<name>/` |
| `exp.save_table(df, name)` | Таблица → `logs/<name>.csv` |
| `exp.save_figure(fig, name)` | График → `logs/<name>.html` |
| `exp.save_result(result, name)` | `Result`: таблица, признаки, `info` и график |
| `exp.save_pipeline(pipe, name)` | `SelectionPipeline`: сводка, лог, воронка, результаты шагов |
| `exp.save_report(report, name)` | `ModelReport`: таблицы и графики, локально — `report.html` |
| `exp.save_feature_test(test, name)` | `FeatureTest`: таблицы и графики прироста |
| `exp.log(obj, name)` | Универсальный вариант: тип объекта определяется сам |
| `exp.plots_summary(check)` | Что произошло с каждым отправленным в ClearML графиком |
| `exp.flush()` / `exp.close()` | Дождаться отправки в ClearML / завершить задачу |
| `Experiment.list_versions(project, root)` | Таблица версий проекта с метриками |
| `Experiment.current()` | Активный эксперимент с ClearML или `None` |

`tracking.clearml` — функции текущей задачи ClearML; без установленного `clearml` ничего не делают.

| Функция | Что делает |
|---|---|
| `init_task(project, name, tags, offline)` / `close_current_task()` | Создать / закрыть задачу |
| `report_figure(fig, title)` | Plotly-график в Plots; слишком большие фигуры прореживаются |
| `report_table(df, title)` | Таблица в Plots |
| `report_metrics(metrics, title)` | Скалярные метрики |
| `connect_params(params, name)` | Параметры в Configuration |
| `upload_artifact(name, obj)` | Артефакт задачи |
| `plots_summary()` / `verify_plots()` | Какие графики отправлены и дошли ли до сервера |
| `diagnose()` | Проверка, почему графики не попадают в Plots: версии, режим, лимит размера сервера |

### `plotting` — стиль и компоновка графиков

| Функция | Что делает |
|---|---|
| `style(fig, title, height, width, xaxis_title, yaxis_title)` | Применяет общий стиль библиотеки |
| `combine(figures, titles, n_cols, size, title)` | Собирает несколько фигур в одну сетку |
| `histogram(x, bins, name)` | Гистограмма с предподсчитанными бинами: размер фигуры не зависит от числа строк |
| `color(i)` / `bar_colors(values)` | i-й цвет палитры / зелёный и красный по знаку значений |
| `figure_size_mb(fig)` | Размер фигуры в МБ в том виде, в каком она уходит в ClearML |
| `shrink_figure(fig, max_points)` | Копия фигуры с прореженными длинными массивами точек |

### `config` и `utils`

| Что | Назначение |
|---|---|
| `config.LGBM_PARAMS`, `CATBOOST_PARAMS` | Параметры моделей по умолчанию |
| `config.LGBM_UNIVARIATE_PARAMS`, `LGBM_INCREMENTAL_PARAMS` | Маленькие модели для однофакторной и пошаговой оценки |
| `config.RANDOM_STATE`, `EARLY_STOPPING_ROUNDS` | 42 и 100 |
| `config.merge_params(defaults, params)` | Копия дефолтов, обновлённая пользовательскими параметрами |
| `utils.io.write_csv(df, path)` | CSV, который корректно открывается в Excel (кириллица) |
| `utils.parallel.tqdm_joblib(total, desc)` | Прогресс-бар tqdm для `joblib.Parallel` |
