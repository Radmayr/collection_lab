# collection_lab

Библиотека для разработки, отбора признаков и валидации ML-моделей (бинарная классификация,
LightGBM / CatBoost; регрессия поддерживается ядром). Графики — plotly, трекинг — локально
и в ClearML.

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

# эксперимент: папки v_N, модель, признаки, метрики (+ ClearML)
exp = cl.tracking.Experiment("RTK_model", clearml=True)
exp.save_model(model); exp.save_features(pipe.selected_, cat_features=cat_cols)
report.save(exp.logs_path / "report")
```

Полный пример на реальных данных — [examples/rtk_pipeline.ipynb](examples/rtk_pipeline.ipynb).

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
| `data` | `split_feature_types`, `cast_types`, `time_split`, `random_split`, `DataSplit` |
| `core` | `make_model`, `LGBMModel`, `CatBoostModel`, `FeaturePreparer`, `cross_validate`, `make_folds`, `Result` |
| `eda` | `overview`, `target_summary`, `plot_distribution`, `plot_target_rate_by_bins`, `target_dynamics`, `maturation_transactions`, `plot_vintage`, `vintage_by_type`, `eda_transactions`, `add_days_since` |
| `metrics` | `roc_auc`, `gini`, `ks`, `metrics_by_segment`, `psi`, `psi_table`, `feature_psi`, `psi_by_period`, `gain_chart`, `gain_chart_metrics`, `hosmer_lemeshow`, `information_value`, `iv_table` |
| `selection` | `quality_filter`, `correlation_filter`, `univariate_scores`, `cumulative_importance_selection`, `drop_column_importance`, `permutation_importance`, `rfe`, `backward_elimination`, `forward_addition`, `incremental_feature_eval`, `SelectionPipeline` |
| `modeling` | `train_model`, `compare_models`, `tune_hyperparams`, `sample_size_curve` |
| `validation` | `plot_stab`, `metric_dynamics`, `feature_metric_dynamics`, `learning_curve`, `model_report` |
| `tracking` | `Experiment`, `clearml.*` |
| `plotting` | `style`, `combine` (сетка графиков) |

## Переход со старого `scr`

| Было (`scr/…`, ноутбуки) | Стало |
|---|---|
| `find_project_root` + `sys.path` | `pip install` и `import collection_lab as cl` |
| `make_lgbm_ready` | `core.make_lgbm_ready` (совместимо) или `core.FeaturePreparer` — автоматически внутри моделей |
| `select_categorical_features` | `data.detect_categorical` / `data.split_feature_types` |
| `missing_exclusion` | `selection.quality_filter` |
| `select_features_by_auc_cv_hybrid` | `selection.correlation_filter(strategy="hybrid")` |
| `select_features_by_auc_cv` | `selection.correlation_filter(strategy="model", model="catboost")` |
| `evaluate_features_univariate` | `selection.univariate_scores` |
| `cum_importance_select_and_compare` | `selection.cumulative_importance_selection` |
| `rfe_lightgbm_binary_auc`, `…_segment`, `rfe_catboost_binary_auc` | `selection.rfe(model=…, segments=…)` |
| `backward_feature_elimination`, `plot_elimination_history`, `plot_delta_per_feature` | `selection.backward_elimination(...).plot()` / `.plot(kind="per_feature")` |
| `forward_feature_addition`, `plot_forward_addition` | `selection.forward_addition(...).plot()` |
| `incremental_feature_eval`, `plot_incremental_curve` | `selection.incremental_feature_eval(...).plot()` |
| `evaluate_feature_importance_by_removal` | `selection.drop_column_importance` (полная модель считается сама) |
| `train_lgbm` | `modeling.train_model(split, features)` |
| `compare_catboost_vs_lgbm(_onedf)` | `modeling.compare_models` |
| Optuna в ноутбуке | `modeling.tune_hyperparams` |
| эксперимент «объём выборки» | `modeling.sample_size_curve` |
| `psi`, `q_psi` | `metrics.psi` (авто: категориальный / квантильный), `metrics.psi_table` |
| `dynamic_auc` | `validation.metric_dynamics(...).plot()` |
| однофакторный AUC по месяцам | `validation.feature_metric_dynamics` |
| `plot_feature_importance` | `selection.importance_plot(model.feature_importance())` |
| `plot_in_line` | `plotting.combine` |
| `show_gainchart_metrics` | `metrics.gain_chart_metrics` |
| `target_dynamics` | `eda.target_dynamics(...).plot()` (сегмент необязателен) |
| `plot_distribution_with_clip` | `eda.plot_distribution` |
| `maturation_transactions`, `plot_vintage`, `plot_vintage_by_tx_type`, `eda_transactions` | `eda.maturation_transactions`, `eda.plot_vintage`, `eda.vintage_by_type`, `eda.eda_transactions` |
| `risk_instruments…gain_chart` | `metrics.gain_chart` (тот же интерфейс) |
| `risk_instruments…plot_stab` | `validation.plot_stab` (тот же интерфейс) |
| ручные `logs/v_1`, `models/v_1`, `features.json` | `tracking.Experiment` |

### Что исправлено по сравнению со старым кодом

- `train_lgbm(auto_scale_pos_weight=True)` падал (`y_tr` не определён).
- `rfe_catboost_binary_auc` падал при явных `cat_features`; `compare_catboost_vs_lgbm` — при
  `cat_cols=None`.
- `q_psi`: подписи бинов шли не по порядку (значение PSI не менялось).
- `make_lgbm_ready` вызывался отдельно на train/val/test — категории кодировались по-разному;
  теперь категории фиксируются по train.
- `subsample` без `subsample_freq` в LightGBM не включал бэггинг — в `config` добавлен
  `subsample_freq=1`.
- `warnings.filterwarnings` на уровне модулей глушил предупреждения во всём ноутбуке — убрано.
- Утечки в ноутбуках RTK: val был частью train; лучшие параметры Optuna выбирались по test;
  отбор признаков шёл на всей выборке вместе с OOT.

### Отличия от `risk_instruments`

- `gain_chart`: логика восстановлена по выводу и сверена с числами (бакеты, 99% ДИ,
  `offset`, полная калибровка). `hl` — стандартная формула Хосмера–Лемешоу по бакетам
  графика; значение может отличаться от исходной функции.
- `plot_stab`: панели и оформление повторяют оригинал; бакеты для признаков с повторяющимися
  значениями совпадают (`[0]`, `[1]`, …), для непрерывных граница может сдвинуться на одно
  соседнее значение.

## Разработка

```bash
pytest                 # быстрые тесты на синтетике
pytest -m slow         # тесты на реальных выборках (папка: переменная COLLECTION_LAB_DATA)
ruff check src tests   # линтер
```

Принципы: функции считают и возвращают данные, графики — через `.plot()` (plotly);
вывод управляется `verbose`; каждая публичная функция покрыта тестом.
