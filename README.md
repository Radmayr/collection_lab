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

Полный пример на реальных данных — [examples/rtk_pipeline.ipynb](examples/rtk_pipeline.ipynb).

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
