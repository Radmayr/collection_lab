# Changelog

## [0.1.1] — 2026-09-24

### Исправлено
- `Experiment(clearml=True)` падал при повторном запуске в одном ядре (`UsageError: Current task
  already created and requested task name 'v_2' does not match ... 'v_1'`): предыдущая задача
  ClearML теперь закрывается автоматически.
- Ошибка создания задачи ClearML оставляла пустую папку версии и «сжигала» номер.
- Явная `version=N` при существующей папке молча перезаписывала файлы — теперь `FileExistsError`
  (или `overwrite=True`).
- В git не попадал пакет `collection_lab.data` (правило `.gitignore` `data/`).

### Добавлено
- `Experiment` как контекстный менеджер, идемпотентный `close()`, `Experiment.list_versions`.

## [0.1.0] — 2026-09-23

Первая версия: перенос наработок из `ML_lib/scr` и ноутбуков RTK в устанавливаемый пакет.

### Добавлено
- `data`: определение и приведение типов, `time_split` / `random_split`, `DataSplit`
  с проверкой непересечения частей.
- `core`: адаптеры LightGBM / CatBoost с единым интерфейсом, `FeaturePreparer`
  (категории фиксируются по train), `cross_validate` с сегментами и OOF, `Result`.
- `metrics`: AUC / Gini / KS / logloss, метрики по сегментам, PSI / CSI / PSI по периодам,
  `gain_chart` (интерфейс `risk_instruments`), калибровка, Хосмер–Лемешоу, WoE / IV.
- `selection`: `quality_filter`, `correlation_filter`, `univariate_scores`, важности
  (кумулятивная, permutation, drop-column), единый `rfe`, backward / forward / incremental,
  `SelectionPipeline`.
- `eda`: обзор таблицы, распределения, динамика таргета, винтажи и EDA транзакций.
- `modeling`: `train_model`, `compare_models`, `tune_hyperparams` (Optuna, выбор по val),
  `sample_size_curve`.
- `validation`: `plot_stab` (интерфейс `risk_instruments`), динамика метрик,
  `model_report` с HTML-выгрузкой.
- `tracking`: `Experiment` (версии `v_N`, модель, признаки, метрики, графики) + ClearML.
- Все графики — plotly; параметры по умолчанию — в `config.py`.

### Исправлено относительно `scr`
- См. раздел «Что исправлено» в README.
