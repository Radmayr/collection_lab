# collection_lab

Инструменты для разработки, отбора признаков и валидации ML-моделей (бинарная классификация,
LightGBM/CatBoost). Графики — plotly, трекинг экспериментов — локально и в ClearML.

## Установка

С GitHub (на ML Core или любой машине с `pip`):

```bash
pip install "collection_lab @ git+https://github.com/<user>/collection_lab.git"
# с опциональными зависимостями:
pip install "collection_lab[all] @ git+https://github.com/<user>/collection_lab.git"
```

Для разработки (изменения в коде видны сразу, без переустановки):

```bash
git clone https://github.com/<user>/collection_lab.git
cd collection_lab
python -m venv .venv
.venv/Scripts/activate          # Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Опциональные зависимости: `catboost`, `optuna`, `clearml`, `all`, `dev`.

## Структура

| Модуль | Назначение |
|---|---|
| `core` | адаптеры моделей, кросс-валидация с сегментами, объекты результатов |
| `data` | типы признаков, сплиты train/val/test без утечек |
| `eda` | обзор данных, распределения, динамика таргета, винтажи |
| `metrics` | AUC/Gini/KS, PSI/CSI, калибровка, WoE/IV |
| `selection` | фильтры, корреляции, однофакторный анализ, RFE, stepwise, пайплайн отбора |
| `modeling` | обучение, сравнение моделей, Optuna |
| `validation` | метрики в динамике, отчёт по модели |
| `plotting` | общий стиль plotly |
| `tracking` | версии экспериментов, артефакты, ClearML |
| `config` | параметры моделей по умолчанию — единственное место, где они заданы |

## Разработка

```bash
pytest                 # быстрые тесты на синтетике
pytest -m slow         # тесты на реальных выборках (путь: переменная COLLECTION_LAB_DATA)
ruff check src tests   # линтер
ruff format src tests  # форматирование
```

Принципы:

- функции считают и возвращают данные; графики строятся отдельно (`.plot()`), возвращают `plotly.Figure`;
- никаких `print` и глобальных `warnings.filterwarnings` — вывод управляется параметром `verbose`;
- отбор признаков и подбор параметров — только на train, тест используется лишь для отчёта;
- каждая публичная функция покрыта тестом.
