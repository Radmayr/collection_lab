"""collection_lab — инструменты для разработки, отбора признаков и валидации ML-моделей.

Подмодули:
    core        — адаптеры моделей, кросс-валидация, объекты результатов
    data        — типы признаков, сплиты train/val/test без утечек
    eda         — обзор данных, распределения, динамика таргета, винтажи
    metrics     — AUC/Gini/KS, PSI/CSI, калибровка (gain chart), WoE/IV
    selection   — фильтры, корреляции, однофакторный анализ, RFE, stepwise, пайплайн отбора
    modeling    — обучение, сравнение моделей, Optuna, объём выборки
    validation  — стабильность признаков (plot_stab), метрики в динамике, отчёт по модели
    plotting    — общий стиль plotly
    tracking    — версии экспериментов, артефакты, ClearML

Пример::

    import collection_lab as cl

    split = cl.data.time_split(df, "target", "rtk_send_date", oot_from="2024-03-01")
    pipe = cl.selection.SelectionPipeline([...]).fit(split.train, "target", features)
    model = cl.modeling.train_model(split, pipe.selected_)
    report = cl.validation.model_report(model, split, date_col="rtk_send_date")
"""

__version__ = "0.1.1"

from collection_lab import (  # noqa: E402
    core,
    data,
    eda,
    metrics,
    modeling,
    plotting,
    selection,
    tracking,
    validation,
)

__all__ = ["core", "data", "eda", "metrics", "modeling", "plotting", "selection", "tracking",
           "validation"]
