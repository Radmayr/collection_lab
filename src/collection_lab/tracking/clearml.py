"""Тонкая обёртка над ClearML: всё работает и без него (вызовы становятся no-op).

Логирование идёт в текущую задачу ``Task.current_task()`` — её создаёт
:class:`~collection_lab.tracking.experiment.Experiment` с ``clearml=True`` или
пользователь сам через ``Task.init``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def is_available() -> bool:
    """Установлен ли пакет clearml."""
    try:
        import clearml  # noqa: F401
    except ImportError:
        return False
    return True


def current_task():
    """Текущая задача ClearML или ``None``."""
    if not is_available():
        return None
    from clearml import Task

    return Task.current_task()


def close_current_task() -> str | None:
    """Закрывает активную задачу ClearML этого процесса (если есть) и возвращает её имя.

    ClearML допускает одну активную задачу на процесс: повторный ``Task.init`` с другим
    именем без закрытия предыдущей падает с ``UsageError``.
    """
    task = current_task()
    if task is None:
        return None
    name = task.name
    task.close()
    return name


def init_task(project: str, name: str, *, tags: list[str] | None = None, offline: bool = False,
              **kwargs: Any):
    """Создаёт задачу ClearML (``offline=True`` — без сервера, архив сохраняется локально)."""
    if not is_available():
        raise ImportError("Нужен clearml: pip install collection_lab[clearml]")
    from clearml import Task

    if offline:
        Task.set_offline(offline_mode=True)
    return Task.init(project_name=project, task_name=name, tags=tags,
                     reuse_last_task_id=False, **kwargs)


def _logger():
    task = current_task()
    return None if task is None else task.get_logger()


def report_figure(fig, title: str, series: str = "plot", iteration: int = 0) -> bool:
    """Plotly-фигура в раздел Plots текущей задачи. Возвращает ``True``, если залогировано."""
    logger = _logger()
    if logger is None:
        return False
    logger.report_plotly(title=title, series=series, iteration=iteration, figure=fig)
    return True


def report_table(df: pd.DataFrame, title: str, series: str = "table", iteration: int = 0) -> bool:
    """DataFrame в раздел Plots (таблица)."""
    logger = _logger()
    if logger is None:
        return False
    logger.report_table(title=title, series=series, iteration=iteration, table_plot=df)
    return True


def report_metrics(metrics: dict[str, float], title: str = "metrics") -> bool:
    """Скалярные метрики как single values (Summary в ClearML)."""
    logger = _logger()
    if logger is None:
        return False
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and value == value:  # пропускаем NaN
            logger.report_single_value(name=f"{title}/{name}", value=float(value))
    return True


def connect_params(params: dict[str, Any], name: str = "params") -> bool:
    """Параметры в раздел Configuration текущей задачи."""
    task = current_task()
    if task is None:
        return False
    task.connect(params, name=name)
    return True


def upload_artifact(name: str, obj: Any) -> bool:
    """Артефакт (файл, DataFrame, dict, объект) в текущую задачу."""
    task = current_task()
    if task is None:
        return False
    task.upload_artifact(name=name, artifact_object=obj)
    return True
