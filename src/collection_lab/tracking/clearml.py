"""Тонкая обёртка над ClearML: всё работает и без него (вызовы становятся no-op).

Логирование идёт в текущую задачу ``Task.current_task()`` — её создаёт
:class:`~collection_lab.tracking.experiment.Experiment` с ``clearml=True`` или
пользователь сам через ``Task.init``.

**Про потерянные графики.** ``Logger.report_plotly`` не выбрасывает исключений, даже если
сервер отверг данные: запрос с графиком больше лимита сервера (или прокси) молча отбрасывается,
а в лог попадает единственная строка ``ERROR ... request exceeds limit``. Поэтому здесь:

- перед отправкой проверяется размер фигуры (лимит ``MAX_PLOT_MB``, можно менять переменной
  окружения ``COLLECTION_LAB_CLEARML_MAX_MB``); слишком большие фигуры прореживаются
  (:func:`~collection_lab.plotting.shrink_figure`) с предупреждением;
- :class:`ErrorWatch` собирает сообщения об ошибках ClearML и показывает их предупреждением;
- :func:`verify_plots` сверяет, какие графики реально дошли до сервера;
- :func:`diagnose` находит фактический лимит вашего сервера.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from collection_lab.plotting.theme import figure_size_mb, shrink_figure

MAX_PLOT_MB = 5.0
"""Лимит размера одной фигуры в МБ (у сервера/прокси он свой; ``diagnose()`` покажет его)."""


def max_plot_mb() -> float:
    """Лимит размера фигуры: ``COLLECTION_LAB_CLEARML_MAX_MB`` или ``MAX_PLOT_MB``."""
    try:
        return float(os.environ.get("COLLECTION_LAB_CLEARML_MAX_MB", MAX_PLOT_MB))
    except ValueError:
        return MAX_PLOT_MB


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


def is_offline() -> bool:
    """Работает ли ClearML в офлайн-режиме (тогда данные на сервер не уходят)."""
    if not is_available():
        return False
    from clearml import Task

    check = getattr(Task, "is_offline", None)
    return bool(check()) if check else False


def flush(wait: bool = True) -> None:
    """Дожидается отправки накопленных событий на сервер (если задача активна)."""
    task = current_task()
    if task is not None:
        task.flush(wait_for_uploads=wait)


def _logger():
    task = current_task()
    return None if task is None else task.get_logger()


# --- перехват ошибок ClearML ---------------------------------------------------------------
class ErrorWatch(logging.Handler):
    """Копит сообщения уровня ERROR из логгеров ClearML (например, «request exceeds limit»).

    Отправка событий идёт в фоне, поэтому об отклонённых сервером графиках узнаёшь только
    из лога; здесь эти сообщения сохраняются и потом выводятся предупреждением.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []
        self._logger = logging.getLogger("clearml")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if msg not in self.messages:
            self.messages.append(msg)

    def start(self) -> ErrorWatch:
        if self not in self._logger.handlers:
            self._logger.addHandler(self)
        return self

    def stop(self) -> None:
        self._logger.removeHandler(self)

    def warn(self) -> None:
        if not self.messages:
            return
        text = "\n".join(f"  - {m}" for m in self.messages[:5])
        hint = ""
        if any("exceeds limit" in m for m in self.messages):
            hint = ("\nСервер отклонил слишком большой запрос: график не сохранён. Узнать лимит: "
                    "collection_lab.tracking.clearml.diagnose(); задать: "
                    "COLLECTION_LAB_CLEARML_MAX_MB.")
        warnings.warn(f"ClearML сообщил об ошибках:\n{text}{hint}", RuntimeWarning, stacklevel=3)


# --- отчёты --------------------------------------------------------------------------------
_REPORTED_PLOTS: list[str] = []


def reported_plots() -> list[str]:
    """Названия графиков, отправленных в этом процессе (для :func:`verify_plots`)."""
    return list(_REPORTED_PLOTS)


def reset_reported_plots() -> None:
    _REPORTED_PLOTS.clear()


def _fit_figure(fig, title: str, limit_mb: float):
    """Возвращает фигуру, укладывающуюся в лимит (прореживая длинные массивы), или ``None``."""
    size = original = figure_size_mb(fig)
    if size <= limit_mb:
        return fig
    longest = max((len(getattr(t, a)) for t in fig.data for a in ("x", "y")
                   if getattr(t, a, None) is not None and hasattr(getattr(t, a), "__len__")),
                  default=0)
    points = longest
    for _ in range(6):
        points = max(200, int(points * min(0.8, limit_mb / max(size, 1e-9))))
        small = shrink_figure(fig, points)
        new_size = figure_size_mb(small)
        if new_size <= limit_mb:
            warnings.warn(f"График {title!r} весил {original:.1f} МБ (лимит {limit_mb:g} МБ) — "
                          f"уменьшен до {points:,} точек ({new_size:.2f} МБ).", RuntimeWarning,
                          stacklevel=4)
            return small
        size = new_size
        fig = small
    warnings.warn(f"График {title!r} слишком большой даже после прореживания "
                  f"({size:.1f} МБ > {limit_mb:g} МБ) и не отправлен в ClearML.", RuntimeWarning,
                  stacklevel=4)
    return None


def report_figure(fig, title: str, series: str = "plot", iteration: int = 0,
                  max_mb: float | None = None) -> bool:
    """Plotly-фигура в раздел Plots текущей задачи. Возвращает ``True``, если отправлено.

    Фигуры больше лимита (``max_mb`` или :func:`max_plot_mb`) прореживаются с предупреждением.
    """
    logger = _logger()
    if logger is None:
        return False
    fitted = _fit_figure(fig, title, max_mb if max_mb is not None else max_plot_mb())
    if fitted is None:
        return False
    logger.report_plotly(title=title, series=series, iteration=iteration, figure=fitted)
    _REPORTED_PLOTS.append(title)
    if fitted is not fig or figure_size_mb(fitted) > 1.0:
        flush()  # крупные фигуры не копим: ClearML склеивает события в один запрос с лимитом
    return True


def report_table(df: pd.DataFrame, title: str, series: str = "table", iteration: int = 0) -> bool:
    """DataFrame в раздел Plots (таблица)."""
    logger = _logger()
    if logger is None:
        return False
    logger.report_table(title=title, series=series, iteration=iteration, table_plot=df)
    _REPORTED_PLOTS.append(title)
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


# --- проверка доставки и диагностика ---------------------------------------------------------
def _server_plot_titles(task) -> set[str]:
    return {p["metric"] for p in task.get_reported_plots(max_iterations=5)}


def verify_plots(expected: list[str] | None = None, *, wait: float = 3.0) -> list[str]:
    """Сверяет с сервером, какие из отправленных графиков дошли; возвращает недошедшие.

    Без сервера (офлайн) или при ошибке чтения возвращает пустой список.
    """
    expected = list(dict.fromkeys(_REPORTED_PLOTS if expected is None else expected))
    task = current_task()
    if task is None or not expected or is_offline():
        return []
    try:
        task.flush(wait_for_uploads=True)
        time.sleep(wait)
        got = _server_plot_titles(task)
    except Exception as e:  # noqa: BLE001 — проверка не должна ронять обучение
        warnings.warn(f"Не удалось сверить графики с сервером ClearML: {e}", RuntimeWarning,
                      stacklevel=2)
        return []
    return [t for t in expected if t not in got]


def diagnose(sizes_mb: tuple[float, ...] = (0.05, 1, 3, 6, 10, 14), *, wait: float = 6.0,
             project: str = "collection_lab_diagnose"):
    """Проверка, почему графики не попадают в Plots: версии, режим, лимит размера сервера.

    Отправляет в ClearML тестовые графики нарастающего размера (``sizes_mb``), читает с
    сервера, какие дошли, и печатает вывод. Запускайте там же, где не сохраняются графики
    (например, на ML Core). Создаёт задачу ``collection_lab_diagnose``, если активной нет.

    Returns
    -------
    pd.DataFrame
        ``size_mb, delivered`` — размер тестового графика и дошёл ли он до сервера.
    """
    import plotly
    import plotly.graph_objects as go

    if not is_available():
        raise ImportError("Нужен clearml: pip install collection_lab[clearml]")
    import clearml

    print(f"python {sys.version.split()[0]} | clearml {clearml.__version__} | "
          f"plotly {plotly.__version__} | numpy {np.__version__}")
    print(f"лимит фигуры в collection_lab: {max_plot_mb():g} МБ "
          f"(COLLECTION_LAB_CLEARML_MAX_MB)")
    created = current_task() is None
    task = current_task() or init_task(project, "diagnose")
    try:
        print(f"задача: {task.name} ({task.id}) | офлайн: {is_offline()}")
        print(f"страница: {task.get_output_log_web_page()}")
        watch = ErrorWatch().start()
        logger = task.get_logger()
        rng = np.random.default_rng(0)

        def probe(n: int):
            return go.Figure(go.Scatter(x=np.arange(n), y=rng.normal(size=n), mode="markers"))

        per_point = figure_size_mb(probe(20_000)) * 1e6 / 20_000  # байт на точку (измерено)
        rows = []
        for size in sizes_mb:
            n = max(10, int(size * 1e6 / per_point))
            fig = probe(n)
            real = figure_size_mb(fig)
            logger.report_plotly(title=f"diag_{real:.2f}MB", series="probe", iteration=0,
                                 figure=fig)
            rows.append({"title": f"diag_{real:.2f}MB", "size_mb": round(real, 2)})
        task.flush(wait_for_uploads=True)
        time.sleep(wait)
        watch.stop()
        if is_offline():
            print("Офлайн-режим: данные на сервер не уходят, сверить доставку нельзя.")
            return pd.DataFrame(rows).drop(columns="title")
        got = _server_plot_titles(task)
        for r in rows:
            r["delivered"] = r["title"] in got
        table = pd.DataFrame(rows).drop(columns="title")
        print(table.to_string(index=False))
        ok = [r["size_mb"] for r in rows if r["delivered"]]
        if not ok:
            print("\nНи один тестовый график не дошёл: проблема не в размере — проверьте "
                  "права на запись в проект и сообщения об ошибках ниже.")
        elif len(ok) == len(rows):
            print(f"\nВсе графики до {max(ok)} МБ доставлены: лимит размера не найден. "
                  f"Если ваши графики не сохраняются, причина в другом — пришлите вывод.")
        else:
            lim = max(ok)
            print(f"\nСервер отбрасывает графики больше ~{lim}–"
                  f"{min(r['size_mb'] for r in rows if not r['delivered'])} МБ. "
                  f"Рекомендуется: COLLECTION_LAB_CLEARML_MAX_MB={max(0.5, lim * 0.6):.1f}")
        for m in watch.messages:
            print(f"ClearML ERROR: {m}")
        return table
    finally:
        if created:
            task.close()
