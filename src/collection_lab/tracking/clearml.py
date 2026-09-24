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


_WATCHES: list[ErrorWatch] = []  # активные наблюдатели (их ставит на паузу diagnose)


# --- перехват ошибок ClearML ---------------------------------------------------------------
class ErrorWatch(logging.Handler):
    """Копит сообщения уровня ERROR из логгеров ClearML (например, «request exceeds limit»).

    Отправка событий идёт в фоне, поэтому об отклонённых сервером графиках узнаёшь только
    из лога; здесь эти сообщения сохраняются и потом выводятся предупреждением.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []
        self.paused = False
        self._logger = logging.getLogger("clearml")

    def emit(self, record: logging.LogRecord) -> None:
        if self.paused:
            return
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if msg not in self.messages:
            self.messages.append(msg)

    def start(self) -> ErrorWatch:
        if self not in self._logger.handlers:
            self._logger.addHandler(self)
        if self not in _WATCHES:
            _WATCHES.append(self)
        return self

    def stop(self) -> None:
        self._logger.removeHandler(self)
        if self in _WATCHES:
            _WATCHES.remove(self)

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
_REPORT_LOG: list[dict[str, Any]] = []  # по записи на каждую попытку отправки графика/таблицы


def reported_plots() -> list[str]:
    """Названия графиков, отправленных в этом процессе (для :func:`verify_plots`)."""
    return list(_REPORTED_PLOTS)


def reset_reported_plots() -> None:
    _REPORTED_PLOTS.clear()
    _REPORT_LOG.clear()


def _log_report(title: str, kind: str, sent: bool, reason: str,
                size_mb: float = float("nan")) -> None:
    _REPORT_LOG.append({"title": title, "kind": kind, "sent": sent, "reason": reason,
                        "size_mb": round(size_mb, 3) if size_mb == size_mb else size_mb})


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
    Результат каждой попытки (в том числе причина отказа) пишется в журнал —
    см. :func:`plots_summary`.
    """
    logger = _logger()
    if logger is None:
        _log_report(title, "figure", False, "нет активной задачи ClearML")
        return False
    fitted = _fit_figure(fig, title, max_mb if max_mb is not None else max_plot_mb())
    if fitted is None:
        _log_report(title, "figure", False, "слишком большой даже после прореживания")
        return False
    try:
        logger.report_plotly(title=title, series=series, iteration=iteration, figure=fitted)
    except Exception as e:  # noqa: BLE001 — фиксируем причину, не роняем обучение
        _log_report(title, "figure", False, f"{type(e).__name__}: {e}"[:200])
        warnings.warn(f"График {title!r} не отправлен в ClearML: {type(e).__name__}: {e}",
                      RuntimeWarning, stacklevel=3)
        return False
    _REPORTED_PLOTS.append(title)
    size = figure_size_mb(fitted)
    _log_report(title, "figure", True, "уменьшен" if fitted is not fig else "ok", size)
    if fitted is not fig or size > 1.0:
        flush()  # крупные фигуры не копим: ClearML склеивает события в один запрос с лимитом
    return True


def report_table(df: pd.DataFrame, title: str, series: str = "table", iteration: int = 0) -> bool:
    """DataFrame в раздел Plots (таблица)."""
    logger = _logger()
    if logger is None:
        _log_report(title, "table", False, "нет активной задачи ClearML")
        return False
    try:
        logger.report_table(title=title, series=series, iteration=iteration, table_plot=df)
    except Exception as e:  # noqa: BLE001
        _log_report(title, "table", False, f"{type(e).__name__}: {e}"[:200])
        warnings.warn(f"Таблица {title!r} не отправлена в ClearML: {type(e).__name__}: {e}",
                      RuntimeWarning, stacklevel=3)
        return False
    _REPORTED_PLOTS.append(title)
    _log_report(title, "table", True, "ok")
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
def _server_plot_titles(task) -> set[str] | None:
    """Названия графиков, которые есть на сервере, или ``None``, если это узнать нельзя.

    Используется список метрик с событиями типа plot — он не зависит от размера графиков.
    Чтение самих графиков (``get_reported_plots``) для проверки не годится: сервер отдаёт его
    страницами, ограниченными по размеру, и крупные графики в ответ не попадают, хотя сохранены.
    """
    try:
        from clearml.backend_api.services import events

        response = task.send(events.GetTaskMetricsRequest(tasks=[task.id], event_type="plot"))
        titles: set[str] = set()
        for item in response.response.metrics:
            for m in item["metrics"]:
                titles.add(m if isinstance(m, str) else (m["metric"] if isinstance(m, dict)
                                                         else m.metric))
        return titles
    except Exception:  # noqa: BLE001 — старый сервер/SDK: проверить нельзя, но и не тревожим
        return None


def plots_summary(*, check: bool = True, wait: float = 3.0) -> pd.DataFrame:
    """Что произошло с каждым графиком: отправлен ли, почему нет, дошёл ли до сервера.

    Колонки: ``title, kind, sent, reason, size_mb, on_server`` (``on_server``: ``True`` —
    сервер подтвердил, ``False`` — не дошёл, ``None`` — проверить нельзя или не проверялось).
    """
    table = pd.DataFrame(_REPORT_LOG, columns=["title", "kind", "sent", "reason", "size_mb"])
    table["on_server"] = None
    task = current_task()
    if check and task is not None and not is_offline() and table["sent"].any():
        try:
            task.flush(wait_for_uploads=True)
            time.sleep(wait)
        except Exception:  # noqa: BLE001
            return table
        got = _server_plot_titles(task)
        if got is not None:
            table["on_server"] = [(t in got) if sent else None
                                  for t, sent in zip(table["title"], table["sent"], strict=True)]
    return table


def verify_plots(expected: list[str] | None = None, *, wait: float = 3.0) -> list[str]:
    """Сверяет с сервером, какие из отправленных графиков дошли; возвращает недошедшие.

    Без сервера (офлайн) или если сервер не позволяет это проверить — пустой список
    (лучше промолчать, чем дать ложную тревогу).
    """
    expected = list(dict.fromkeys(_REPORTED_PLOTS if expected is None else expected))
    task = current_task()
    if task is None or not expected or is_offline():
        return []
    try:
        task.flush(wait_for_uploads=True)
        time.sleep(wait)
    except Exception as e:  # noqa: BLE001 — проверка не должна ронять обучение
        warnings.warn(f"Не удалось сверить графики с сервером ClearML: {e}", RuntimeWarning,
                      stacklevel=2)
        return []
    got = _server_plot_titles(task)
    if got is None:
        return []
    return [t for t in expected if t not in got]


def diagnose(sizes_mb: tuple[float, ...] = (0.05, 1, 3, 6, 10, 14), *, wait: float = 6.0,
             project: str = "collection_lab_diagnose", keep: bool = False):
    """Проверка, почему графики не попадают в Plots: версии, режим, лимит размера сервера.

    Отправляет тестовые графики нарастающего размера (``sizes_mb``) в **отдельную**
    временную задачу (ваша активная задача не затрагивается), узнаёт у сервера, какие
    графики дошли, печатает вывод и удаляет временную задачу (``keep=True`` — оставить,
    чтобы посмотреть вкладку Plots). Запускайте там же, где не сохраняются графики.

    Returns
    -------
    pd.DataFrame
        ``size_mb, delivered`` — размер тестового графика и дошёл ли он до сервера
        (``None``, если сервер не позволяет это проверить автоматически).
    """
    import plotly
    import plotly.graph_objects as go

    if not is_available():
        raise ImportError("Нужен clearml: pip install collection_lab[clearml]")
    import clearml
    from clearml import Task

    print(f"python {sys.version.split()[0]} | clearml {clearml.__version__} | "
          f"plotly {plotly.__version__} | numpy {np.__version__}")
    print(f"лимит фигуры в collection_lab: {max_plot_mb():g} МБ "
          f"(COLLECTION_LAB_CLEARML_MAX_MB)")
    main = current_task()
    print(f"активная задача: {main.name if main else 'нет'} | офлайн: {is_offline()}")
    task = Task.create(project_name=project, task_name="diagnose")
    print(f"временная задача: {task.get_output_log_web_page()}")
    others = list(_WATCHES)
    for w in others:  # намеренные ошибки пробы не должны попасть в чужие эксперименты
        w.paused = True
    watch = ErrorWatch().start()
    verified = True
    try:
        logger = task.get_logger()
        rng = np.random.default_rng(0)

        def probe(n: int):
            return go.Figure(go.Scatter(x=np.arange(n), y=rng.normal(size=n), mode="markers"))

        per_point = figure_size_mb(probe(20_000)) * 1e6 / 20_000  # байт на точку (измерено)
        rows = []
        for size in sizes_mb:
            fig = probe(max(10, int(size * 1e6 / per_point)))
            real = figure_size_mb(fig)
            logger.report_plotly(title=f"diag_{real:.2f}MB", series="probe", iteration=0,
                                 figure=fig)
            rows.append({"title": f"diag_{real:.2f}MB", "size_mb": round(real, 2)})
        task.flush(wait_for_uploads=True)
        time.sleep(wait)
        watch.stop()
        if is_offline():
            print("Офлайн-режим: данные на сервер не уходят, сверить доставку нельзя.")
            keep = True
            return pd.DataFrame(rows).drop(columns="title")
        got = _server_plot_titles(task)
        if got is None:
            verified = False
            keep = True
            table = pd.DataFrame([{**r, "delivered": None} for r in rows]).drop(columns="title")
            print(table.to_string(index=False))
            print("\nСервер не позволяет проверить доставку автоматически. Откройте страницу "
                  "временной задачи (ссылка выше), вкладка Plots: какие из diag_* есть, а какие "
                  "нет. Задача оставлена; удалите её после просмотра.")
            return table
        for r in rows:
            r["delivered"] = r["title"] in got
        table = pd.DataFrame(rows).drop(columns="title")
        print(table.to_string(index=False))
        ok = [r["size_mb"] for r in rows if r["delivered"]]
        lost = [r["size_mb"] for r in rows if not r["delivered"]]
        if not ok:
            print("\nНи один тестовый график не дошёл: проблема не в размере — проверьте "
                  "права на запись в проект и сообщения об ошибках ниже.")
        elif not lost:
            print(f"\nВсе графики до {max(ok)} МБ доставлены: лимит размера не найден. "
                  f"Если ваши графики не сохраняются, причина в другом — пришлите вывод.")
        else:
            lim = max(ok)
            print(f"\nСервер отбрасывает графики больше ~{lim}–{min(lost)} МБ. "
                  f"Рекомендуется: COLLECTION_LAB_CLEARML_MAX_MB={max(0.5, lim * 0.6):.1f}")
        for m in watch.messages:
            print(f"ClearML ERROR: {m}")
        return table
    finally:
        watch.stop()
        for w in others:
            w.paused = False
        if verified and not keep:
            try:
                task.delete(delete_artifacts_and_models=True, skip_models_used_by_other_tasks=True,
                            raise_on_error=False)
            except Exception:  # noqa: BLE001 — очистка необязательна
                print("Временную задачу удалить не удалось — удалите её вручную.")
