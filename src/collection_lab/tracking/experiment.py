"""Версионированный эксперимент: папки logs/models, признаки, параметры, метрики, графики.

Пример::

    with Experiment("RTK_model", root="/workdir", clearml=True) as exp:   # v_1, v_2, ...
        exp.log_params(params)
        exp.save_features(features, cat_features=cats, target="target")
        exp.save_model(model)
        exp.log_metrics({"auc_test": 0.63})
        exp.save_result(rfe_result)
        exp.save_figure(fig, "gain_chart")

    Experiment.list_versions("RTK_model", root="/workdir")   # все версии и их метрики

Структура на диске::

    <root>/<project>/v_<N>/
        models/   — модели (joblib)
        logs/     — таблицы, json, графики (html)
        meta.json — параметры, метрики, время
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from collection_lab.core.results import Result, _jsonable
from collection_lab.tracking import clearml as cml
from collection_lab.utils.io import write_csv


def _next_version(project_dir: Path) -> int:
    versions = [int(m.group(1)) for p in project_dir.glob("v_*")
                if (m := re.fullmatch(r"v_(\d+)", p.name))]
    return max(versions, default=0) + 1


class Experiment:
    """Эксперимент с локальным хранением и (опционально) ClearML.

    Parameters
    ----------
    project : str
        Имя проекта (папка и проект в ClearML).
    root : str | Path
        Корень, где создаётся ``<project>/v_<N>``.
    version : int | str
        ``"auto"`` — следующая свободная версия (максимальный ``v_N`` в папке проекта + 1).
        Число — конкретная версия; если папка ``v_N`` уже содержит данные, будет
        ``FileExistsError`` (защита от случайной перезаписи), либо передайте ``overwrite=True``.
    name : str, optional
        Имя задачи в ClearML (по умолчанию ``v_<N>``).
    clearml : bool
        Создать задачу ClearML и дублировать туда всё, что логируется. Если в этом процессе
        (например, в том же ноутбуке) уже есть активная задача ClearML — она закрывается
        автоматически, поэтому ячейку можно перезапускать.
    clearml_offline : bool
        Режим ClearML без сервера.
    overwrite : bool
        Разрешить использовать уже существующую версию (файлы могут быть перезаписаны).

    Версионирование
    ---------------
    Локально версия — это папка ``<root>/<project>/v_<N>``; номер выдаётся один раз при
    создании ``Experiment`` и не переиспользуется. В ClearML версия — имя задачи ``v_<N>``,
    а идентичность задачи — её id: одинаковые имена в разных запусках допустимы, поэтому
    номер ``N`` берётся из локальных папок и совпадает в обоих местах. Список версий:
    :meth:`Experiment.list_versions`.
    """

    def __init__(
        self,
        project: str,
        root: str | Path = "experiments",
        version: int | str = "auto",
        *,
        name: str | None = None,
        clearml: bool = False,
        clearml_offline: bool = False,
        tags: list[str] | None = None,
        overwrite: bool = False,
    ):
        self.project = project
        project_dir = Path(root) / project
        project_dir.mkdir(parents=True, exist_ok=True)
        self.version = _next_version(project_dir) if version == "auto" else int(version)
        self.path = project_dir / f"v_{self.version}"
        if self.path.exists() and any(self.path.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Версия v_{self.version} уже существует: {self.path}. Используйте "
                f"version='auto' для новой версии или overwrite=True, чтобы перезаписать."
            )
        self.models_path = self.path / "models"
        self.logs_path = self.path / "logs"
        self.meta: dict[str, Any] = {"project": project, "version": self.version,
                                     "created": datetime.now().isoformat(timespec="seconds"),
                                     "params": {}, "metrics": {}}
        self.task = None
        self._watch = None
        self._summary_printed = False
        cml.reset_reported_plots()  # журнал отправок графиков — на каждый эксперимент свой
        if clearml:
            # сначала ClearML: при ошибке не остаётся пустой локальной папки версии
            previous = cml.close_current_task()
            if previous:
                print(f"[collection_lab] закрыта активная задача ClearML {previous!r} "
                      f"(одна активная задача на процесс)")
            self.task = cml.init_task(project, name or f"v_{self.version}", tags=tags,
                                      offline=clearml_offline)
            self._watch = cml.ErrorWatch().start()  # ошибки ClearML в фоне: покажем при close()
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self._dump_meta()

    def __enter__(self) -> Experiment:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def list_versions(project: str, root: str | Path = "experiments") -> pd.DataFrame:
        """Таблица версий проекта: ``version, created, path`` + метрики из ``meta.json``."""
        rows = []
        for p in sorted((Path(root) / project).glob("v_*")):
            if not (m := re.fullmatch(r"v_(\d+)", p.name)) or not (p / "meta.json").exists():
                continue
            with open(p / "meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            rows.append({"version": int(m.group(1)), "created": meta.get("created"),
                         "path": str(p), **meta.get("metrics", {})})
        if not rows:
            return pd.DataFrame(columns=["version", "created", "path"])
        return pd.DataFrame(rows).sort_values("version", ignore_index=True)

    # --- служебное ---------------------------------------------------------------------
    def _dump_meta(self) -> None:
        with open(self.path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(self.meta), f, ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        return f"Experiment({self.project!r}, v_{self.version}, path={str(self.path)!r})"

    # --- логирование -------------------------------------------------------------------
    def log_params(self, params: dict[str, Any], name: str = "params") -> None:
        """Параметры (модели, отбора, сплита) → meta.json и ClearML Configuration."""
        self.meta["params"][name] = params
        self._dump_meta()
        cml.connect_params(dict(_jsonable(params)), name=name)

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        """Скалярные метрики → meta.json и ClearML."""
        clean = {f"{prefix}{k}": float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float))}
        self.meta["metrics"].update(clean)
        self._dump_meta()
        cml.report_metrics(clean, title=prefix.rstrip("_") or "metrics")

    def save_features(self, features: list[str], *, cat_features: list[str] | None = None,
                      target: str | None = None, **extra: Any) -> Path:
        """``features.json``: признаки, категориальные, таргет и произвольные поля."""
        cat = list(cat_features or [])
        payload = {"features": list(features), "cat_features": cat,
                   "num_features": [f for f in features if f not in cat],
                   "target_column": target, **extra}
        path = self.logs_path / "features.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        cml.upload_artifact("features", payload)
        return path

    def load_features(self) -> dict[str, Any]:
        with open(self.logs_path / "features.json", encoding="utf-8") as f:
            return json.load(f)

    def save_model(self, model: Any, name: str = "model") -> Path:
        """Модель (адаптер collection_lab или любой объект) → ``models/<name>.pkl``."""
        path = self.models_path / f"{name}.pkl"
        joblib.dump(model, path)
        cml.upload_artifact(name, str(path))
        return path

    def load_model(self, name: str = "model") -> Any:
        return joblib.load(self.models_path / f"{name}.pkl")

    def save_table(self, df: pd.DataFrame, name: str, *, to_clearml: bool = True) -> Path:
        """Таблица → ``logs/<name>.csv`` (+ ClearML)."""
        path = self.logs_path / f"{name}.csv"
        write_csv(df, path)
        if to_clearml:
            cml.report_table(df.head(1000), title=name)
        return path

    def save_result(self, result: Result, name: str | None = None) -> Path:
        """Результат анализа/отбора (таблица + selected + info) и его график, если есть."""
        prefix = name or result.name
        result.save(self.logs_path, prefix=prefix)
        cml.report_table(result.table.head(1000), title=prefix)
        if result.plotter is not None:
            try:
                self.save_figure(result.plot(), prefix)
            except Exception as e:  # noqa: BLE001 — график необязателен, но ошибку показываем
                warnings.warn(f"График результата {prefix!r} не сохранён: "
                              f"{type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)
        return self.logs_path

    def save_figure(self, fig, name: str) -> Path:
        """Plotly-фигура → ``logs/<name>.html`` (+ ClearML Plots)."""
        path = self.logs_path / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        cml.report_figure(fig, title=name)
        return path

    def save_report(self, report, name: str = "report") -> Path:
        """Отчёт по модели (:func:`~collection_lab.validation.model_report`): таблицы и графики
        уходят в ClearML (Plots), локальная копия — в ``logs/<name>/`` (csv + ``report.html``).
        """
        for key in ("metrics", "calibration", "segments", "feature_psi", "dynamics"):
            table = getattr(report, key)
            if table is not None:
                has_index = not isinstance(table.index, pd.RangeIndex)
                self.save_table(table.reset_index() if has_index else table, f"{name}_{key}")
        for key, fig in report.figures.items():
            self.save_figure(fig, f"{name}_{key}")
        return report.save(self.logs_path / name)

    def save_pipeline(self, pipe, name: str = "selection") -> Path:
        """Пайплайн отбора признаков (:class:`~collection_lab.selection.SelectionPipeline`):
        сводка, общий лог, воронка и результаты каждого шага — в ClearML и в ``logs/``.
        """
        self.save_table(pipe.summary(), f"{name}_summary")
        self.save_table(pipe.log_, f"{name}_log")
        self.save_figure(pipe.plot(), f"{name}_funnel")
        for key, result in pipe.results_.items():
            self.save_result(result, f"{name}_{key}")
        return self.logs_path

    def log(self, obj, name: str | None = None) -> None:
        """Отправить объект библиотеки в эксперимент (локально и в ClearML), тип определяется сам.

        Поддерживаются: ``Result``, ``ModelReport``, ``SelectionPipeline``, ``DataFrame``
        (нужен ``name``) и plotly-фигура (нужен ``name``).
        """
        from collection_lab.selection.pipeline import SelectionPipeline
        from collection_lab.validation.report import ModelReport

        if isinstance(obj, Result):
            self.save_result(obj, name)
        elif isinstance(obj, ModelReport):
            self.save_report(obj, name or "report")
        elif isinstance(obj, SelectionPipeline):
            self.save_pipeline(obj, name or "selection")
        elif isinstance(obj, pd.DataFrame):
            self.save_table(obj, self._need_name(name, "DataFrame"))
        elif hasattr(obj, "to_plotly_json"):
            self.save_figure(obj, self._need_name(name, "график"))
        else:
            raise TypeError(f"Не знаю, как сохранить объект типа {type(obj).__name__}")

    @staticmethod
    def _need_name(name: str | None, what: str) -> str:
        if not name:
            raise ValueError(f"Для объекта типа {what} укажите name: exp.log(obj, name='...')")
        return name

    def flush(self) -> None:
        """Дождаться отправки накопленных событий в ClearML."""
        if self.task is not None:
            self.task.flush(wait_for_uploads=True)

    def plots_summary(self, check: bool = True) -> pd.DataFrame:
        """Таблица по всем отправленным в ClearML графикам и таблицам: отправлено ли, причина
        отказа, подтверждён ли сервером (см. :func:`collection_lab.tracking.clearml.plots_summary`).
        """
        return cml.plots_summary(check=check)

    def close(self, verify: bool = True, quiet: bool = False) -> None:
        """Завершить задачу ClearML (локальные файлы уже сохранены). Можно вызывать повторно.

        Перед закрытием дожидается отправки и (``verify=True``, не офлайн) сверяет с сервером,
        все ли графики дошли. Печатает итог одной строкой (``quiet=True`` — без итога);
        недошедшие графики и ошибки ClearML выводятся предупреждением.
        """
        if self.task is None:
            summary = cml.plots_summary(check=False)
            unsent = summary[~summary["sent"].astype(bool)] if len(summary) else summary
            if len(unsent) and not quiet and not self._summary_printed:
                print("[collection_lab] ClearML: графики НЕ отправлены — нет активной задачи "
                      "(создайте Experiment(..., clearml=True) или Task.init): "
                      + ", ".join(unsent["title"]))
            self._summary_printed = True
            return
        summary = cml.plots_summary(check=verify)
        missing = summary.loc[summary["on_server"] == False, "title"].tolist()  # noqa: E712
        if missing:
            warnings.warn(
                "В ClearML не дошли графики: " + ", ".join(missing) + ". Обычно сервер отклоняет "
                "слишком большие запросы. Диагностика: collection_lab.tracking.clearml.diagnose()",
                RuntimeWarning, stacklevel=2)
        if not quiet:
            print(self._plots_line(summary))
        self._summary_printed = True
        self.task.close()
        self.task = None
        if self._watch is not None:
            self._watch.stop()
            self._watch.warn()
            self._watch = None

    @staticmethod
    def _plots_line(summary: pd.DataFrame) -> str:
        total, sent = len(summary), int(summary["sent"].sum()) if len(summary) else 0
        if total == 0:
            return ("[collection_lab] ClearML: ни один график не отправлялся "
                    "(save_figure / save_result не вызывались?)")
        confirmed = int((summary["on_server"] == True).sum())  # noqa: E712
        line = f"[collection_lab] ClearML: отправлено {sent} из {total}"
        if confirmed:
            line += f", подтверждено сервером {confirmed}"
        elif sent:
            line += ", сервер не позволяет подтвердить доставку — смотрите вкладку Plots"
        skipped = summary[~summary["sent"].astype(bool)]
        if len(skipped):
            reasons = "; ".join(f"{t}: {r}" for t, r in zip(skipped["title"], skipped["reason"],
                                                            strict=True))
            line += f". НЕ отправлены: {reasons}"
        return line
