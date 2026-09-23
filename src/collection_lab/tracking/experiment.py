"""Версионированный эксперимент: папки logs/models, признаки, параметры, метрики, графики.

Пример::

    exp = Experiment("RTK_model", root="/workdir", version="auto", clearml=True)
    exp.log_params(params)
    exp.save_features(features, cat_features=cats, target="target")
    exp.save_model(model)
    exp.log_metrics({"auc_test": 0.63})
    exp.save_result(rfe_result)
    exp.save_figure(fig, "gain_chart")

Структура на диске::

    <root>/<project>/v_<N>/
        models/   — модели (joblib)
        logs/     — таблицы, json, графики (html)
        meta.json — параметры, метрики, время
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from collection_lab.core.results import Result, _jsonable
from collection_lab.tracking import clearml as cml


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
        Номер версии или ``"auto"`` — следующая свободная.
    name : str, optional
        Имя задачи в ClearML (по умолчанию ``v_<N>``).
    clearml : bool
        Создать задачу ClearML и дублировать туда всё, что логируется.
    clearml_offline : bool
        Режим ClearML без сервера.
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
    ):
        self.project = project
        project_dir = Path(root) / project
        project_dir.mkdir(parents=True, exist_ok=True)
        self.version = _next_version(project_dir) if version == "auto" else int(version)
        self.path = project_dir / f"v_{self.version}"
        self.models_path = self.path / "models"
        self.logs_path = self.path / "logs"
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = {"project": project, "version": self.version,
                                     "created": datetime.now().isoformat(timespec="seconds"),
                                     "params": {}, "metrics": {}}
        self.task = None
        if clearml:
            self.task = cml.init_task(project, name or f"v_{self.version}", tags=tags,
                                      offline=clearml_offline)
        self._dump_meta()

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
        df.to_csv(path, index=False)
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
            except Exception:  # noqa: BLE001 — график необязателен
                pass
        return self.logs_path

    def save_figure(self, fig, name: str) -> Path:
        """Plotly-фигура → ``logs/<name>.html`` (+ ClearML Plots)."""
        path = self.logs_path / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        cml.report_figure(fig, title=name)
        return path

    def close(self) -> None:
        """Завершить задачу ClearML (локальные файлы уже сохранены)."""
        if self.task is not None:
            self.task.close()
            self.task = None
