"""Единый объект результата анализа: таблица + выбранные признаки + метаданные + график."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from collection_lab.utils.io import write_csv


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy-скаляры
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _maybe_send(obj, name: str, to_clearml: bool | None) -> None:
    """Отправка в ClearML из ``.save()``: авто (активный Experiment) или по флагу."""
    if to_clearml is False:
        return
    from collection_lab.tracking.experiment import Experiment, auto_send

    if to_clearml is None and Experiment.current() is None:
        return
    auto_send(obj, name)


@dataclass
class Result:
    """Результат функции анализа или отбора.

    Attributes
    ----------
    name : str
        Название метода (``'rfe'``, ``'correlation_filter'``, ...).
    table : pd.DataFrame
        Основная таблица: лог решений по признакам или история шагов.
    selected : list[str] | None
        Отобранные признаки (для методов отбора).
    info : dict
        Параметры запуска и сводные метрики.
    """

    name: str
    table: pd.DataFrame
    selected: list[str] | None = None
    info: dict[str, Any] = field(default_factory=dict)
    plotter: Callable[..., Any] | None = field(default=None, repr=False)

    @property
    def log(self) -> pd.DataFrame:
        """Синоним ``table`` — лог решений по признакам."""
        return self.table

    @property
    def dropped(self) -> list[str]:
        """Признаки, помеченные в логе как удалённые."""
        if "action" not in self.table.columns or "feature" not in self.table.columns:
            return []
        return self.table.loc[self.table["action"] == "drop", "feature"].tolist()

    def plot(self, **kwargs):
        """Строит plotly-график результата."""
        if self.plotter is None:
            raise NotImplementedError(f"Для {self.name!r} график не предусмотрен.")
        return self.plotter(self, **kwargs)

    def save(self, directory: str | Path, prefix: str | None = None, *,
             to_clearml: bool | None = None) -> Path:
        """Сохраняет таблицу (csv), список признаков и info (json) в ``directory``.

        ``to_clearml``: ``None`` — дублировать в ClearML, если есть активный
        ``Experiment(clearml=True)``; ``True``/``False`` — принудительно.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = prefix or self.name
        write_csv(self.table, directory / f"{prefix}_table.csv")
        meta = {"name": self.name, "selected": self.selected, "info": _jsonable(self.info)}
        with open(directory / f"{prefix}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _maybe_send(self, prefix, to_clearml)
        return directory

    def __repr__(self) -> str:
        sel = "" if self.selected is None else f", selected={len(self.selected)}"
        return f"Result({self.name!r}, rows={len(self.table)}{sel})"

    def _repr_html_(self) -> str:
        head = f"<b>{self.name}</b>"
        if self.selected is not None:
            head += f" — отобрано признаков: {len(self.selected)}"
        return head + self.table.head(30)._repr_html_()
