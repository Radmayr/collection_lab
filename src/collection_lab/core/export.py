"""Выгрузка модели для инференса без библиотеки.

``model.export(directory)`` кладёт в папку:

- ``model.txt`` (LightGBM) или ``model.cbm`` (CatBoost) — модель в нативном формате;
- ``preprocessing.json`` — спецификация подготовки признаков: порядок колонок, категориальные
  признаки, их категории, обозначения пропусков;
- ``inference.py`` — автономный скрипт с ``prepare(df)`` и ``predict(df)``: не импортирует
  collection_lab, нужны только pandas, numpy и lightgbm/catboost. Код подготовки признаков
  берётся из исходников библиотеки автоматически (не набирается вручную), а тест сверяет
  предсказания скрипта с предсказаниями библиотеки;
- ``requirements.txt`` — версии пакетов, с которыми модель обучена.

Свои преобразования (feature engineering и т.п.) прописываются в ``custom_preprocess()``
внутри ``inference.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from collection_lab.data import types as _types

TEMPLATE = '''"""Автономный инференс модели @@MODEL_TYPE@@ (collection_lab @@VERSION@@).

Не зависит от collection_lab. Нужны: python, numpy, pandas, @@DEPENDENCY@@ (версии — в
requirements.txt). Файлы рядом со скриптом: preprocessing.json и @@MODEL_FILE@@.

Использование:
    import inference
    scores = inference.predict(df)   # df — таблица с колонками признаков
    X = inference.prepare(df)        # только подготовка признаков (типы, категории)

Свои преобразования (feature engineering и т.п.), которые нужно выполнить ДО стандартной
подготовки признаков, прописываются в custom_preprocess() ниже.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable  # noqa: F401 — нужен для аннотаций

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "preprocessing.json"), encoding="utf-8") as _f:
    SPEC = json.load(_f)

FEATURES = SPEC["features"]
CAT_FEATURES = SPEC["cat_features"]
CATEGORIES = SPEC["categories"]
MISSING_TOKENS = tuple(SPEC["missing_tokens"])


# --- подготовка признаков: код скопирован из collection_lab.data.types ---------------------
@@HELPERS@@

def custom_preprocess(df):
    """Ваши преобразования ДО стандартной подготовки признаков (по умолчанию — без изменений)."""
    return df


def prepare(df):
    """Таблица -> признаки в формате модели: порядок колонок, типы, категории."""
    df = custom_preprocess(df)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"В данных нет признаков: {missing}")
    out = to_numeric_frame(df[FEATURES], cat_cols=CAT_FEATURES)
    for c in CAT_FEATURES:
        values = pd.Categorical(category_strings(out[c]), categories=CATEGORIES[c])
@@CAT_STEP@@
    return out


_MODEL = None


def load_model():
    """Загружает модель один раз."""
    global _MODEL
    if _MODEL is None:
@@LOAD@@
    return _MODEL


def predict(df):
    """Предсказание: вероятность класса 1 (binary) или значение (regression)."""
    X = prepare(df)
    model = load_model()
@@PREDICT@@
'''

_CAT_STEP = {
    # LightGBM: категориальный dtype с категориями, зафиксированными на обучении
    "lgbm": "        out[c] = values",
    # CatBoost: строки, пропуск и неизвестная категория -> "NaN"
    "catboost": ('        series = pd.Series(values, index=out.index)\n'
                 '        out[c] = series.astype(object).where(series.notna(), "NaN").astype(str)'),
}

_LOAD = {
    "lgbm": ('        import lightgbm as lgb\n\n'
             '        _MODEL = lgb.Booster(model_file=os.path.join(HERE, SPEC["model_file"]))'),
    "catboost": ('        from catboost import CatBoostClassifier, CatBoostRegressor\n\n'
                 '        _MODEL = (CatBoostClassifier() if SPEC["task"] == "binary"\n'
                 '                  else CatBoostRegressor())\n'
                 '        _MODEL.load_model(os.path.join(HERE, SPEC["model_file"]))'),
}

_PREDICT = {
    "lgbm": "    return np.asarray(model.predict(X))",
    "catboost": ('    if SPEC["task"] == "binary":\n'
                 '        return model.predict_proba(X)[:, 1]\n'
                 '    return np.asarray(model.predict(X))'),
}


def _helpers_source() -> str:
    """Исходники функций подготовки признаков прямо из библиотеки — без ручного дублирования."""
    return "\n\n".join(inspect.getsource(f) for f in (
        _types.normalize_missing, _types.category_strings, _types.to_numeric_frame))


def _package_versions(*names: str) -> list[str]:
    from importlib import metadata

    out = []
    for n in names:
        try:
            out.append(f"{n}=={metadata.version(n)}")
        except metadata.PackageNotFoundError:
            out.append(n)
    return out


def render_inference_script(model_type: str, model_file: str, version: str) -> str:
    """Текст ``inference.py`` для модели указанного типа."""
    dependency = "lightgbm" if model_type == "lgbm" else "catboost"
    text = TEMPLATE
    for key, value in {
        "@@MODEL_TYPE@@": model_type, "@@VERSION@@": version, "@@DEPENDENCY@@": dependency,
        "@@MODEL_FILE@@": model_file, "@@HELPERS@@": _helpers_source(),
        "@@CAT_STEP@@": _CAT_STEP[model_type], "@@LOAD@@": _LOAD[model_type],
        "@@PREDICT@@": _PREDICT[model_type],
    }.items():
        text = text.replace(key, value)
    return text


def load_inference_module(path: str | Path):
    """Импортирует сгенерированный ``inference.py`` как модуль (для проверки паритета)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"_inference_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_model(model, directory: str | Path, *, X_check=None,
                 atol: float = 1e-6) -> dict[str, Any]:
    """Выгружает обученный адаптер в папку (см. описание модуля).

    Parameters
    ----------
    model : BaseModel
        Обученная модель (``LGBMModel`` или ``CatBoostModel``).
    directory : str | Path
        Куда писать (создаётся).
    X_check : pd.DataFrame, optional
        Данные для проверки: предсказания автономного скрипта сравниваются с предсказаниями
        библиотеки; при расхождении больше ``atol`` — ``RuntimeError``.

    Returns
    -------
    dict
        ``directory, model, preprocessing, inference, requirements`` (пути) и ``max_abs_diff``
        (если задан ``X_check``).
    """
    import collection_lab

    model._check_fitted()
    if model.name not in ("lgbm", "catboost"):
        raise ValueError(f"Выгрузка не поддерживается для модели {model.name!r}")
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    if model.name == "lgbm":
        model_file = "model.txt"
        model.estimator_.booster_.save_model(str(d / model_file),
                                             num_iteration=model.best_iteration_)
        deps = _package_versions("numpy", "pandas", "lightgbm")
    else:
        model_file = "model.cbm"
        model.estimator_.save_model(str(d / model_file))
        deps = _package_versions("numpy", "pandas", "catboost")

    spec = {
        "format_version": 1,
        "library": "collection_lab",
        "library_version": collection_lab.__version__,
        "model_type": model.name,
        "task": model.task,
        "model_file": model_file,
        "features": list(model.features_),
        "cat_features": list(model.cat_features_),
        "categories": {k: list(v) for k, v in model.preparer_.categories_.items()},
        "missing_tokens": list(_types.MISSING_TOKENS),
        "numeric_rules": ("числовые: даты/интервалы -> наносекунды (float), bool -> 0/1, строки -> "
                          "to_numeric(errors=coerce); категориальные: строки, числа без '.0', "
                          "неизвестные значения -> пропуск"),
    }
    paths = {"directory": d, "model": d / model_file, "preprocessing": d / "preprocessing.json",
             "inference": d / "inference.py", "requirements": d / "requirements.txt"}
    with open(paths["preprocessing"], "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    paths["inference"].write_text(
        render_inference_script(model.name, model_file, collection_lab.__version__),
        encoding="utf-8", newline="\n")
    paths["requirements"].write_text("\n".join(deps) + "\n", encoding="utf-8")

    result: dict[str, Any] = dict(paths)
    if X_check is not None:
        module = load_inference_module(paths["inference"])
        new = np.asarray(module.predict(X_check), dtype=float)
        old = np.asarray(model.predict(X_check), dtype=float)
        diff = float(np.max(np.abs(new - old))) if len(new) else 0.0
        result["max_abs_diff"] = diff
        if diff > atol:
            raise RuntimeError(
                f"Предсказания автономного скрипта расходятся с библиотекой: max|Δ|={diff:.3g} "
                f"> {atol:g}. Файлы выгружены в {d} — проверьте вручную.")
    return result
