"""Выгрузка модели: автономный inference.py должен давать те же предсказания, что библиотека."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from collection_lab.core import make_model
from collection_lab.data import category_strings
from collection_lab.tracking import Experiment


@pytest.fixture(scope="module")
def messy():
    """«Неудобные» данные: разные виды пропусков, числовые и pandas-категории, bool, даты."""
    rng = np.random.default_rng(0)
    n = 2500
    df = pd.DataFrame({
        "cat_str": rng.choice(["a", "b", "c", None, "nan", "None", ""], n),
        "cat_int": rng.choice([1, 2, 3], n),
        "cat_pd": pd.Series(rng.choice(["x", "y"], n)).astype("category"),
        "flag": rng.choice([True, False], n),
        "dt": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        "num": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
        "num_str": rng.choice(["1.5", "2", "null", ""], n),
        "extra_col": rng.normal(size=n),                      # лишняя колонка не мешает
    })
    y = ((df["num"].fillna(0) + (df["cat_str"] == "a") * 1.0 + rng.normal(size=n)) > 0.5)
    return df, y.astype(int)


def _inference_frame(df):
    """Данные «из продакшена»: неизвестные категории, int-категория стала float из-за NaN."""
    new = df.head(300).copy()
    new.loc[new.index[:5], "cat_str"] = "НОВАЯ_КАТЕГОРИЯ"
    new["cat_int"] = new["cat_int"].astype("float64")
    new.loc[new.index[5:10], "cat_int"] = np.nan
    new.loc[new.index[10:13], "cat_int"] = 99.0
    return new


CATS = ["cat_str", "cat_int", "cat_pd"]
PARAMS = {"lgbm": {"n_estimators": 40}, "catboost": {"iterations": 40}}


def _fit(name, df, y):
    if name == "catboost":
        pytest.importorskip("catboost")
    m = make_model(name, PARAMS[name], early_stopping_rounds=None)
    return m.fit(df, y, cat_features=CATS)


@pytest.mark.parametrize("name", ["lgbm", "catboost"])
def test_export_parity_with_library(tmp_path, messy, name):
    df, y = messy
    model = _fit(name, df, y)
    info = model.export(tmp_path / name, X_check=_inference_frame(df))
    assert info["max_abs_diff"] < 1e-6
    files = {p.name for p in (tmp_path / name).iterdir()}
    assert {"preprocessing.json", "inference.py", "requirements.txt"} <= files
    assert ("model.txt" if name == "lgbm" else "model.cbm") in files


@pytest.mark.parametrize("name", ["lgbm", "catboost"])
def test_inference_script_is_standalone(tmp_path, messy, name):
    """Скрипт не импортирует библиотеку и в чистом процессе даёт те же числа."""
    df, y = messy
    model = _fit(name, df, y)
    model.export(tmp_path / "exp")
    text = (tmp_path / "exp" / "inference.py").read_text(encoding="utf-8")
    assert "import collection_lab" not in text and "from collection_lab" not in text

    new = _inference_frame(df)
    new.to_pickle(tmp_path / "new.pkl")
    code = (
        "import sys, numpy as np, pandas as pd\n"
        f"sys.path.insert(0, {str(tmp_path / 'exp')!r})\n"
        "import inference\n"
        f"p = inference.predict(pd.read_pickle({str(tmp_path / 'new.pkl')!r}))\n"
        "assert 'collection_lab' not in sys.modules, 'скрипт подтянул библиотеку'\n"
        f"np.save({str(tmp_path / 'out.npy')!r}, p)\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-800:]
    assert np.allclose(np.load(tmp_path / "out.npy"), model.predict(new), atol=1e-6)


def test_custom_preprocess_hook_is_used(tmp_path, messy):
    df, y = messy
    model = _fit("lgbm", df, y)
    model.export(tmp_path / "exp")
    module_path = tmp_path / "exp" / "inference.py"
    text = module_path.read_text(encoding="utf-8")
    # пользователь дописывает свою подготовку в custom_preprocess
    old = "    return df\n\n\ndef prepare"
    new = "    df = df.copy()\n    df['num'] = 0.0\n    return df\n\n\ndef prepare"
    assert old in text
    text = text.replace(old, new, 1)
    module_path.write_text(text, encoding="utf-8")
    from collection_lab.core.export import load_inference_module

    module = load_inference_module(module_path)
    changed = module.predict(df.head(200))
    zeroed = model.predict(df.head(200).assign(num=0.0))
    assert np.allclose(changed, zeroed, atol=1e-6)


def test_numeric_categories_are_same_for_int_and_float():
    a = category_strings(pd.Series([1, 2, 3]))
    b = category_strings(pd.Series([1.0, 2.0, np.nan, 3.0]))
    assert a.tolist() == ["1", "2", "3"] and b.tolist() == ["1", "2", np.nan, "3"]


def test_predictions_do_not_depend_on_int_vs_float_category(messy):
    df, y = messy
    model = _fit("lgbm", df, y)
    as_int = df.head(200)
    as_float = as_int.assign(cat_int=as_int["cat_int"].astype("float64"))
    assert np.allclose(model.predict(as_int), model.predict(as_float))


def test_experiment_export_model(tmp_path, messy):
    df, y = messy
    model = _fit("lgbm", df, y)
    with Experiment("p", root=tmp_path) as exp:
        info = exp.export_model(model, "final", X_check=df.head(100))
    assert info["directory"] == exp.path / "export" / "final"
    assert info["max_abs_diff"] < 1e-6


def test_export_requires_fitted_model(tmp_path):
    with pytest.raises(RuntimeError, match="не обучена"):
        make_model("lgbm").export(tmp_path / "x")
