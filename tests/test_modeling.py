import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from collection_lab.data import random_split, time_split
from collection_lab.modeling import compare_models, sample_size_curve, train_model, tune_hyperparams
from collection_lab.selection import rfe
from collection_lab.tracking import Experiment
from collection_lab.tracking import clearml as cml

FAST = {"n_estimators": 150, "learning_rate": 0.1}
FEATS = ["x1", "x2", "cat", "noise"]


@pytest.fixture(scope="module")
def split(binary_df):
    return time_split(binary_df, "target", "report_date", oot_from="2024-03-01")


def test_train_model(split):
    m = train_model(split, FEATS, params=FAST, auto_scale_pos_weight=True)
    assert set(m.scores_) == {"train", "val", "test"}
    assert m.scores_["val"] > 0.7
    assert "scale_pos_weight" in m.params
    final = train_model(split, FEATS, params=FAST, refit_on_train_val=True)
    assert final.early_stopping_rounds is None
    assert final.n_iterations_ <= 150


def test_train_model_catboost(split):
    pytest.importorskip("catboost")
    m = train_model(split, FEATS, model="catboost", params={"iterations": 150,
                                                            "learning_rate": 0.1})
    assert m.scores_["test"] > 0.65


def test_compare_models(binary_df):
    pytest.importorskip("catboost")
    summary, oof = compare_models(
        binary_df[FEATS], binary_df["target"], ["lgbm", "catboost"], n_splits=2,
        params={"lgbm": FAST, "catboost": {"iterations": 150, "learning_rate": 0.1}})
    assert list(summary.columns[:3]) == ["model", "mean_auc", "std_auc"]
    assert {"lgbm_pred", "catboost_pred", "fold", "y_true"} <= set(oof.columns)
    assert oof[["lgbm_pred", "catboost_pred"]].notna().all().all()


def test_tune_hyperparams_selects_by_val(split):
    pytest.importorskip("optuna")
    res = tune_hyperparams(split, FEATS, n_trials=4, fixed_params={"n_estimators": 100},
                           verbose=False)
    t = res.table
    assert len(t) == 4 and {"auc_train", "auc_val", "auc_test"} <= set(t.columns)
    assert res.info["best_val"] == pytest.approx(t["auc_val"].max())
    assert "n_estimators" in res.info["best_params"]
    assert isinstance(res.plot(), go.Figure)


def test_sample_size_curve(split):
    res = sample_size_curve(split, FEATS, [200, 800, 10**9], n_repeats=2, params=FAST)
    assert sorted(res.table["sample_size"].unique())[-1] == len(split.train)
    assert isinstance(res.plot(), go.Figure)


def test_experiment_local(tmp_path, binary_df, split):
    exp = Experiment("demo", root=tmp_path)
    assert exp.version == 1 and exp.path.name == "v_1"
    assert Experiment("demo", root=tmp_path).version == 2
    model = train_model(split, FEATS, params=FAST)
    exp.log_params(FAST, name="model")
    exp.log_metrics(model.scores_, prefix="auc_")
    exp.save_features(FEATS, cat_features=["cat"], target="target")
    exp.save_model(model)
    res = rfe(binary_df, binary_df["target"], ["x1", "noise"], n_splits=2, params=FAST,
              verbose=False)
    exp.save_result(res)
    meta = json.loads((exp.path / "meta.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["auc_val"] == pytest.approx(model.scores_["val"])
    assert exp.load_features()["num_features"] == ["x1", "x2", "noise"]
    loaded = exp.load_model()
    assert np.allclose(loaded.predict(split.test[FEATS]), model.predict(split.test[FEATS]))
    assert (exp.logs_path / "rfe_table.csv").exists()
    assert (exp.logs_path / "rfe.html").exists()


def test_clearml_helpers_noop_without_task():
    # без активной задачи ClearML логирование молча пропускается
    assert cml.report_metrics({"a": 1.0}) is False
    assert cml.report_figure(go.Figure(), "x") is False


def test_random_split_used_in_train(binary_df):
    s = random_split(binary_df, "target", test_size=0.2)
    m = train_model(s, ["x1"], params=FAST)
    assert m.features_ == ["x1"]


class _FakeTask:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def test_experiment_versions_are_unique_and_listed(tmp_path):
    e1, e2, e3 = (Experiment("p", root=tmp_path) for _ in range(3))
    assert [e.version for e in (e1, e2, e3)] == [1, 2, 3]
    e2.log_metrics({"auc": 0.6})
    table = Experiment.list_versions("p", root=tmp_path)
    assert list(table["version"]) == [1, 2, 3] and table.loc[1, "auc"] == 0.6
    assert Experiment.list_versions("nothing", root=tmp_path).empty


def test_experiment_explicit_version_protected_from_overwrite(tmp_path):
    Experiment("p", root=tmp_path).log_metrics({"auc": 0.5})           # v_1 с данными
    with pytest.raises(FileExistsError, match="v_1 уже существует"):
        Experiment("p", root=tmp_path, version=1)
    again = Experiment("p", root=tmp_path, version=1, overwrite=True)  # осознанная перезапись
    assert again.version == 1
    fresh = Experiment("p", root=tmp_path, version=7)                  # новой явной версии можно
    assert fresh.path.name == "v_7"


def test_experiment_closes_previous_clearml_task(tmp_path, monkeypatch):
    """Повторный запуск в одном ядре: старая задача закрывается до Task.init новой."""
    calls, state = [], {"task": _FakeTask("v_1")}
    monkeypatch.setattr(cml, "current_task", lambda: state["task"])

    def fake_init(project, name, **kw):
        calls.append(("init", name, state["task"].closed))
        state["task"] = _FakeTask(name)
        return state["task"]

    monkeypatch.setattr(cml, "init_task", fake_init)
    old = state["task"]
    exp = Experiment("p", root=tmp_path, clearml=True)
    assert old.closed and calls == [("init", "v_1", True)]  # init произошёл после close
    exp.close()
    exp.close()  # повторный close безопасен
    assert state["task"].closed


def test_experiment_failed_clearml_init_leaves_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(cml, "current_task", lambda: None)

    def broken(*a, **kw):
        raise RuntimeError("сервер недоступен")

    monkeypatch.setattr(cml, "init_task", broken)
    with pytest.raises(RuntimeError):
        Experiment("p", root=tmp_path, clearml=True)
    assert not (tmp_path / "p" / "v_1").exists()   # пустая папка версии не остаётся
    assert Experiment("p", root=tmp_path).version == 1  # и номер не «сгорел»


def test_experiment_context_manager(tmp_path, monkeypatch):
    task = _FakeTask("v_1")
    monkeypatch.setattr(cml, "current_task", lambda: None)
    monkeypatch.setattr(cml, "init_task", lambda *a, **k: task)
    with Experiment("p", root=tmp_path, clearml=True) as exp:
        exp.log_metrics({"x": 1})
    assert task.closed and exp.task is None


def test_csv_files_are_excel_friendly(tmp_path, binary_df):
    """CSV с русским текстом открывается в Excel: UTF-8 с BOM, pandas читает без искажений."""
    from collection_lab.selection import CorrelationFilter, QualityFilter, SelectionPipeline

    pipe = SelectionPipeline([QualityFilter(), CorrelationFilter(n_splits=2, verbose=False)]).fit(
        binary_df, "target", ["x1", "x2", "x1_copy", "const", "all_nan"], verbose=False)
    pipe.save(tmp_path)
    raw = (tmp_path / "selection_log.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                               # BOM есть
    assert "Пропущено".encode() in raw                                   # текст в UTF-8
    assert "Пропущено" in raw.decode("utf-8-sig")
    log = pd.read_csv(tmp_path / "selection_log.csv")                    # pandas: без ﻿ в заголовках
    assert list(log.columns) == ["step", "feature", "action", "reason"]
    assert log["reason"].str.contains("Пропущено|Стандартное").any()
    for f in tmp_path.glob("*_table.csv"):
        assert f.read_bytes().startswith(b"\xef\xbb\xbf")
