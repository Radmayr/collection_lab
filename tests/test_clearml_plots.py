"""Графики в ClearML: размер фигур, прореживание, видимость ошибок сервера."""

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from collection_lab.eda import eda_transactions, plot_distribution
from collection_lab.plotting import figure_size_mb, histogram, shrink_figure
from collection_lab.tracking import Experiment
from collection_lab.tracking import clearml as cml


def scatter(n):
    rng = np.random.default_rng(0)
    return go.Figure(go.Scatter(x=np.arange(n), y=rng.normal(size=n), mode="markers"))


class FakeLogger:
    def __init__(self):
        self.reported = []

    def report_plotly(self, title, series, iteration, figure):
        self.reported.append((title, figure))


class FakeTask:
    """Имитация задачи ClearML: сервер «получает» только графики из ``delivered``."""

    def __init__(self, delivered=None, truncated_readback=False, metrics_api_fails=False):
        self.name, self.id = "v_1", "fake"
        self.truncated_readback = truncated_readback
        self.metrics_api_fails = metrics_api_fails
        self.logger = FakeLogger()
        self.delivered = delivered
        self.closed = False
        self.active = False
        self.flushed = 0

    def get_logger(self):
        return self.logger

    def flush(self, wait_for_uploads=False):
        self.flushed += 1

    def _on_server(self):
        return self.delivered if self.delivered is not None else [t for t, _ in
                                                                  self.logger.reported]

    def send(self, request):
        """Список метрик с графиками — не зависит от размера графиков."""
        if self.metrics_api_fails:
            raise RuntimeError("старый сервер: event_type не поддерживается")
        return SimpleNamespace(response=SimpleNamespace(
            metrics=[{"task": self.id, "metrics": list(self._on_server())}]))

    def get_reported_plots(self, max_iterations=None):
        """Чтение самих графиков: у сервера ответ ограничен по размеру (крупные не попадают)."""
        titles = list(self._on_server())
        if self.truncated_readback:
            titles = titles[:1]
        return [{"metric": t} for t in titles]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_clearml(monkeypatch):
    holder = {}

    def install(delivered=None, **kwargs):
        task = FakeTask(delivered, **kwargs)
        holder["task"] = task
        def init(*a, **k):
            task.active = True
            return task

        monkeypatch.setattr(cml, "current_task",
                            lambda: task if task.active and not task.closed else None)
        monkeypatch.setattr(cml, "init_task", init)
        monkeypatch.setattr(cml, "is_offline", lambda: False)
        monkeypatch.setattr(cml.time, "sleep", lambda s: None)
        return task

    return install


def test_histogram_size_does_not_depend_on_rows():
    small = figure_size_mb(go.Figure(histogram(np.random.default_rng(0).normal(size=10_000))))
    huge = figure_size_mb(go.Figure(histogram(np.random.default_rng(0).normal(size=2_000_000))))
    assert huge < 0.05 and abs(huge - small) < 0.01
    h = histogram(np.array([1.0, 2.0, 2.0, np.nan, 3.0]), bins=2)
    assert h.y.sum() == 4                                        # NaN отброшен, значения сохранены


def test_eda_figures_are_small_on_big_data():
    s = pd.Series(np.random.default_rng(0).lognormal(size=500_000), name="money")
    assert figure_size_mb(plot_distribution(s)) < 0.05
    rng = np.random.default_rng(1)
    n = 50_000
    tx = pd.DataFrame({
        "contract_number": rng.integers(0, 5000, n), "transaction_amt": rng.gamma(1, 5000, n),
        "rtk_balance": rng.gamma(2, 50_000, n), "rtk_send_date": pd.Timestamp("2024-01-01"),
        "tx_days": rng.integers(0, 300, n),
        "transaction_dt": pd.Timestamp("2024-02-01") + pd.to_timedelta(rng.integers(0, 200, n),
                                                                        unit="D")})
    figs = eda_transactions(tx, tx_dt_col="transaction_dt", verbose=False)["figures"]
    assert max(figure_size_mb(f) for f in figs.values()) < 1.0


def test_shrink_figure_keeps_original():
    big = scatter(100_000)
    small = shrink_figure(big, 1_000)
    assert len(small.data[0].x) == 1_000 and len(big.data[0].x) == 100_000
    assert len(small.data[0].y) == 1_000
    aggregated = go.Figure(go.Bar(x=[1, 2, 3], y=[1, 2, 3]))
    assert list(shrink_figure(aggregated, 10).data[0].x) == [1, 2, 3]  # короткие не трогаем


def test_report_figure_shrinks_large_figure_with_warning(fake_clearml):
    task = fake_clearml()
    task.active = True
    with pytest.warns(RuntimeWarning, match="уменьшен"):
        assert cml.report_figure(scatter(300_000), "big", max_mb=1.0)
    (_, sent), = task.logger.reported
    assert figure_size_mb(sent) <= 1.0 and len(sent.data[0].x) < 300_000
    assert task.flushed >= 1                                     # крупный график сразу отправлен


def test_report_figure_skips_when_cannot_fit(fake_clearml):
    task = fake_clearml()
    task.active = True
    fig = go.Figure(go.Heatmap(z=np.random.default_rng(0).normal(size=(1500, 1500))))
    with pytest.warns(RuntimeWarning, match="не отправлен"):
        assert cml.report_figure(fig, "heat", max_mb=0.5) is False
    assert task.logger.reported == []


def test_max_plot_mb_from_env(monkeypatch):
    assert cml.max_plot_mb() == cml.MAX_PLOT_MB
    monkeypatch.setenv("COLLECTION_LAB_CLEARML_MAX_MB", "2.5")
    assert cml.max_plot_mb() == 2.5
    monkeypatch.setenv("COLLECTION_LAB_CLEARML_MAX_MB", "мусор")
    assert cml.max_plot_mb() == cml.MAX_PLOT_MB


def test_error_watch_collects_server_errors_and_warns():
    watch = cml.ErrorWatch().start()
    try:
        logging.getLogger("clearml.session").error(
            "events.add_batch request exceeds limit 25616875 > 15728640 bytes")
        logging.getLogger("clearml.session").info("не ошибка")
    finally:
        watch.stop()
    assert watch.messages == ["events.add_batch request exceeds limit 25616875 > 15728640 bytes"]
    with pytest.warns(RuntimeWarning, match="слишком большой запрос"):
        watch.warn()
    logging.getLogger("clearml.session").error("после stop не собирается")
    assert len(watch.messages) == 1


def test_experiment_warns_about_plots_that_did_not_arrive(tmp_path, fake_clearml):
    fake_clearml(delivered=["a"])                                # сервер принял только «a»
    exp = Experiment("p", root=tmp_path, clearml=True)
    exp.save_figure(go.Figure(go.Scatter(y=[1, 2])), "a")
    exp.save_figure(go.Figure(go.Scatter(y=[1, 2])), "b")
    with pytest.warns(RuntimeWarning, match="не дошли графики: b"):
        exp.close()


def test_experiment_silent_when_all_plots_arrived(tmp_path, fake_clearml):
    task = fake_clearml()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with Experiment("p", root=tmp_path, clearml=True) as exp:
            exp.save_figure(go.Figure(go.Scatter(y=[1, 2])), "a")
    assert task.closed


def test_save_result_warns_when_plot_fails(tmp_path):
    from collection_lab.core.results import Result

    def broken(result):
        raise ValueError("нельзя построить")

    res = Result("demo", pd.DataFrame({"a": [1]}), plotter=broken)
    with pytest.warns(RuntimeWarning, match="не сохранён: ValueError"):
        Experiment("p", root=tmp_path).save_result(res)


def test_verify_plots_without_task_or_offline(monkeypatch):
    monkeypatch.setattr(cml, "current_task", lambda: None)
    assert cml.verify_plots(["x"]) == []


def test_verify_uses_metrics_listing_not_truncated_plot_readback(tmp_path, fake_clearml):
    """Ложная тревога с ML Core: get_reported_plots отдавал только часть графиков."""
    fake_clearml(truncated_readback=True)                       # readback видит только первый
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)          # любое предупреждение = провал
        with Experiment("p", root=tmp_path, clearml=True) as exp:
            for name in ("a", "b", "c"):
                exp.save_figure(go.Figure(go.Scatter(y=[1, 2])), name)


def test_verify_silent_when_server_cannot_be_checked(tmp_path, fake_clearml):
    fake_clearml(delivered=[], metrics_api_fails=True)          # проверить нельзя → не тревожим
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with Experiment("p", root=tmp_path, clearml=True) as exp:
            exp.save_figure(go.Figure(go.Scatter(y=[1, 2])), "a")


def test_paused_error_watch_ignores_errors():
    watch = cml.ErrorWatch().start()
    try:
        watch.paused = True
        logging.getLogger("clearml.session").error("намеренная ошибка диагностики")
        watch.paused = False
        logging.getLogger("clearml.session").error("настоящая ошибка")
    finally:
        watch.stop()
    assert watch.messages == ["настоящая ошибка"]
