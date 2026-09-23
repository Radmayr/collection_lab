"""Прогресс-бар tqdm для задач joblib."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import joblib
from tqdm.auto import tqdm


@contextlib.contextmanager
def tqdm_joblib(total: int, desc: str = "", disable: bool = False) -> Iterator[tqdm]:
    """Контекст, в котором ``joblib.Parallel`` обновляет tqdm по завершении каждой задачи.

    Пример::

        with tqdm_joblib(total=len(tasks), desc="Расчёт"):
            Parallel(n_jobs=-1)(tasks)
    """
    bar = tqdm(total=total, desc=desc, disable=disable)

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        bar.close()
