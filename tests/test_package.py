import importlib

import pytest

import collection_lab
from collection_lab.config import LGBM_PARAMS, merge_params

SUBPACKAGES = [
    "core", "data", "eda", "metrics", "selection",
    "modeling", "validation", "plotting", "tracking",
]


def test_version():
    assert collection_lab.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    importlib.import_module(f"collection_lab.{name}")


def test_merge_params_does_not_mutate_defaults():
    before = dict(LGBM_PARAMS)
    merged = merge_params(LGBM_PARAMS, {"learning_rate": 0.1, "extra": 1})
    assert merged["learning_rate"] == 0.1
    assert merged["extra"] == 1
    assert LGBM_PARAMS == before


def test_binary_fixture(binary_df):
    assert binary_df["target"].isin([0, 1]).all()
    assert 0.05 < binary_df["target"].mean() < 0.6
