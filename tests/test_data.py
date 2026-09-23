import numpy as np
import pandas as pd
import pytest

from collection_lab.data import (
    DataSplit,
    cast_types,
    detect_categorical,
    random_split,
    split_feature_types,
    time_split,
    to_numeric_frame,
)


def test_detect_categorical_handles_string_missing_tokens():
    df = pd.DataFrame({
        "num_as_str": ["1", "2.5", "nan", None, ""],
        "cat": ["a", "b", "None", None, "c"],
        "num": [1.0, 2.0, np.nan, 3.0, 4.0],
        "flag": [True, False, True, True, False],
    })
    assert detect_categorical(df) == ["cat"]
    num, cat = split_feature_types(df)
    assert cat == ["cat"]
    assert num == ["num_as_str", "num", "flag"]


def test_cast_types_keeps_nan_in_categories():
    df = pd.DataFrame({"n": ["1", "null", "3"], "c": ["a", "nan", "b"], "t": [0.0, 1.0, 0.0]})
    out = cast_types(df, ["n"], ["c"], target_col="t")
    assert out["n"].dtype == "float64" and np.isnan(out["n"][1])
    assert out["c"].isna().tolist() == [False, True, False]
    assert out["t"].dtype == int


def test_to_numeric_frame_dates_and_bools():
    df = pd.DataFrame({
        "d": pd.to_datetime(["2024-01-01", None]),
        "b": [True, False],
        "c": ["x", "y"],
    })
    out = to_numeric_frame(df, cat_cols=["c"])
    assert out["d"].dtype == "float64" and np.isnan(out["d"][1])
    assert out["b"].tolist() == [1.0, 0.0]
    assert out["c"].tolist() == ["x", "y"]


def test_random_split_disjoint_and_stratified(binary_df):
    split = random_split(binary_df, "target", val_size=0.2, test_size=0.2)
    sizes = split.summary()["n"]
    assert sizes.sum() == len(binary_df)
    rates = split.summary()["target_rate"]
    assert rates.max() - rates.min() < 0.02


def test_time_split_oot(binary_df):
    split = time_split(binary_df, "target", "report_date", oot_from="2024-03-01")
    assert split.test["report_date"].min() >= pd.Timestamp("2024-03-01")
    assert split.train["report_date"].max() < pd.Timestamp("2024-03-01")
    assert len(split.train) + len(split.val) + len(split.test) == len(binary_df)


def test_time_split_val_by_time(binary_df):
    split = time_split(binary_df, "target", "report_date", oot_from="2024-03-01",
                       val_mode="time")
    assert split.train["report_date"].max() <= split.val["report_date"].min()


def test_datasplit_rejects_overlap(binary_df):
    part = binary_df.iloc[:100]
    with pytest.raises(ValueError, match="пересекаются"):
        DataSplit(train=part, val=part.iloc[:10], test=None, target="target")


def test_datasplit_helpers(binary_df):
    split = random_split(binary_df, "target", test_size=0.2)
    X, y = split.xy("val", ["x1", "x2"])
    assert list(X.columns) == ["x1", "x2"] and len(X) == len(y)
    assert set(split.labeled()["sample"]) == {"train", "val", "test"}
