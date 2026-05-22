# FILE: tests/test_data_loader.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    DatasetEmptyException,
    MissingTargetColumnException,
    UnsupportedFileTypeException,
)
from app.pipeline.data_loader import (
    DataLoader,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_dataset(tmp_path):
    """
    Stable CSV fixture.
    """

    dataframe = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "salary": [
                50000,
                60000,
                70000,
            ],
            "department": [
                "engineering",
                "sales",
                "finance",
            ],
            "target": [1, 0, 1],
        }
    )

    dataset_path = (
        tmp_path / "sample.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
    )

    return dataset_path


@pytest.fixture
def duplicate_column_dataset(
    tmp_path,
):
    """
    Dataset with duplicate columns.
    """

    dataframe = pd.DataFrame(
        np.random.rand(5, 3)
    )

    dataframe.columns = [
        "feature",
        "feature",
        "target",
    ]

    dataset_path = (
        tmp_path
        / "duplicate_columns.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
    )

    return dataset_path


# ============================================================================
# LOAD TESTS
# ============================================================================


def test_load_data_success(
    sample_dataset,
):
    """
    Dataset must load successfully.
    """

    loader = DataLoader(
        sample_dataset
    )

    dataframe = loader.load_data()

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert len(dataframe) == 3


def test_loaded_dataframe_cached(
    sample_dataset,
):
    """
    Loaded dataframe should persist.
    """

    loader = DataLoader(
        sample_dataset
    )

    dataframe = loader.load_data()

    assert (
        loader.loaded_dataframe
        is not None
    )

    assert loader.loaded_dataframe.equals(
        dataframe
    )


# ============================================================================
# FILE VALIDATION TESTS
# ============================================================================


def test_missing_file_raises():
    """
    Missing files must fail cleanly.
    """

    loader = DataLoader(
        "missing_file.csv"
    )

    with pytest.raises(Exception):
        loader.load_data()


def test_invalid_extension_raises(
    tmp_path,
):
    """
    Unsupported file types must fail.
    """

    invalid_file = (
        tmp_path / "data.xlsx"
    )

    invalid_file.write_text(
        "invalid"
    )

    loader = DataLoader(
        invalid_file
    )

    with pytest.raises(
        UnsupportedFileTypeException
    ):
        loader.load_data()


# ============================================================================
# EMPTY DATASET TESTS
# ============================================================================


def test_empty_dataset_raises(
    tmp_path,
):
    """
    Empty datasets must fail safely.
    """

    empty_file = (
        tmp_path / "empty.csv"
    )

    pd.DataFrame().to_csv(
        empty_file,
        index=False,
    )

    loader = DataLoader(
        empty_file
    )

    with pytest.raises(Exception):
        loader.load_data()


# ============================================================================
# TARGET VALIDATION TESTS
# ============================================================================


def test_target_column_validation(
    sample_dataset,
):
    """
    Valid targets must pass.
    """

    loader = DataLoader(
        sample_dataset
    )

    loader.load_data()

    loader.validate_target_column(
        "target"
    )


def test_missing_target_validation(
    sample_dataset,
):
    """
    Missing targets must fail.
    """

    loader = DataLoader(
        sample_dataset
    )

    loader.load_data()

    with pytest.raises(
        MissingTargetColumnException
    ):
        loader.validate_target_column(
            "missing_target"
        )


# ============================================================================
# NORMALIZATION TESTS
# ============================================================================


def test_column_whitespace_normalization(
    tmp_path,
):
    """
    Column names should be stripped.
    """

    dataframe = pd.DataFrame(
        {
            " age ": [1, 2],
            " salary ": [3, 4],
        }
    )

    dataset_path = (
        tmp_path / "whitespace.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
    )

    loader = DataLoader(
        dataset_path
    )

    dataframe = loader.load_data()

    assert "age" in dataframe.columns

    assert "salary" in dataframe.columns


def test_duplicate_columns_resolved(
    duplicate_column_dataset,
):
    """
    Duplicate columns must become deterministic.
    """

    loader = DataLoader(
        duplicate_column_dataset
    )

    dataframe = loader.load_data()

    assert len(
        dataframe.columns
    ) == len(
        set(dataframe.columns)
    )


# ============================================================================
# FEATURE AUDIT TESTS
# ============================================================================


def test_basic_info_schema(
    sample_dataset,
):
    """
    Metadata schema must remain stable.
    """

    loader = DataLoader(
        sample_dataset
    )

    dataframe = loader.load_data()

    info = loader.basic_info(
        dataframe
    )

    required_keys = {
        "num_rows",
        "num_columns",
        "columns",
        "dtypes",
        "memory_usage_mb",
        "duplicate_rows",
        "missing_cells",
        "feature_audit",
        "preview",
    }

    assert required_keys.issubset(
        set(info.keys())
    )


def test_feature_audit_schema(
    sample_dataset,
):
    """
    Feature audit rows must remain stable.
    """

    loader = DataLoader(
        sample_dataset
    )

    dataframe = loader.load_data()

    info = loader.basic_info(
        dataframe
    )

    audit = info[
        "feature_audit"
    ]

    assert isinstance(
        audit,
        list,
    )

    row = audit[0]

    required_keys = {
        "feature",
        "dtype",
        "missing_pct",
        "unique_values",
        "is_numeric",
        "is_categorical",
    }

    assert required_keys.issubset(
        set(row.keys())
    )


# ============================================================================
# MISSING VALUE TESTS
# ============================================================================


def test_missing_token_normalization(
    tmp_path,
):
    """
    Missing tokens should normalize to NaN.
    """

    dataframe = pd.DataFrame(
        {
            "feature": [
                "NA",
                "null",
                "?",
                "valid",
            ]
        }
    )

    dataset_path = (
        tmp_path / "missing.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
    )

    loader = DataLoader(
        dataset_path
    )

    dataframe = loader.load_data()

    assert (
        dataframe["feature"]
        .isna()
        .sum()
        >= 3
    )


# ============================================================================
# PREVIEW TESTS
# ============================================================================


def test_preview_contract(
    sample_dataset,
):
    """
    Preview output must remain dashboard-safe.
    """

    loader = DataLoader(
        sample_dataset
    )

    dataframe = loader.load_data()

    info = loader.basic_info(
        dataframe
    )

    preview = info["preview"]

    assert isinstance(
        preview,
        list,
    )

    if preview:
        assert isinstance(
            preview[0],
            dict,
        )


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_no_none_outputs(
    sample_dataset,
):
    """
    Metadata must not contain dangerous None contracts.
    """

    loader = DataLoader(
        sample_dataset
    )

    dataframe = loader.load_data()

    info = loader.basic_info(
        dataframe
    )

    for (
        key,
        value,
    ) in info.items():
        assert value is not None


def test_dataframe_preserves_shape(
    sample_dataset,
):
    """
    Normalization must not corrupt row count.
    """

    original = pd.read_csv(
        sample_dataset
    )

    loader = DataLoader(
        sample_dataset
    )

    loaded = loader.load_data()

    assert len(original) == len(
        loaded
    )
