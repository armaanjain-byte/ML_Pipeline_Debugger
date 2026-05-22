# FILE: tests/test_feature_utils.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.utils.feature_utils import (
    AdvancedDiagnostics,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def numeric_dataframe():
    """
    Stable numeric test dataframe.
    """

    rng = np.random.default_rng(42)

    rows = 300

    dataframe = pd.DataFrame(
        {
            "feature_a": rng.normal(
                0,
                1,
                rows,
            ),
            "feature_b": rng.normal(
                5,
                2,
                rows,
            ),
            "feature_c": rng.normal(
                10,
                5,
                rows,
            ),
        }
    )

    return dataframe


@pytest.fixture
def train_test_pair():
    """
    Stable train/test pair for drift analysis.
    """

    rng = np.random.default_rng(42)

    train = pd.DataFrame(
        {
            "age": rng.normal(
                30,
                5,
                400,
            ),
            "salary": rng.normal(
                50000,
                12000,
                400,
            ),
            "score": rng.normal(
                0.5,
                0.1,
                400,
            ),
        }
    )

    test = pd.DataFrame(
        {
            "age": rng.normal(
                35,
                8,
                200,
            ),
            "salary": rng.normal(
                70000,
                25000,
                200,
            ),
            "score": rng.normal(
                0.55,
                0.15,
                200,
            ),
        }
    )

    return train, test


# ============================================================================
# PSI TESTS
# ============================================================================


def test_compute_psi_returns_float():
    """
    PSI must always return float.
    """

    expected = pd.Series(
        np.random.normal(
            0,
            1,
            1000,
        )
    )

    actual = pd.Series(
        np.random.normal(
            0.5,
            1.2,
            1000,
        )
    )

    psi = (
        AdvancedDiagnostics.compute_psi(
            expected,
            actual,
        )
    )

    assert isinstance(
        psi,
        float,
    )


def test_compute_psi_non_negative():
    """
    PSI should not be negative.
    """

    expected = pd.Series(
        np.random.normal(
            0,
            1,
            1000,
        )
    )

    actual = pd.Series(
        np.random.normal(
            0,
            1,
            1000,
        )
    )

    psi = (
        AdvancedDiagnostics.compute_psi(
            expected,
            actual,
        )
    )

    assert psi >= 0


def test_compute_psi_empty_series():
    """
    Empty PSI inputs must not crash.
    """

    psi = (
        AdvancedDiagnostics.compute_psi(
            pd.Series([]),
            pd.Series([]),
        )
    )

    assert psi == 0.0


# ============================================================================
# PSI TABLE TESTS
# ============================================================================


def test_compute_all_psi_schema(
    train_test_pair,
):
    """
    Protect PSI table schema.
    """

    train, test = train_test_pair

    results = (
        AdvancedDiagnostics.compute_all_psi(
            train,
            test,
        )
    )

    assert isinstance(
        results,
        list,
    )

    if results:
        row = results[0]

        required_keys = {
            "Feature",
            "PSI Score",
            "Drift Severity",
        }

        assert required_keys.issubset(
            set(row.keys())
        )


def test_psi_severity_mapping():
    """
    Protect PSI severity semantics.
    """

    assert (
        AdvancedDiagnostics._resolve_psi_severity(
            0.05
        )
        == "LOW"
    )

    assert (
        AdvancedDiagnostics._resolve_psi_severity(
            0.30
        )
        == "MEDIUM"
    )

    assert (
        AdvancedDiagnostics._resolve_psi_severity(
            0.70
        )
        == "HIGH"
    )


# ============================================================================
# OVERLAP TESTS
# ============================================================================


def test_row_overlap_detection():
    """
    Overlap detection must return valid types.
    """

    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )

    overlap_count, overlap_pct = (
        AdvancedDiagnostics.calculate_row_overlap(
            dataframe,
            dataframe.copy(),
        )
    )

    assert isinstance(
        overlap_count,
        int,
    )

    assert isinstance(
        overlap_pct,
        float,
    )

    assert overlap_count > 0

    assert overlap_pct > 0


# ============================================================================
# VARIANCE TESTS
# ============================================================================


def test_variance_ratio_schema(
    train_test_pair,
):
    """
    Protect variance issue schema.
    """

    train, test = train_test_pair

    issues = (
        AdvancedDiagnostics.compute_train_test_variance_ratio(
            train,
            test,
        )
    )

    assert isinstance(
        issues,
        list,
    )

    if issues:
        issue = issues[0]

        required_keys = {
            "type",
            "column",
            "severity",
            "description",
        }

        assert required_keys.issubset(
            set(issue.keys())
        )


# ============================================================================
# MISSINGNESS TESTS
# ============================================================================


def test_informative_missingness_schema():
    """
    Missingness analysis must remain stable.
    """

    rng = np.random.default_rng(42)

    dataframe = pd.DataFrame(
        {
            "feature": rng.normal(
                0,
                1,
                200,
            ),
            "target": rng.integers(
                0,
                2,
                200,
            ),
        }
    )

    dataframe.loc[
        dataframe.index[:50],
        "feature",
    ] = np.nan

    issues = (
        AdvancedDiagnostics.compute_informative_missingness(
            dataframe,
            target_column="target",
            task_type="classification",
        )
    )

    assert isinstance(
        issues,
        list,
    )

    if issues:
        issue = issues[0]

        required_keys = {
            "type",
            "column",
            "severity",
            "description",
        }

        assert required_keys.issubset(
            set(issue.keys())
        )


# ============================================================================
# VIF TESTS
# ============================================================================


def test_vif_schema(
    numeric_dataframe,
):
    """
    VIF output must remain stable.
    """

    vif_scores = (
        AdvancedDiagnostics.compute_vif(
            numeric_dataframe
        )
    )

    assert isinstance(
        vif_scores,
        dict,
    )

    for (
        feature,
        score,
    ) in vif_scores.items():
        assert isinstance(
            feature,
            str,
        )

        assert isinstance(
            score,
            float,
        )


def test_vif_handles_constant_columns():
    """
    Constant columns must not crash VIF.
    """

    dataframe = pd.DataFrame(
        {
            "constant": [1] * 100,
            "feature": np.random.normal(
                0,
                1,
                100,
            ),
        }
    )

    vif_scores = (
        AdvancedDiagnostics.compute_vif(
            dataframe
        )
    )

    assert isinstance(
        vif_scores,
        dict,
    )


def test_vif_handles_missing_values():
    """
    Missing values must not crash VIF.
    """

    dataframe = pd.DataFrame(
        {
            "a": np.random.normal(
                0,
                1,
                100,
            ),
            "b": np.random.normal(
                0,
                1,
                100,
            ),
        }
    )

    dataframe.loc[
        dataframe.index[:20],
        "a",
    ] = np.nan

    vif_scores = (
        AdvancedDiagnostics.compute_vif(
            dataframe
        )
    )

    assert isinstance(
        vif_scores,
        dict,
    )


# ============================================================================
# CARDINALITY TESTS
# ============================================================================


def test_high_cardinality_detection():
    """
    Cardinality detection schema safety.
    """

    dataframe = pd.DataFrame(
        {
            "high_card": [
                f"user_{i}"
                for i in range(200)
            ]
        }
    )

    issues = (
        AdvancedDiagnostics.detect_high_cardinality_features(
            dataframe,
            threshold=50,
        )
    )

    assert isinstance(
        issues,
        list,
    )

    if issues:
        issue = issues[0]

        required_keys = {
            "type",
            "column",
            "severity",
            "description",
        }

        assert required_keys.issubset(
            set(issue.keys())
        )


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_no_nan_outputs_in_psi(
    train_test_pair,
):
    """
    PSI outputs must not contain NaN.
    """

    train, test = train_test_pair

    results = (
        AdvancedDiagnostics.compute_all_psi(
            train,
            test,
        )
    )

    for row in results:
        psi_score = row[
            "PSI Score"
        ]

        assert not np.isnan(
            psi_score
        )


def test_no_infinite_vif_outputs(
    numeric_dataframe,
):
    """
    VIF outputs must not contain infinities.
    """

    vif_scores = (
        AdvancedDiagnostics.compute_vif(
            numeric_dataframe
        )
    )

    for score in vif_scores.values():
        assert not np.isinf(
            score
        )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_single_column_vif():
    """
    Single-column dataframe must not crash.
    """

    dataframe = pd.DataFrame(
        {
            "feature": np.random.normal(
                0,
                1,
                100,
            )
        }
    )

    vif_scores = (
        AdvancedDiagnostics.compute_vif(
            dataframe
        )
    )

    assert isinstance(
        vif_scores,
        dict,
    )


def test_non_numeric_psi_inputs():
    """
    PSI should safely handle non-numeric data.
    """

    expected = pd.Series(
        ["a", "b", "c"]
    )

    actual = pd.Series(
        ["d", "e", "f"]
    )

    psi = (
        AdvancedDiagnostics.compute_psi(
            expected,
            actual,
        )
    )

    assert isinstance(
        psi,
        float,
    )
