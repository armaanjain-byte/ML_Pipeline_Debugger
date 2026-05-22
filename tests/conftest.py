# FILE: tests/conftest.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# GLOBAL RANDOM SEED
# ============================================================================

GLOBAL_RANDOM_SEED = 42

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """
    Global pytest initialization.

    Stabilizes:
    - deterministic test behavior
    - reproducible numerical tests
    - random-state consistency
    """

    np.random.seed(
        GLOBAL_RANDOM_SEED
    )

# ============================================================================
# COMMON DATASET FIXTURES
# ============================================================================


@pytest.fixture
def classification_dataframe():
    """
    Shared classification dataframe fixture.
    """

    rng = np.random.default_rng(
        GLOBAL_RANDOM_SEED
    )

    rows = 500

    dataframe = pd.DataFrame(
        {
            "age": rng.normal(
                35,
                8,
                rows,
            ),
            "salary": rng.normal(
                65000,
                15000,
                rows,
            ),
            "experience": rng.normal(
                7,
                3,
                rows,
            ),
            "department": rng.choice(
                [
                    "engineering",
                    "sales",
                    "finance",
                    "hr",
                ],
                rows,
            ),
            "target": rng.integers(
                0,
                2,
                rows,
            ),
        }
    )

    return dataframe


@pytest.fixture
def regression_dataframe():
    """
    Shared regression dataframe fixture.
    """

    rng = np.random.default_rng(
        GLOBAL_RANDOM_SEED
    )

    rows = 500

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
                4,
                rows,
            ),
            "target": rng.normal(
                100,
                25,
                rows,
            ),
        }
    )

    return dataframe

# ============================================================================
# FILE FIXTURES
# ============================================================================


@pytest.fixture
def classification_csv(
    tmp_path,
    classification_dataframe,
):
    """
    Shared classification CSV fixture.
    """

    path = (
        tmp_path
        / "classification.csv"
    )

    classification_dataframe.to_csv(
        path,
        index=False,
    )

    return path


@pytest.fixture
def regression_csv(
    tmp_path,
    regression_dataframe,
):
    """
    Shared regression CSV fixture.
    """

    path = (
        tmp_path
        / "regression.csv"
    )

    regression_dataframe.to_csv(
        path,
        index=False,
    )

    return path

# ============================================================================
# STABILITY HELPERS
# ============================================================================


@pytest.fixture
def empty_safe_payload():
    """
    Dashboard-safe empty payload.

    Prevents repetitive schema definitions
    across tests.
    """

    return {
        "pipeline_status": "success",
        "dataset": {},
        "feature_audit": [],
        "telemetry": {},
        "issues": [],
        "metrics": {},
        "recommendations": [],
        "psi_table": [],
        "vif_table": [],
        "feature_importance": {},
        "critical_issues": 0,
    }
