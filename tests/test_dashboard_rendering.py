
# FILE: tests/test_dashboard_rendering.py

from __future__ import annotations

import copy

import pandas as pd
import pytest

from dashboard import (
    AdvancedVisualizationEngine,
    Normalizer,
    ReliabilityAuditor,
    SchemaAdapter,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def dashboard_payload():
    """
    Stable dashboard payload fixture.
    """

    return {
        "pipeline_status": "success",
        "dataset": {
            "rows": 1000,
            "columns": 20,
            "target": "target",
            "task_type": "classification",
            "overlap_pct": 0.0,
            "overlap_count": 0,
        },
        "issues": [
            {
                "type": "feature_drift",
                "column": "salary",
                "severity": "high",
                "description": (
                    "Drift detected."
                ),
            }
        ],
        "metrics": {
            "train": {
                "f1": 0.92,
                "accuracy": 0.91,
            },
            "holdout": {
                "f1": 0.82,
                "accuracy": 0.81,
            },
            "cv": {
                "cv_mean_f1": 0.80,
                "cv_std_f1": 0.04,
            },
            "observability_flags": [
                {
                    "flag": "Overfit Risk",
                    "severity": "medium",
                    "detail": (
                        "Generalization gap "
                        "detected."
                    ),
                }
            ],
        },
        "recommendations": [
            {
                "recommendation_id":
                "feature_drift",
                "severity":
                "high",
                "recommendation":
                "Investigate feature drift.",
                "issue_count":
                1,
                "affected_columns":
                ["salary"],
            }
        ],
        "psi_table": [
            {
                "Feature": "salary",
                "PSI Score": 0.32,
                "Drift Severity": "MEDIUM",
            },
            {
                "Feature": "experience",
                "PSI Score": 0.11,
                "Drift Severity": "LOW",
            },
        ],
        "vif_table": [
            {
                "Feature": "income",
                "VIF Score": 8.2,
            }
        ],
        "feature_importance": {
            "salary": 0.45,
            "experience": 0.21,
            "age": 0.12,
        },
        "telemetry": {
            "data_loading_seconds": 0.1,
            "split_seconds": 0.05,
            "diagnostics_seconds": 0.2,
            "training_seconds": 0.4,
            "evaluation_seconds": 0.1,
            "total_pipeline_seconds": 1.0,
        },
    }

# ============================================================================
# VISUALIZATION TESTS
# ============================================================================


def test_drift_heatmap_creation(
    dashboard_payload,
):
    """
    Drift visualization must render safely.
    """

    psi_table = dashboard_payload[
        "psi_table"
    ]

    figure = (
        AdvancedVisualizationEngine.create_drift_severity_heatmap(
            psi_table
        )
    )

    assert figure is not None


def test_drift_heatmap_empty_input():
    """
    Empty PSI tables must not crash rendering.
    """

    figure = (
        AdvancedVisualizationEngine.create_drift_severity_heatmap(
            []
        )
    )

    assert figure is None


def test_drift_heatmap_invalid_schema():
    """
    Invalid PSI schemas must fail safely.
    """

    invalid_table = [
        {
            "bad_key": 123
        }
    ]

    figure = (
        AdvancedVisualizationEngine.create_drift_severity_heatmap(
            invalid_table
        )
    )

    assert figure is None


# ============================================================================
# SCHEMA SAFETY TESTS
# ============================================================================


def test_schema_adapter_handles_none(
    dashboard_payload,
):
    """
    None payloads must remain safe.
    """

    payload = copy.deepcopy(
        dashboard_payload
    )

    payload["issues"] = None
    payload["recommendations"] = None
    payload["feature_importance"] = None

    assert (
        SchemaAdapter.get_issues(
            payload
        )
        == []
    )

    assert (
        SchemaAdapter.get_recommendations(
            payload
        )
        == []
    )

    assert (
        SchemaAdapter.get_feature_importance(
            payload
        )
        == {}
    )


# ============================================================================
# RELIABILITY SCORE TESTS
# ============================================================================


def test_deployability_score_generation(
    dashboard_payload,
):
    """
    Reliability score must generate safely.
    """

    issues = dashboard_payload[
        "issues"
    ]

    result = (
        ReliabilityAuditor.compute_deployability_score(
            issues,
            dashboard_payload,
        )
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        "overall_score"
        in result
    )


def test_deployability_score_bounds(
    dashboard_payload,
):
    """
    Reliability scores must remain bounded.
    """

    issues = dashboard_payload[
        "issues"
    ]

    result = (
        ReliabilityAuditor.compute_deployability_score(
            issues,
            dashboard_payload,
        )
    )

    score = result[
        "overall_score"
    ]

    assert (
        0
        <= score
        <= 100
    )


# ============================================================================
# METRIC TESTS
# ============================================================================


def test_primary_metric_selection():
    """
    Metric prioritization stability.
    """

    metrics = {
        "accuracy": 0.88,
        "f1": 0.81,
        "precision": 0.85,
    }

    metric = (
        ReliabilityAuditor.get_primary_metric_name(
            metrics
        )
    )

    assert metric == "f1"


def test_primary_metric_fallback():
    """
    Fallback metric selection safety.
    """

    metrics = {
        "custom_metric": 0.5
    }

    metric = (
        ReliabilityAuditor.get_primary_metric_name(
            metrics
        )
    )

    assert (
        metric
        == "custom_metric"
    )


# ============================================================================
# NORMALIZATION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HIGH", "high"),
        ("Critical", "critical"),
        ("warning", "medium"),
        ("warn", "medium"),
        (None, "medium"),
    ],
)
def test_severity_normalization(
    raw,
    expected,
):
    """
    Severity normalization stability.
    """

    normalized = (
        Normalizer.normalize_severity(
            raw
        )
    )

    assert (
        normalized
        == expected
    )


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_dashboard_payload_no_none_roots(
    dashboard_payload,
):
    """
    Prevent dangerous None root contracts.
    """

    for (
        key,
        value,
    ) in dashboard_payload.items():
        assert value is not None


def test_feature_importance_numeric(
    dashboard_payload,
):
    """
    Importance values must remain numeric.
    """

    importance = dashboard_payload[
        "feature_importance"
    ]

    for value in importance.values():
        assert isinstance(
            value,
            (
                int,
                float,
            ),
        )


def test_telemetry_numeric(
    dashboard_payload,
):
    """
    Telemetry values must remain numeric.
    """

    telemetry = dashboard_payload[
        "telemetry"
    ]

    for value in telemetry.values():
        assert isinstance(
            value,
            (
                int,
                float,
            ),
        )
