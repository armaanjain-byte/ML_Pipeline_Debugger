# FILE: tests/test_dashboard_schema.py

from __future__ import annotations

import copy

import pytest

from dashboard import (
    Normalizer,
    ReliabilityAuditor,
    SchemaAdapter,
)

# ============================================================================
# FIXTURE
# ============================================================================


@pytest.fixture
def valid_dashboard_payload():
    """
    Stable dashboard-safe payload.
    """

    return {
        "pipeline_status": "success",
        "dataset": {
            "rows": 1000,
            "columns": 25,
            "target": "target",
            "task_type": "classification",
            "overlap_pct": 0.0,
            "overlap_count": 0,
        },
        "feature_audit": [
            {
                "feature": "age",
                "dtype": "int64",
                "missing_pct": 0.0,
                "unique_values": 50,
                "is_numeric": True,
                "is_categorical": False,
            }
        ],
        "telemetry": {
            "data_loading_seconds": 0.1,
            "split_seconds": 0.05,
            "diagnostics_seconds": 0.2,
            "training_seconds": 0.4,
            "evaluation_seconds": 0.1,
            "total_pipeline_seconds": 1.0,
        },
        "issues": [
            {
                "type": "feature_drift",
                "column": "salary",
                "severity": "high",
                "description": "Drift detected.",
            }
        ],
        "metrics": {
            "train": {
                "f1": 0.92,
                "accuracy": 0.91,
            },
            "holdout": {
                "f1": 0.81,
                "accuracy": 0.80,
            },
            "cv": {
                "cv_mean_f1": 0.82,
                "cv_std_f1": 0.04,
            },
            "observability_flags": [
                {
                    "flag": "Overfit Risk",
                    "severity": "high",
                    "detail": (
                        "Train performance exceeds "
                        "holdout."
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
            }
        ],
        "vif_table": [
            {
                "Feature": "income",
                "VIF Score": 8.2,
            }
        ],
        "feature_importance": {
            "salary": 0.42,
            "experience": 0.25,
        },
        "critical_issues": 0,
    }

# ============================================================================
# SCHEMA ADAPTER TESTS
# ============================================================================


def test_schema_adapter_dataset(
    valid_dashboard_payload,
):
    dataset = (
        SchemaAdapter.get_dataset(
            valid_dashboard_payload
        )
    )

    assert isinstance(
        dataset,
        dict,
    )

    assert dataset["rows"] == 1000


def test_schema_adapter_metrics(
    valid_dashboard_payload,
):
    metrics = (
        SchemaAdapter.get_metrics(
            valid_dashboard_payload
        )
    )

    assert isinstance(
        metrics,
        dict,
    )

    assert "train" in metrics

    assert "holdout" in metrics


def test_schema_adapter_observability_flags(
    valid_dashboard_payload,
):
    flags = (
        SchemaAdapter.get_observability_flags(
            valid_dashboard_payload
        )
    )

    assert isinstance(
        flags,
        list,
    )

    assert len(flags) == 1


def test_schema_adapter_recommendations(
    valid_dashboard_payload,
):
    recommendations = (
        SchemaAdapter.get_recommendations(
            valid_dashboard_payload
        )
    )

    assert isinstance(
        recommendations,
        list,
    )

    assert len(recommendations) == 1


def test_schema_adapter_feature_importance(
    valid_dashboard_payload,
):
    importance = (
        SchemaAdapter.get_feature_importance(
            valid_dashboard_payload
        )
    )

    assert isinstance(
        importance,
        dict,
    )

    assert "salary" in importance

# ============================================================================
# EMPTY STATE TESTS
# ============================================================================


def test_dashboard_empty_state_safety():
    """
    Prevents None/empty crashes.
    """

    payload = {}

    assert (
        SchemaAdapter.get_dataset(
            payload
        )
        == {}
    )

    assert (
        SchemaAdapter.get_metrics(
            payload
        )
        == {}
    )

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
        SchemaAdapter.get_psi_table(
            payload
        )
        == []
    )

    assert (
        SchemaAdapter.get_vif_table(
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
# NORMALIZATION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HIGH", "high"),
        ("Critical", "critical"),
        ("warning", "medium"),
        ("warn", "medium"),
        ("severe", "critical"),
        (None, "medium"),
    ],
)
def test_severity_normalization(
    raw,
    expected,
):
    assert (
        Normalizer.normalize_severity(
            raw
        )
        == expected
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1.5, 1.5),
        ("2.3", 2.3),
        (None, 0.0),
        ("bad", 0.0),
    ],
)
def test_safe_float(
    raw,
    expected,
):
    value = (
        Normalizer.safe_float(raw)
    )

    assert value == expected

# ============================================================================
# RELIABILITY AUDITOR TESTS
# ============================================================================


def test_reliability_score_generation(
    valid_dashboard_payload,
):
    issues = (
        valid_dashboard_payload[
            "issues"
        ]
    )

    result = (
        ReliabilityAuditor.compute_deployability_score(
            issues,
            valid_dashboard_payload,
        )
    )

    assert isinstance(
        result,
        dict,
    )

    required_keys = {
        "overall_score",
        "components",
        "recommendation",
    }

    assert required_keys.issubset(
        set(result.keys())
    )


def test_reliability_score_bounds(
    valid_dashboard_payload,
):
    issues = (
        valid_dashboard_payload[
            "issues"
        ]
    )

    result = (
        ReliabilityAuditor.compute_deployability_score(
            issues,
            valid_dashboard_payload,
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
# OBSERVABILITY TESTS
# ============================================================================


def test_observability_flag_contract(
    valid_dashboard_payload,
):
    flags = (
        SchemaAdapter.get_observability_flags(
            valid_dashboard_payload
        )
    )

    flag = flags[0]

    required_keys = {
        "flag",
        "severity",
        "detail",
    }

    assert required_keys.issubset(
        set(flag.keys())
    )

# ============================================================================
# RECOMMENDATION TESTS
# ============================================================================


def test_recommendation_contract(
    valid_dashboard_payload,
):
    recommendations = (
        SchemaAdapter.get_recommendations(
            valid_dashboard_payload
        )
    )

    recommendation = (
        recommendations[0]
    )

    required_keys = {
        "recommendation_id",
        "severity",
        "recommendation",
        "issue_count",
    }

    assert required_keys.issubset(
        set(
            recommendation.keys()
        )
    )

# ============================================================================
# PSI CONTRACT TESTS
# ============================================================================


def test_psi_table_contract(
    valid_dashboard_payload,
):
    psi_table = (
        SchemaAdapter.get_psi_table(
            valid_dashboard_payload
        )
    )

    row = psi_table[0]

    required_keys = {
        "Feature",
        "PSI Score",
        "Drift Severity",
    }

    assert required_keys.issubset(
        set(row.keys())
    )

# ============================================================================
# VIF CONTRACT TESTS
# ============================================================================


def test_vif_table_contract(
    valid_dashboard_payload,
):
    vif_table = (
        SchemaAdapter.get_vif_table(
            valid_dashboard_payload
        )
    )

    row = vif_table[0]

    required_keys = {
        "Feature",
        "VIF Score",
    }

    assert required_keys.issubset(
        set(row.keys())
    )

# ============================================================================
# TELEMETRY CONTRACT TESTS
# ============================================================================


def test_telemetry_contract(
    valid_dashboard_payload,
):
    telemetry = (
        SchemaAdapter.get_telemetry(
            valid_dashboard_payload
        )
    )

    for (
        metric_name,
        metric_value,
    ) in telemetry.items():
        assert isinstance(
            metric_name,
            str,
        )

        assert isinstance(
            metric_value,
            (
                int,
                float,
            ),
        )

# ============================================================================
# REGRESSION TESTS
# ============================================================================


def test_dashboard_handles_missing_sections():
    """
    Prevents future schema regressions.
    """

    payload = {
        "pipeline_status": "success"
    }

    assert isinstance(
        SchemaAdapter.get_metrics(
            payload
        ),
        dict,
    )

    assert isinstance(
        SchemaAdapter.get_issues(
            payload
        ),
        list,
    )

    assert isinstance(
        SchemaAdapter.get_recommendations(
            payload
        ),
        list,
    )


def test_dashboard_handles_none_sections(
    valid_dashboard_payload,
):
    """
    Prevents None-based rendering crashes.
    """

    payload = copy.deepcopy(
        valid_dashboard_payload
    )

    payload["issues"] = None
    payload["recommendations"] = None
    payload["psi_table"] = None
    payload["vif_table"] = None
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
        SchemaAdapter.get_psi_table(
            payload
        )
        == []
    )

    assert (
        SchemaAdapter.get_vif_table(
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
# PRIMARY METRIC TESTS
# ============================================================================


def test_primary_metric_selection():
    """
    Prevents metric-selection regressions.
    """

    metrics = {
        "accuracy": 0.88,
        "f1": 0.81,
    }

    primary_metric = (
        ReliabilityAuditor.get_primary_metric_name(
            metrics
        )
    )

    assert (
        primary_metric
        == "f1"
    )

# ============================================================================
# RECOMMENDATION STATUS TESTS
# ============================================================================


@pytest.mark.parametrize(
    "score,status",
    [
        (90, "✓ DEPLOYMENT READY"),
        (
            75,
            "⚠ CONDITIONAL DEPLOYMENT",
        ),
        (
            55,
            "✗ DEPLOYMENT BLOCKED",
        ),
    ],
)
def test_deployment_status_mapping(
    score,
    status,
):
    recommendation = (
        ReliabilityAuditor._get_recommendation(
            score
        )
    )

    assert (
        recommendation["status"]
        == status
    )
