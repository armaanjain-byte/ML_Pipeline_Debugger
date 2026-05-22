# FILE: tests/test_recommendation_engine.py

from __future__ import annotations

import pytest

from app.pipeline.recommendation_engine import (RecommendationEngine)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_issues():
    """
    Stable issue fixture.
    """

    return [
        {
            "type": "feature_drift",
            "column": "salary",
            "severity": "high",
            "description": (
                "Feature drift detected."
            ),
        },
        {
            "type": "multicollinearity",
            "column": "income",
            "severity": "medium",
            "description": (
                "High VIF detected."
            ),
        },
        {
            "type": "leakage",
            "column": "future_value",
            "severity": "critical",
            "description": (
                "Potential target leakage."
            ),
        },
    ]


# ============================================================================
# RECOMMENDATION GENERATION TESTS
# ============================================================================


def test_recommendation_generation(
    sample_issues,
):
    """
    Recommendation engine must execute.
    """

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            sample_issues
        )
    )

    assert isinstance(
        recommendations,
        list,
    )


def test_recommendation_schema(
    sample_issues,
):
    """
    Recommendation schema stability.
    """

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            sample_issues
        )
    )

    if recommendations:
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
# SEVERITY TESTS
# ============================================================================


def test_recommendation_severity_values(
    sample_issues,
):
    """
    Severity normalization stability.
    """

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            sample_issues
        )
    )

    valid_severities = {
        "critical",
        "high",
        "medium",
        "low",
        "info",
    }

    for recommendation in recommendations:
        assert (
            recommendation[
                "severity"
            ]
            in valid_severities
        )


# ============================================================================
# EMPTY INPUT TESTS
# ============================================================================


def test_empty_issue_list():
    """
    Empty issues must not crash.
    """

    engine = RecommendationEngine()

    recommendations = (
        engine.generate([])
    )

    assert isinstance(
        recommendations,
        list,
    )


# ============================================================================
# GROUPING TESTS
# ============================================================================


def test_issue_grouping():
    """
    Similar issue types should group safely.
    """

    issues = [
        {
            "type": "feature_drift",
            "column": "salary",
            "severity": "high",
            "description": "Drift 1",
        },
        {
            "type": "feature_drift",
            "column": "income",
            "severity": "high",
            "description": "Drift 2",
        },
    ]

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            issues
        )
    )

    assert isinstance(
        recommendations,
        list,
    )

    assert len(
        recommendations
    ) >= 1


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_no_none_recommendations(
    sample_issues,
):
    """
    Recommendations must not contain dangerous None values.
    """

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            sample_issues
        )
    )

    for recommendation in recommendations:
        for value in recommendation.values():
            assert value is not None


def test_recommendation_ids_are_strings(
    sample_issues,
):
    """
    Recommendation IDs must remain stable.
    """

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            sample_issues
        )
    )

    for recommendation in recommendations:
        assert isinstance(
            recommendation[
                "recommendation_id"
            ],
            str,
        )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_unknown_issue_type():
    """
    Unknown issue types must not crash.
    """

    issues = [
        {
            "type": "unknown_issue",
            "column": "x",
            "severity": "medium",
            "description": "Unknown",
        }
    ]

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            issues
        )
    )

    assert isinstance(
        recommendations,
        list,
    )


def test_missing_optional_fields():
    """
    Partial issue payloads must remain safe.
    """

    issues = [
        {
            "type": "feature_drift",
        }
    ]

    engine = RecommendationEngine()

    recommendations = (
        engine.generate(
            issues
        )
    )

    assert isinstance(
        recommendations,
        list,
    )
