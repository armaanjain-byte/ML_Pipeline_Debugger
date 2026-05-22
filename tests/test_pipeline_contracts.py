# FILE: tests/test_pipeline_contracts.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.pipeline.pipeline_runner import PipelineRunner


# ============================================================================
# TEST DATASET
# ============================================================================


@pytest.fixture
def sample_classification_dataset(tmp_path):
    """
    Stable synthetic classification dataset.
    """

    rng = np.random.default_rng(42)

    rows = 300

    dataframe = pd.DataFrame(
        {
            "age": rng.integers(18, 70, rows),
            "salary": rng.normal(
                50000,
                12000,
                rows,
            ),
            "experience": rng.normal(
                8,
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
            "target": rng.choice(
                [0, 1],
                rows,
            ),
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


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================


def test_pipeline_executes_successfully(
    sample_classification_dataset,
):
    """
    Ensures pipeline completes end-to-end.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["pipeline_status"]
        == "success"
    )


# ============================================================================
# ROOT CONTRACT
# ============================================================================


def test_pipeline_root_schema(
    sample_classification_dataset,
):
    """
    Protects top-level report schema.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    required_keys = {
        "pipeline_status",
        "dataset",
        "feature_audit",
        "telemetry",
        "issues",
        "metrics",
        "recommendations",
        "psi_table",
        "vif_table",
        "feature_importance",
        "critical_issues",
    }

    missing_keys = (
        required_keys
        - set(result.keys())
    )

    assert (
        not missing_keys
    ), f"Missing keys: {missing_keys}"


# ============================================================================
# DATASET CONTRACT
# ============================================================================


def test_dataset_schema(
    sample_classification_dataset,
):
    """
    Protects dataset metadata contract.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    dataset = result["dataset"]

    required_keys = {
        "rows",
        "columns",
        "target",
        "task_type",
        "overlap_pct",
        "overlap_count",
    }

    assert required_keys.issubset(
        set(dataset.keys())
    )

    assert dataset["rows"] > 0

    assert dataset["columns"] > 0


# ============================================================================
# METRICS CONTRACT
# ============================================================================


def test_metrics_contract(
    sample_classification_dataset,
):
    """
    Protects metrics schema.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    metrics = result["metrics"]

    required_sections = {
        "train",
        "holdout",
        "cv",
        "observability_flags",
    }

    assert required_sections.issubset(
        set(metrics.keys())
    )

    assert isinstance(
        metrics["train"],
        dict,
    )

    assert isinstance(
        metrics["holdout"],
        dict,
    )

    assert isinstance(
        metrics["cv"],
        dict,
    )

    assert isinstance(
        metrics[
            "observability_flags"
        ],
        list,
    )


# ============================================================================
# ISSUE CONTRACT
# ============================================================================


def test_issue_schema(
    sample_classification_dataset,
):
    """
    Protects issue schema consistency.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    issues = result["issues"]

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
# RECOMMENDATION CONTRACT
# ============================================================================


def test_recommendation_schema(
    sample_classification_dataset,
):
    """
    Protects recommendation schema.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    recommendations = result[
        "recommendations"
    ]

    assert isinstance(
        recommendations,
        list,
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
# TELEMETRY CONTRACT
# ============================================================================


def test_telemetry_schema(
    sample_classification_dataset,
):
    """
    Protects telemetry consistency.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    telemetry = result[
        "telemetry"
    ]

    expected_metrics = {
        "data_loading_seconds",
        "split_seconds",
        "diagnostics_seconds",
        "training_seconds",
        "evaluation_seconds",
        "total_pipeline_seconds",
    }

    assert expected_metrics.issubset(
        set(telemetry.keys())
    )

    for value in telemetry.values():
        assert isinstance(
            value,
            (
                int,
                float,
            ),
        )


# ============================================================================
# FEATURE IMPORTANCE CONTRACT
# ============================================================================


def test_feature_importance_contract(
    sample_classification_dataset,
):
    """
    Ensures importance structure remains stable.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    importance = result[
        "feature_importance"
    ]

    assert isinstance(
        importance,
        dict,
    )

    for (
        feature,
        value,
    ) in importance.items():
        assert isinstance(
            feature,
            str,
        )

        assert isinstance(
            value,
            (
                int,
                float,
            ),
        )


# ============================================================================
# SERIALIZATION CONTRACT
# ============================================================================


def test_report_is_json_serializable(
    sample_classification_dataset,
):
    """
    Prevents serialization regressions.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    try:
        json.dumps(result)

    except TypeError as error:
        pytest.fail(
            f"Report is not JSON serializable: "
            f"{str(error)}"
        )


# ============================================================================
# FAILURE CONTRACT
# ============================================================================


def test_failure_contract():
    """
    Protects dashboard-safe failure payload.
    """

    runner = PipelineRunner(
        file_path="non_existent.csv",
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    assert (
        result["pipeline_status"]
        == "failure"
    )

    required_keys = {
        "pipeline_status",
        "error",
        "dataset",
        "issues",
        "metrics",
        "recommendations",
        "psi_table",
        "vif_table",
        "feature_importance",
    }

    assert required_keys.issubset(
        set(result.keys())
    )


# ============================================================================
# REGRESSION SAFETY
# ============================================================================


def test_no_none_root_contracts(
    sample_classification_dataset,
):
    """
    Prevents dangerous None-based schemas.
    """

    runner = PipelineRunner(
        file_path=str(
            sample_classification_dataset
        ),
        target_column="target",
        task_type="classification",
        dev_mode=True,
    )

    result = runner.run()

    for key, value in result.items():
        assert value is not None, (
            f"Root key '{key}' "
            f"contains None."
        )
