# FILE: tests/test_exceptions.py

from __future__ import annotations

import pytest

from app.core.exceptions import (
    ConfigurationException,
    DashboardException,
    DataLoadException,
    DiagnosticsException,
    DriftAnalysisException,
    FeatureEngineeringException,
    InvalidTaskTypeException,
    ModelTrainingException,
    PipelineDebuggerException,
    PreprocessingException,
    ReportingException,
    SerializationException,
    VisualizationException,
    normalize_exception,
    safe_exception_payload,
)

# ============================================================================
# BASE EXCEPTION TESTS
# ============================================================================


def test_base_exception_initialization():
    """
    Base exception contract stability.
    """

    error = (
        PipelineDebuggerException(
            "Test error"
        )
    )

    assert (
        error.message
        == "Test error"
    )

    assert (
        error.error_code
        == "pipeline_error"
    )

    assert (
        error.context
        == {}
    )


def test_base_exception_custom_context():
    """
    Context propagation stability.
    """

    error = (
        PipelineDebuggerException(
            "Failure",
            error_code="custom_error",
            context={
                "column": "age"
            },
        )
    )

    assert (
        error.error_code
        == "custom_error"
    )

    assert (
        error.context["column"]
        == "age"
    )


def test_base_exception_to_dict():
    """
    Serialization contract stability.
    """

    error = (
        PipelineDebuggerException(
            "Failure",
            error_code="custom_error",
            context={
                "step": "training"
            },
        )
    )

    payload = error.to_dict()

    required_keys = {
        "error_type",
        "error_code",
        "message",
        "context",
    }

    assert required_keys.issubset(
        set(payload.keys())
    )


def test_base_exception_string():
    """
    String representation consistency.
    """

    error = (
        PipelineDebuggerException(
            "Failure"
        )
    )

    assert str(error) == "Failure"


# ============================================================================
# INHERITANCE TESTS
# ============================================================================


@pytest.mark.parametrize(
    "exception_class",
    [
        DataLoadException,
        PreprocessingException,
        ModelTrainingException,
        DiagnosticsException,
        ReportingException,
        DashboardException,
        ConfigurationException,
        DriftAnalysisException,
        FeatureEngineeringException,
        SerializationException,
        VisualizationException,
        InvalidTaskTypeException,
    ],
)
def test_exception_inheritance(
    exception_class,
):
    """
    All custom exceptions must inherit properly.
    """

    error = exception_class(
        "Test"
    )

    assert isinstance(
        error,
        PipelineDebuggerException,
    )


# ============================================================================
# ERROR CODE TESTS
# ============================================================================


@pytest.mark.parametrize(
    "exception_class,expected_code",
    [
        (
            DataLoadException,
            "data_load_failure",
        ),
        (
            PreprocessingException,
            "preprocessing_failure",
        ),
        (
            ModelTrainingException,
            "model_training_failure",
        ),
        (
            DriftAnalysisException,
            "drift_analysis_failure",
        ),
        (
            SerializationException,
            "serialization_failure",
        ),
        (
            VisualizationException,
            "visualization_failure",
        ),
    ],
)
def test_default_error_codes(
    exception_class,
    expected_code,
):
    """
    Default error codes must remain stable.
    """

    error = exception_class(
        "Failure"
    )

    assert (
        error.error_code
        == expected_code
    )


# ============================================================================
# SAFE PAYLOAD TESTS
# ============================================================================


def test_safe_exception_payload_custom():
    """
    Structured exceptions must serialize safely.
    """

    error = (
        DataLoadException(
            "Dataset failed",
            context={
                "file": "data.csv"
            },
        )
    )

    payload = (
        safe_exception_payload(
            error
        )
    )

    required_keys = {
        "error_type",
        "error_code",
        "message",
        "context",
    }

    assert required_keys.issubset(
        set(payload.keys())
    )


def test_safe_exception_payload_generic():
    """
    Generic exceptions must remain dashboard-safe.
    """

    error = ValueError(
        "Unexpected failure"
    )

    payload = (
        safe_exception_payload(
            error
        )
    )

    assert (
        payload["error_code"]
        == "unhandled_exception"
    )

    assert (
        payload["message"]
        == "Unexpected failure"
    )


# ============================================================================
# NORMALIZATION TESTS
# ============================================================================


def test_normalize_custom_exception():
    """
    Existing custom exceptions should remain unchanged.
    """

    error = (
        DataLoadException(
            "Failure"
        )
    )

    normalized = (
        normalize_exception(
            error
        )
    )

    assert normalized is error


def test_normalize_generic_exception():
    """
    Generic exceptions must become structured.
    """

    error = ValueError(
        "Unexpected issue"
    )

    normalized = (
        normalize_exception(
            error
        )
    )

    assert isinstance(
        normalized,
        PipelineDebuggerException,
    )

    assert (
        normalized.error_code
        == "unhandled_exception"
    )


# ============================================================================
# CONTEXT TESTS
# ============================================================================


def test_context_defaults_to_dict():
    """
    Context must never become None.
    """

    error = (
        PipelineDebuggerException(
            "Failure"
        )
    )

    assert isinstance(
        error.context,
        dict,
    )


def test_context_preserved():
    """
    Context integrity regression test.
    """

    context = {
        "column": "salary",
        "severity": "high",
    }

    error = (
        PipelineDebuggerException(
            "Failure",
            context=context,
        )
    )

    assert (
        error.context
        == context
    )


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_exception_payload_json_safe():
    """
    Exception payloads must remain serializable.
    """

    error = (
        PipelineDebuggerException(
            "Failure",
            context={
                "step": "training"
            },
        )
    )

    payload = error.to_dict()

    assert isinstance(
        payload["message"],
        str,
    )

    assert isinstance(
        payload["context"],
        dict,
    )


def test_exception_payload_has_no_none_keys():
    """
    Prevent unstable None contracts.
    """

    error = (
        PipelineDebuggerException(
            "Failure"
        )
    )

    payload = error.to_dict()

    for key in payload.keys():
        assert key is not None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_empty_message_exception():
    """
    Empty messages should not crash.
    """

    error = (
        PipelineDebuggerException(
            ""
        )
    )

    assert (
        error.message == ""
    )


def test_large_context_payload():
    """
    Large context payloads should remain stable.
    """

    context = {
        f"key_{i}": i
        for i in range(100)
    }

    error = (
        PipelineDebuggerException(
            "Failure",
            context=context,
        )
    )

    assert len(
        error.context
    ) == 100
