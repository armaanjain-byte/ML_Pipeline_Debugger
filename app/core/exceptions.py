# FILE: app/core/exceptions.py

from __future__ import annotations

from typing import Any, Dict, Optional


# ============================================================================
# BASE EXCEPTION
# ============================================================================


class PipelineDebuggerException(Exception):
    """
    Root exception for the ML Pipeline Debugger.

    Goals:
    - deterministic exception contracts
    - structured failure propagation
    - dashboard-safe error reporting
    - compatibility-safe exception hierarchy
    """

    DEFAULT_ERROR_CODE = "pipeline_error"

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)

        self.message = str(message)

        self.error_code = (
            error_code
            or self.DEFAULT_ERROR_CODE
        )

        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Stable serialization contract.
        """

        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        return self.message


# ============================================================================
# DATA LOADING EXCEPTIONS
# ============================================================================


class DataLoadException(
    PipelineDebuggerException
):
    """
    Dataset loading failure.
    """

    DEFAULT_ERROR_CODE = (
        "data_load_failure"
    )


class DatasetEmptyException(
    DataLoadException
):
    """
    Raised when dataset has no rows.
    """

    DEFAULT_ERROR_CODE = (
        "empty_dataset"
    )


class InvalidDatasetFormatException(
    DataLoadException
):
    """
    Raised when dataset format is invalid.
    """

    DEFAULT_ERROR_CODE = (
        "invalid_dataset_format"
    )


class MissingTargetColumnException(
    DataLoadException
):
    """
    Raised when target column does not exist.
    """

    DEFAULT_ERROR_CODE = (
        "missing_target_column"
    )


class UnsupportedFileTypeException(
    DataLoadException
):
    """
    Raised when unsupported file format is used.
    """

    DEFAULT_ERROR_CODE = (
        "unsupported_file_type"
    )


# ============================================================================
# PREPROCESSING EXCEPTIONS
# ============================================================================


class PreprocessingException(
    PipelineDebuggerException
):
    """
    Base preprocessing failure.
    """

    DEFAULT_ERROR_CODE = (
        "preprocessing_failure"
    )


class FeatureEngineeringException(
    PreprocessingException
):
    """
    Feature transformation failure.
    """

    DEFAULT_ERROR_CODE = (
        "feature_engineering_failure"
    )


class FeatureNameAlignmentException(
    PreprocessingException
):
    """
    Raised when transformed feature names
    cannot be aligned safely.
    """

    DEFAULT_ERROR_CODE = (
        "feature_name_alignment_failure"
    )


class InvalidFeatureSchemaException(
    PreprocessingException
):
    """
    Raised when feature contracts are invalid.
    """

    DEFAULT_ERROR_CODE = (
        "invalid_feature_schema"
    )


# ============================================================================
# MODEL EXCEPTIONS
# ============================================================================


class ModelException(
    PipelineDebuggerException
):
    """
    Base model-related failure.
    """

    DEFAULT_ERROR_CODE = (
        "model_failure"
    )


class ModelTrainingException(
    ModelException
):
    """
    Model training failure.
    """

    DEFAULT_ERROR_CODE = (
        "model_training_failure"
    )


class ModelPredictionException(
    ModelException
):
    """
    Prediction execution failure.
    """

    DEFAULT_ERROR_CODE = (
        "model_prediction_failure"
    )


class FeatureImportanceException(
    ModelException
):
    """
    Feature importance extraction failure.
    """

    DEFAULT_ERROR_CODE = (
        "feature_importance_failure"
    )


class CrossValidationException(
    ModelException
):
    """
    Cross-validation execution failure.
    """

    DEFAULT_ERROR_CODE = (
        "cross_validation_failure"
    )


# ============================================================================
# DIAGNOSTICS EXCEPTIONS
# ============================================================================


class DiagnosticsException(
    PipelineDebuggerException
):
    """
    Base diagnostics failure.
    """

    DEFAULT_ERROR_CODE = (
        "diagnostics_failure"
    )


class DriftAnalysisException(
    DiagnosticsException
):
    """
    PSI/drift analysis failure.
    """

    DEFAULT_ERROR_CODE = (
        "drift_analysis_failure"
    )


class VIFAnalysisException(
    DiagnosticsException
):
    """
    Multicollinearity analysis failure.
    """

    DEFAULT_ERROR_CODE = (
        "vif_analysis_failure"
    )


class LeakageDetectionException(
    DiagnosticsException
):
    """
    Leakage detection failure.
    """

    DEFAULT_ERROR_CODE = (
        "leakage_detection_failure"
    )


class ObservabilityException(
    DiagnosticsException
):
    """
    Observability-analysis failure.
    """

    DEFAULT_ERROR_CODE = (
        "observability_failure"
    )


# ============================================================================
# REPORTING EXCEPTIONS
# ============================================================================


class ReportingException(
    PipelineDebuggerException
):
    """
    Base reporting failure.
    """

    DEFAULT_ERROR_CODE = (
        "reporting_failure"
    )


class SerializationException(
    ReportingException
):
    """
    JSON/report serialization failure.
    """

    DEFAULT_ERROR_CODE = (
        "serialization_failure"
    )


class ReportPersistenceException(
    ReportingException
):
    """
    Report file write failure.
    """

    DEFAULT_ERROR_CODE = (
        "report_persistence_failure"
    )


class SchemaValidationException(
    ReportingException
):
    """
    Report schema contract failure.
    """

    DEFAULT_ERROR_CODE = (
        "schema_validation_failure"
    )


# ============================================================================
# DASHBOARD EXCEPTIONS
# ============================================================================


class DashboardException(
    PipelineDebuggerException
):
    """
    Dashboard rendering failure.
    """

    DEFAULT_ERROR_CODE = (
        "dashboard_failure"
    )


class VisualizationException(
    DashboardException
):
    """
    Visualization rendering failure.
    """

    DEFAULT_ERROR_CODE = (
        "visualization_failure"
    )


class DashboardSchemaException(
    DashboardException
):
    """
    Dashboard/report schema mismatch.
    """

    DEFAULT_ERROR_CODE = (
        "dashboard_schema_failure"
    )


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================


class ConfigurationException(
    PipelineDebuggerException
):
    """
    Invalid runtime configuration.
    """

    DEFAULT_ERROR_CODE = (
        "configuration_failure"
    )


class InvalidTaskTypeException(
    ConfigurationException
):
    """
    Unsupported task type.
    """

    DEFAULT_ERROR_CODE = (
        "invalid_task_type"
    )


class InvalidThresholdException(
    ConfigurationException
):
    """
    Invalid threshold configuration.
    """

    DEFAULT_ERROR_CODE = (
        "invalid_threshold_configuration"
    )


# ============================================================================
# UTILITY HELPERS
# ============================================================================


def safe_exception_payload(
    error: Exception,
) -> Dict[str, Any]:
    """
    Dashboard-safe exception serialization.

    Prevents:
    - unsafe raw trace propagation
    - schema instability
    - frontend rendering failures
    """

    if isinstance(
        error,
        PipelineDebuggerException,
    ):
        return error.to_dict()

    return {
        "error_type": error.__class__.__name__,
        "error_code": "unhandled_exception",
        "message": str(error),
        "context": {},
    }


def normalize_exception(
    error: Exception,
) -> PipelineDebuggerException:
    """
    Ensures all propagated failures become
    structured pipeline exceptions.
    """

    if isinstance(
        error,
        PipelineDebuggerException,
    ):
        return error

    return PipelineDebuggerException(
        message=str(error),
        error_code="unhandled_exception",
    )
