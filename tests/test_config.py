# FILE: tests/test_config.py

from __future__ import annotations

import pytest

from app.core.config import (
    DashboardConfig,
    DevModeConfig,
    DriftConfig,
    FeatureQualityConfig,
    ModelConfig,
    ObservabilityConfig,
    PipelineConfig,
    PreprocessingConfig,
    RecommendationConfig,
    ReportConfig,
    TelemetryConfig,
)

# ============================================================================
# PREPROCESSING CONFIG TESTS
# ============================================================================


def test_preprocessing_defaults():
    """
    Preprocessing defaults must remain stable.
    """

    config = PreprocessingConfig()

    assert (
        0
        < config.test_size
        < 1
    )

    assert isinstance(
        config.random_state,
        int,
    )

    assert (
        config.numeric_imputation_strategy
        == "median"
    )


# ============================================================================
# MODEL CONFIG TESTS
# ============================================================================


def test_model_config_defaults():
    """
    Model configuration defaults stability.
    """

    config = ModelConfig()

    assert (
        config.cv_folds >= 2
    )

    assert (
        config.rf_n_estimators
        > 0
    )

    assert isinstance(
        config.random_state,
        int,
    )


# ============================================================================
# DRIFT CONFIG TESTS
# ============================================================================


def test_drift_threshold_ordering():
    """
    Drift thresholds must remain ordered.
    """

    config = DriftConfig()

    assert (
        config.low_drift_threshold
        < config.medium_drift_threshold
    )

    assert (
        config.medium_drift_threshold
        < config.high_drift_threshold
    )


# ============================================================================
# OBSERVABILITY CONFIG TESTS
# ============================================================================


def test_observability_thresholds():
    """
    Observability thresholds must remain valid.
    """

    config = (
        ObservabilityConfig()
    )

    assert (
        config.mild_overfit_gap
        < config.high_overfit_gap
    )

    assert (
        config.high_overfit_gap
        < config.critical_overfit_gap
    )


# ============================================================================
# FEATURE QUALITY TESTS
# ============================================================================


def test_feature_quality_thresholds():
    """
    Feature-quality thresholds must remain ordered.
    """

    config = (
        FeatureQualityConfig()
    )

    assert (
        config.moderate_vif_threshold
        < config.high_vif_threshold
    )

    assert (
        config.high_vif_threshold
        < config.critical_vif_threshold
    )


# ============================================================================
# REPORT CONFIG TESTS
# ============================================================================


def test_report_path_generation():
    """
    Report path must remain deterministic.
    """

    config = ReportConfig()

    path = config.report_path

    assert str(path).endswith(
        config.report_filename
    )


# ============================================================================
# PIPELINE CONFIG TESTS
# ============================================================================


def test_pipeline_config_initialization():
    """
    Master config object stability.
    """

    config = PipelineConfig()

    assert isinstance(
        config.preprocessing,
        PreprocessingConfig,
    )

    assert isinstance(
        config.model,
        ModelConfig,
    )

    assert isinstance(
        config.drift,
        DriftConfig,
    )

    assert isinstance(
        config.observability,
        ObservabilityConfig,
    )

    assert isinstance(
        config.feature_quality,
        FeatureQualityConfig,
    )

    assert isinstance(
        config.telemetry,
        TelemetryConfig,
    )

    assert isinstance(
        config.recommendations,
        RecommendationConfig,
    )

    assert isinstance(
        config.report,
        ReportConfig,
    )

    assert isinstance(
        config.dashboard,
        DashboardConfig,
    )

    assert isinstance(
        config.dev_mode,
        DevModeConfig,
    )


# ============================================================================
# VALIDATION TESTS
# ============================================================================


def test_pipeline_config_validation_success():
    """
    Valid configurations must pass.
    """

    config = PipelineConfig()

    config.validate()


def test_invalid_test_size_validation():
    """
    Invalid split sizes must fail.
    """

    config = PipelineConfig()

    config.preprocessing.test_size = 2.0

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_invalid_cv_folds_validation():
    """
    Invalid CV folds must fail.
    """

    config = PipelineConfig()

    config.model.cv_folds = 1

    with pytest.raises(
        ValueError
    ):
        config.validate()


def test_invalid_drift_thresholds_validation():
    """
    Invalid drift thresholds must fail.
    """

    config = PipelineConfig()

    config.drift.low_drift_threshold = (
        0.5
    )

    config.drift.medium_drift_threshold = (
        0.2
    )

    with pytest.raises(
        ValueError
    ):
        config.validate()


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


def test_pipeline_to_dict():
    """
    Serialization contract stability.
    """

    config = PipelineConfig()

    payload = config.to_dict()

    required_sections = {
        "preprocessing",
        "model",
        "drift",
        "observability",
        "feature_quality",
        "telemetry",
        "recommendations",
        "report",
        "dashboard",
        "dev_mode",
    }

    assert required_sections.issubset(
        set(payload.keys())
    )


def test_pipeline_to_dict_nested_objects():
    """
    Nested configs must serialize to dicts.
    """

    config = PipelineConfig()

    payload = config.to_dict()

    for value in payload.values():
        assert isinstance(
            value,
            dict,
        )


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_no_none_root_configs():
    """
    Prevent dangerous None configs.
    """

    config = PipelineConfig()

    for (
        key,
        value,
    ) in config.__dict__.items():
        assert value is not None


def test_numeric_thresholds_positive():
    """
    Critical numeric thresholds must remain positive.
    """

    config = PipelineConfig()

    assert (
        config.model.rf_n_estimators
        > 0
    )

    assert (
        config.drift.drift_bin_count
        > 0
    )

    assert (
        config.dashboard.default_plot_height
        > 0
    )


# ============================================================================
# DEV MODE TESTS
# ============================================================================


def test_dev_mode_defaults():
    """
    Dev-mode defaults stability.
    """

    config = DevModeConfig()

    assert (
        config.sample_size
        > 0
    )

    assert isinstance(
        config.verbose_logging,
        bool,
    )


# ============================================================================
# DASHBOARD CONFIG TESTS
# ============================================================================


def test_dashboard_defaults():
    """
    Dashboard configuration stability.
    """

    config = DashboardConfig()

    assert (
        config.max_feature_importance_display
        > 0
    )

    assert (
        config.default_plot_height
        > 0
    )
