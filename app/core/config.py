# FILE: app/core/config.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass, field


 
# GLOBAL CONSTANTS
 

DEFAULT_RANDOM_STATE = 42

DEFAULT_TEST_SIZE = 0.20

DEFAULT_REPORTS_DIRECTORY = "reports"

DEFAULT_REPORT_FILENAME = "report.json"

DEFAULT_DEV_SAMPLE_SIZE = 5000

SUPPORTED_TASK_TYPES = {
    "classification",
    "regression",
}

SUPPORTED_EXPORT_FORMATS = {
    "json",
}

VALID_SEVERITIES = {
    "critical",
    "high",
    "medium",
    "low",
    "info",
}

SEVERITY_PRIORITY = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "info": 1,
}


 
# PREPROCESSING CONFIGURATION
 


@dataclass
class PreprocessingConfig:
    """
    Stable preprocessing configuration contract.

    Centralizes:
    - split behavior
    - encoding behavior
    - imputation behavior
    - preprocessing determinism
    """

    test_size: float = DEFAULT_TEST_SIZE

    random_state: int = DEFAULT_RANDOM_STATE

    numeric_imputation_strategy: str = "median"

    categorical_imputation_strategy: str = (
        "most_frequent"
    )

    enable_missing_indicators: bool = True

    scaling_strategy: str = "robust"

    onehot_handle_unknown: str = "ignore"

    onehot_min_frequency: float = 0.01

    dense_output: bool = True

    max_categories: Optional[int] = None

    stratify_threshold: int = 20

    preserve_feature_order: bool = True


 
# MODEL CONFIGURATION
 


@dataclass
class ModelConfig:
    """
    Stable model configuration layer.
    """

    random_state: int = DEFAULT_RANDOM_STATE

    cv_folds: int = 5

    n_jobs: int = -1

    enable_permutation_importance: bool = True

    permutation_repeats: int = 5

    normalize_importance_scores: bool = True

    fallback_to_permutation_importance: bool = (
        True
    )

    classification_estimator: str = (
        "random_forest"
    )

    regression_estimator: str = (
        "random_forest"
    )

    rf_n_estimators: int = 100

    rf_max_depth: Optional[int] = None

    dev_mode_rf_n_estimators: int = 10

    dev_mode_rf_max_depth: int = 5

    classification_scoring: str = (
        "f1_weighted"
    )

    regression_scoring: str = (
        "neg_root_mean_squared_error"
    )


 
# DRIFT CONFIGURATION
 


@dataclass
class DriftConfig:
    """
    PSI and drift-analysis configuration.
    """

    low_drift_threshold: float = 0.10

    medium_drift_threshold: float = 0.25

    high_drift_threshold: float = 0.50

    drift_bin_count: int = 10

    epsilon: float = 1e-8

    max_allowed_high_drift_features: int = 5

    enable_kl_divergence: bool = True

    stability_histogram_bins: int = 30


 
# OBSERVABILITY CONFIGURATION
 


@dataclass
class ObservabilityConfig:
    """
    Centralized observability thresholds.

    Prevents:
    - scattered magic numbers
    - semantic drift
    - inconsistent deployment logic
    """

    mild_overfit_gap: float = 0.08

    high_overfit_gap: float = 0.15

    critical_overfit_gap: float = 0.25

    moderate_cv_std: float = 0.04

    high_cv_std: float = 0.08

    critical_cv_std: float = 0.15

    moderate_generalization_gap: float = (
        0.05
    )

    high_generalization_gap: float = 0.10

    critical_generalization_gap: float = (
        0.20
    )

    deployment_ready_threshold: float = 85.0

    staging_ready_threshold: float = 70.0

    needs_review_threshold: float = 50.0


 
# FEATURE QUALITY CONFIGURATION
 


@dataclass
class FeatureQualityConfig:
    """
    Feature quality and multicollinearity settings.
    """

    moderate_vif_threshold: float = 2.0

    high_vif_threshold: float = 5.0

    critical_vif_threshold: float = 10.0

    high_cardinality_threshold: int = 100

    enable_vif_analysis: bool = True

    enable_feature_stability_analysis: bool = (
        True
    )

    max_feature_importance_display: int = 15


 
# TELEMETRY CONFIGURATION
 


@dataclass
class TelemetryConfig:
    """
    Telemetry and runtime diagnostics configuration.
    """

    enable_telemetry: bool = True

    track_runtime_breakdown: bool = True

    enable_performance_logging: bool = True

    telemetry_precision: int = 4

    store_timestamps: bool = True

    include_memory_metrics: bool = False


 
# RECOMMENDATION CONFIGURATION
 


@dataclass
class RecommendationConfig:
    """
    Recommendation-engine configuration.
    """

    max_recommendations: int = 50

    include_remediation_steps: bool = True

    prioritize_critical_issues: bool = True

    enable_grouped_recommendations: bool = (
        True
    )

    normalize_issue_types: bool = True

    deterministic_sorting: bool = True


 
# REPORT CONFIGURATION
 


@dataclass
class ReportConfig:
    """
    Stable report-generation configuration.
    """

    reports_directory: str = (
        DEFAULT_REPORTS_DIRECTORY
    )

    report_filename: str = (
        DEFAULT_REPORT_FILENAME
    )

    export_format: str = "json"

    pretty_print_json: bool = True

    ensure_ascii: bool = False

    include_telemetry: bool = True

    include_feature_importance: bool = True

    include_observability: bool = True

    include_recommendations: bool = True

    include_psi_analysis: bool = True

    include_vif_analysis: bool = True

    schema_version: str = "2.0.0"

    deterministic_serialization: bool = True

    fail_safe_empty_structures: bool = True

    @property
    def report_path(self) -> Path:
        return (
            Path(self.reports_directory)
            / self.report_filename
        )


 
# DASHBOARD CONFIGURATION
 


@dataclass
class DashboardConfig:
    """
    Dashboard rendering safeguards.
    """

    max_feature_importance_display: int = 15

    max_drift_features_display: int = 25

    enable_heavy_visualizations: bool = True

    max_visualization_rows: int = 10000

    default_plot_height: int = 400

    enable_safe_rendering: bool = True

    graceful_empty_state_handling: bool = True

    enable_schema_adapters: bool = True


 
# DEV MODE CONFIGURATION
 


@dataclass
class DevModeConfig:
    """
    Development-mode runtime configuration.
    """

    enabled: bool = False

    sample_size: int = DEFAULT_DEV_SAMPLE_SIZE

    reduce_estimators: bool = True

    disable_heavy_visualizations: bool = False

    verbose_logging: bool = True


 
# MASTER PIPELINE CONFIGURATION
 
 
# BACKWARD COMPATIBILITY CONFIGS
 


 
# BACKWARD COMPATIBILITY CONFIGS
 


class DiagnosticConfig:
    """
    Backward-compatibility shim for legacy modules.

    Older debugger modules expect class-level constants.
    """

    # ==========================================================
    # DRIFT / PSI
    # ==========================================================

    PSI_BINS = 10

    LOW_DRIFT_THRESHOLD = 0.10

    MEDIUM_DRIFT_THRESHOLD = 0.25

    HIGH_DRIFT_THRESHOLD = 0.50

    # ==========================================================
    # LEAKAGE
    # ==========================================================

    LEAKAGE_THRESHOLD = 0.95

    # ==========================================================
    # MULTICOLLINEARITY
    # ==========================================================

    MODERATE_VIF_THRESHOLD = 5.0

    HIGH_VIF_THRESHOLD = 10.0

    CRITICAL_VIF_THRESHOLD = 20.0

    # ==========================================================
    # OVERFITTING
    # ==========================================================

    MILD_OVERFIT_GAP = 0.05

    HIGH_OVERFIT_GAP = 0.10

    CRITICAL_OVERFIT_GAP = 0.20

    # ==========================================================
    # DATA QUALITY
    # ==========================================================

    HIGH_MISSING_THRESHOLD = 0.40

    MODERATE_MISSING_THRESHOLD = 0.20

    HIGH_CARDINALITY_THRESHOLD = 50

    LOW_VARIANCE_THRESHOLD = 0.0001



@dataclass
class PipelineConfig:
    """
    Master centralized configuration object.

    Goals:
    - deterministic runtime behavior
    - centralized thresholds
    - compatibility-safe defaults
    - environment portability
    - reduced configuration drift
    """

    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    drift: DriftConfig = field(
        default_factory=DriftConfig
    )

    observability: ObservabilityConfig = field(
        default_factory=ObservabilityConfig
    )

    feature_quality: FeatureQualityConfig = (
        field(
            default_factory=FeatureQualityConfig
        )
    )

    telemetry: TelemetryConfig = field(
        default_factory=TelemetryConfig
    )

    recommendations: RecommendationConfig = (
        field(
            default_factory=RecommendationConfig
        )
    )

    report: ReportConfig = field(
        default_factory=ReportConfig
    )

    dashboard: DashboardConfig = field(
        default_factory=DashboardConfig
    )

    dev_mode: DevModeConfig = field(
        default_factory=DevModeConfig
    )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def validate(self) -> None:
        """
        Prevents invalid runtime configurations.
        """

        if not (
            0 < self.preprocessing.test_size < 1
        ):
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        if (
            self.model.cv_folds < 2
        ):
            raise ValueError(
                "cv_folds must be >= 2."
            )

        if (
            self.drift.low_drift_threshold
            >= self.drift.medium_drift_threshold
        ):
            raise ValueError(
                "Drift thresholds are invalid."
            )

        if (
            self.drift.medium_drift_threshold
            >= self.drift.high_drift_threshold
        ):
            raise ValueError(
                "Drift thresholds are invalid."
            )

        if (
            self.report.export_format
            not in SUPPORTED_EXPORT_FORMATS
        ):
            raise ValueError(
                "Unsupported export format."
            )

    # ==========================================================
    # SERIALIZATION
    # ==========================================================
    
    def to_dict(self) -> Dict[str, Dict]:
        """
        Stable serialization helper.
        """

        return {
            "preprocessing": asdict(
                self.preprocessing
            ),
            "model": asdict(
                self.model
            ),
            "drift": asdict(
                self.drift
            ),
            "observability": asdict(
                self.observability
            ),
            "feature_quality": asdict(
                self.feature_quality
            ),
            "telemetry": asdict(
                self.telemetry
            ),
            "recommendations": asdict(
                self.recommendations
            ),
            "report": asdict(
                self.report
            ),
            "dashboard": asdict(
                self.dashboard
            ),
            "dev_mode": asdict(
                self.dev_mode
            ),
        }
