# FILE: app/pipeline/pipeline_runner.py


from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import PipelineConfig
from app.core.exceptions import DataLoadException, ModelTrainingException
from app.debugger.data_checks import DataChecks
from app.debugger.recommendations import RecommendationEngine
from app.pipeline.data_loader import DataLoader
from app.pipeline.model import Model
from app.utils.feature_utils import AdvancedDiagnostics
from app.utils.logger import get_logger

logger = get_logger()


class PipelineRunner:
    """
    Central orchestration layer for the ML Pipeline Debugger.

    Responsibilities:
    - dataset loading
    - split integrity
    - diagnostics orchestration
    - recommendation generation
    - model training/evaluation
    - report assembly
    - telemetry tracking
    - report persistence

    IMPORTANT:
    This implementation preserves existing analytical semantics
    while stabilizing:
    - report schema consistency
    - telemetry naming
    - serialization safety
    - filesystem compatibility
    - optional field handling
    - dashboard compatibility
    """

    REPORTS_DIRECTORY_NAME = "reports"
    REPORT_FILENAME = "report.json"

    def __init__(
        self,
        file_path: str,
        target_column: str,
        task_type: str,
        config: PipelineConfig | None = None,
        dev_mode: bool = False,
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type.lower().strip()
        self.config = config or PipelineConfig()
        self.dev_mode = dev_mode

        self.loader = DataLoader(file_path)
        self.model = Model(
            task_type=self.task_type,
            dev_mode=self.dev_mode,
        )

        self.checker = DataChecks()

        self.telemetry: Dict[str, float] = {}

        self.report_directory = self._resolve_report_directory()
        self.report_path = self.report_directory / self.REPORT_FILENAME

        logger.info(
            "PipelineRunner initialized | task_type=%s | target=%s",
            self.task_type,
            self.target_column,
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def run(self) -> Dict[str, Any]:
        """
        Full pipeline execution entrypoint.
        """

        pipeline_start_time = time.time()

        try:
            self._log_pipeline_start()

            # ==================================================
            # 1. LOAD DATA
            # ==================================================

            data_loading_start = time.time()

            dataframe = self.loader.load_data()

            self._validate_target_column(dataframe)

            dataframe = self._apply_dev_mode_sampling(dataframe)

            dataset_metadata = self.loader.basic_info(dataframe)

            self.telemetry["data_loading_seconds"] = (
                time.time() - data_loading_start
            )

            # ==================================================
            # 2. SPLIT DATA
            # ==================================================

            split_start = time.time()

            X = dataframe.drop(columns=[self.target_column])
            y = dataframe[self.target_column]

            is_classification = (
                self.task_type == "classification"
                and y.nunique(dropna=False) < 20
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.preprocessing.test_size,
                random_state=self.config.preprocessing.random_state,
                stratify=y if is_classification else None,
            )

            self.telemetry["split_seconds"] = (
                time.time() - split_start
            )

            # ==================================================
            # 3. DATA INTEGRITY & DRIFT ANALYSIS
            # ==================================================

            integrity_start = time.time()

            overlap_count, overlap_pct = (
                AdvancedDiagnostics.calculate_row_overlap(
                    X_train,
                    X_test,
                )
            )

            psi_scores = AdvancedDiagnostics.compute_all_psi(
                X_train,
                X_test,
            )

            drift_warnings = [
                drift_entry
                for drift_entry in psi_scores
                if drift_entry.get("Drift Severity") in {"MEDIUM", "HIGH"}
            ]

            volatility_warnings = (
                AdvancedDiagnostics.compute_train_test_variance_ratio(
                    X_train,
                    X_test,
                )
            )

            self.telemetry["integrity_checks_seconds"] = (
                time.time() - integrity_start
            )

            # ==================================================
            # 4. TRAIN-ONLY DIAGNOSTICS
            # ==================================================

            diagnostics_start = time.time()

            train_dataframe = pd.concat(
                [X_train, y_train],
                axis=1,
            )

            checks_output = self.checker.run_all_checks(
                train_dataframe,
                self.target_column,
                self.task_type,
            )

            issues = checks_output.setdefault("issues", [])

            missingness_warnings = (
                AdvancedDiagnostics.compute_informative_missingness(
                    train_dataframe,
                    self.target_column,
                    self.task_type,
                )
            )

            issues.extend(missingness_warnings)
            issues.extend(volatility_warnings)

            self._append_overlap_issue(
                issues=issues,
                overlap_pct=overlap_pct,
            )

            self._append_drift_issues(
                issues=issues,
                drift_warnings=drift_warnings,
            )

            vif_scores = AdvancedDiagnostics.compute_vif(X_train)

            self._append_vif_issues(
                issues=issues,
                vif_scores=vif_scores,
            )

            self.telemetry["diagnostics_seconds"] = (
                time.time() - diagnostics_start
            )

            # ==================================================
            # 5. RECOMMENDATIONS
            # ==================================================

            recommendations_start = time.time()

            recommendation_engine = RecommendationEngine(
                df=train_dataframe,
                target=self.target_column,
                task=self.task_type,
            )

            recommendation_output = recommendation_engine.generate(
                checks_output
            )

            self.telemetry["recommendations_seconds"] = (
                time.time() - recommendations_start
            )

            # ==================================================
            # 6. MODEL TRAINING
            # ==================================================

            training_start = time.time()

            try:
                self.model.train(X_train, y_train)

            except Exception as error:
                raise ModelTrainingException(
                    f"Failed to train model: {str(error)}"
                ) from error

            self.telemetry["training_seconds"] = (
                time.time() - training_start
            )

            # ==================================================
            # 7. MODEL EVALUATION
            # ==================================================

            evaluation_start = time.time()

            y_train_predictions = self.model.predict(X_train)
            y_test_predictions = self.model.predict(X_test)

            train_metrics = self.model.evaluate(
                y_train,
                y_train_predictions,
            )

            holdout_metrics = self.model.evaluate(
                y_test,
                y_test_predictions,
            )

            cv_results = self.model.cross_validate(
                X_train,
                y_train,
            )

            observability_flags = self._compute_observability_flags(
                train_metrics=train_metrics,
                holdout_metrics=holdout_metrics,
                cv_results=cv_results,
            )

            metrics_payload = {
                "train": train_metrics,
                "holdout": holdout_metrics,
                "cv": cv_results,
                "observability_flags": observability_flags,
            }

            self.telemetry["evaluation_seconds"] = (
                time.time() - evaluation_start
            )

            # ==================================================
            # 8. FEATURE IMPORTANCE
            # ==================================================

            feature_importance_start = time.time()

            try:
                feature_importance = self.model.feature_importance()

            except Exception as error:
                logger.warning(
                    "Feature importance computation failed: %s",
                    str(error),
                )
                feature_importance = {}

            self.telemetry["importance_seconds"] = (
                time.time() - feature_importance_start
            )

            # ==================================================
            # 9. REPORT ASSEMBLY
            # ==================================================

            self.telemetry["total_pipeline_seconds"] = (
                time.time() - pipeline_start_time
            )

            vif_table = self._build_vif_table(vif_scores)

            report_payload = {
                "pipeline_status": "success",
                "dataset": {
                    "rows": dataset_metadata.get("num_rows", 0),
                    "columns": dataset_metadata.get("num_columns", 0),
                    "target": self.target_column,
                    "task_type": self.task_type,
                    "overlap_pct": float(overlap_pct),
                    "overlap_count": int(overlap_count),
                },
                "feature_audit": dataset_metadata.get(
                    "feature_audit",
                    [],
                ),
                "psi_table": psi_scores,
                "vif_table": vif_table,
                "telemetry": self.telemetry,
                "issues": issues,
                "metrics": metrics_payload,
                "recommendations": recommendation_output.get(
                    "recommendations",
                    [],
                ),
                "critical_issues": int(
                    recommendation_output.get(
                        "critical_issues",
                        0,
                    )
                ),
                "feature_importance": feature_importance,
            }

            self._persist_report(report_payload)

            logger.info("Pipeline execution completed successfully.")

            return report_payload

        except Exception as error:
            logger.error(
                "Pipeline execution failed: %s",
                str(error),
                exc_info=True,
            )

            return self._failure_response(str(error))

    # ==========================================================
    # Report Persistence
    # ==========================================================

    def _persist_report(self, report_payload: Dict[str, Any]) -> None:
        """
        Stable report persistence layer.

        Goals:
        - safe serialization
        - deterministic output location
        - dashboard compatibility
        - filesystem portability
        """

        self.report_directory.mkdir(parents=True, exist_ok=True)

        with self.report_path.open(
            mode="w",
            encoding="utf-8",
        ) as report_file:
            json.dump(
                report_payload,
                report_file,
                indent=4,
                ensure_ascii=False,
                default=self._json_serializer,
            )

    def _resolve_report_directory(self) -> Path:
        """
        Avoids fragile working-directory assumptions.
        """

        repository_root = Path.cwd()

        return repository_root / self.REPORTS_DIRECTORY_NAME

    # ==========================================================
    # Diagnostics Helpers
    # ==========================================================

    def _append_overlap_issue(
        self,
        issues: List[Dict[str, Any]],
        overlap_pct: float,
    ) -> None:
        if overlap_pct <= 0:
            return

        severity = "critical" if overlap_pct > 5 else "high"

        issues.append(
            {
                "type": "split_overlap",
                "column": "dataset_wide",
                "severity": severity,
                "description": (
                    f"Potential train/test overlap detected "
                    f"({overlap_pct:.2f}% overlap)."
                ),
            }
        )

    def _append_drift_issues(
        self,
        issues: List[Dict[str, Any]],
        drift_warnings: List[Dict[str, Any]],
    ) -> None:
        for drift_entry in drift_warnings:
            issues.append(
                {
                    "type": "feature_drift",
                    "column": drift_entry.get("Feature", "unknown"),
                    "severity": drift_entry.get(
                        "Drift Severity",
                        "medium",
                    ).lower(),
                    "description": (
                        f"Distribution drift detected "
                        f"(PSI: {drift_entry.get('PSI Score', 0):.2f})."
                    ),
                }
            )

    def _append_vif_issues(
        self,
        issues: List[Dict[str, Any]],
        vif_scores: Dict[str, float],
    ) -> None:
        for column_name, vif_score in vif_scores.items():
            if vif_score <= 5:
                continue

            severity = "critical" if vif_score > 10 else "high"

            issues.append(
                {
                    "type": "multicollinearity",
                    "column": column_name,
                    "severity": severity,
                    "description": (
                        f"High multicollinearity detected "
                        f"(VIF: {vif_score:.2f})."
                    ),
                }
            )

    # ==========================================================
    # Observability
    # ==========================================================

    def _compute_observability_flags(
        self,
        train_metrics: Dict[str, float],
        holdout_metrics: Dict[str, float],
        cv_results: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Preserves original observability semantics.
        """

        primary_metric = (
            "f1"
            if self.task_type == "classification"
            else "r2"
        )

        train_score = train_metrics.get(primary_metric, 0.0)
        holdout_score = holdout_metrics.get(primary_metric, 0.0)

        cv_mean_key = f"cv_mean_{primary_metric}"
        cv_std_key = f"cv_std_{primary_metric}"

        cv_mean = cv_results.get(cv_mean_key, 0.0)
        cv_std = cv_results.get(cv_std_key, 0.0)

        observability_flags: List[Dict[str, str]] = []

        if (train_score - holdout_score) > 0.15:
            observability_flags.append(
                {
                    "flag": "High Overfit Risk",
                    "detail": (
                        f"Train {primary_metric} ({train_score:.3f}) "
                        f"significantly exceeds holdout "
                        f"({holdout_score:.3f})."
                    ),
                }
            )

        if cv_std > 0.08:
            observability_flags.append(
                {
                    "flag": "Fold Instability",
                    "detail": (
                        f"Cross-validation variance is high "
                        f"(std: {cv_std:.3f})."
                    ),
                }
            )

        if (cv_mean - holdout_score) > 0.10:
            observability_flags.append(
                {
                    "flag": "Generalization Decay",
                    "detail": (
                        f"CV mean ({cv_mean:.3f}) exceeds holdout "
                        f"({holdout_score:.3f})."
                    ),
                }
            )

        return observability_flags

    # ==========================================================
    # Table Builders
    # ==========================================================

    def _build_vif_table(
        self,
        vif_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Dashboard-safe VIF table construction.
        """

        vif_table = [
            {
                "Feature": feature_name,
                "VIF Score": round(float(vif_score), 2),
            }
            for feature_name, vif_score in vif_scores.items()
            if vif_score > 2.0
        ]

        vif_table.sort(
            key=lambda entry: entry["VIF Score"],
            reverse=True,
        )

        return vif_table

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate_target_column(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        if self.target_column not in dataframe.columns:
            raise DataLoadException(
                f"Target column '{self.target_column}' not found."
            )

    def _apply_dev_mode_sampling(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preserves original dev-mode behavior.
        """

        if not self.dev_mode:
            return dataframe

        if len(dataframe) <= 5000:
            return dataframe

        logger.warning(
            "DEV MODE active: sampling dataset down to 5000 rows."
        )

        return dataframe.sample(
            n=5000,
            random_state=self.config.preprocessing.random_state,
        )

    # ==========================================================
    # Serialization
    # ==========================================================

    def _json_serializer(self, value):
        """
        Stable serialization fallback.

        Prevents:
        - numpy serialization crashes
        - pandas scalar incompatibilities
        - silent serialization corruption
        """

        try:
            import numpy as np

            if isinstance(value, np.integer):
                return int(value)

            if isinstance(value, np.floating):
                return float(value)

            if isinstance(value, np.ndarray):
                return value.tolist()

        except Exception:
            pass

        return str(value)

    # ==========================================================
    # Failure Handling
    # ==========================================================

    def _failure_response(
        self,
        error_message: str,
    ) -> Dict[str, Any]:
        """
        Stable failure contract.

        Uses dashboard-safe empty structures instead of None-heavy
        payloads that cause rendering instability.
        """

        return {
            "pipeline_status": "failure",
            "error": error_message,
            "dataset": {},
            "feature_audit": [],
            "telemetry": self.telemetry,
            "issues": [],
            "recommendations": [],
            "metrics": {},
            "psi_table": [],
            "vif_table": [],
            "feature_importance": {},
            "critical_issues": 0,
        }

    # ==========================================================
    # Logging
    # ==========================================================

    def _log_pipeline_start(self) -> None:
        logger.info("=" * 60)
        logger.info("PIPELINE START - DIGNOSTICS & AUDIT")
        logger.info("=" * 60)