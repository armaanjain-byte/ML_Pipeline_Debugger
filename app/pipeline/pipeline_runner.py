import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split

from app.pipeline.data_loader import DataLoader
from app.pipeline.model import Model
from app.debugger.data_checks import DataChecks
from app.debugger.recommendations import RecommendationEngine
from app.core.config import PipelineConfig
from app.utils.feature_utils import AdvancedDiagnostics
from app.core.exceptions import DataLoadException, ModelTrainingException
from app.utils.logger import get_logger

logger = get_logger()

class PipelineRunner:
    """
    Orchestrates the ML pipeline with a zero-leakage architecture.
    Operates in STRICT AUDIT MODE to evaluate leakage, drift, VIF, and overfitting.
    """

    def __init__(self, file_path: str, target_column: str, task_type: str, config: PipelineConfig = None, dev_mode: bool = False):
        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type
        self.config = config or PipelineConfig()
        self.dev_mode = dev_mode

        self.loader = DataLoader(file_path)
        self.model = Model(task_type=self.task_type, dev_mode=self.dev_mode)
        self.checker = DataChecks()
        self.recommender = RecommendationEngine()
        self.telemetry = {}

        logger.info(f"PipelineRunner initialized: task_type={task_type}, target={target_column}")

    def run(self) -> Dict[str, Any]:
        pipeline_start_time = time.time()
        try:
            logger.info("=" * 60)
            logger.info("PIPELINE START - DIAGNOSTICS & AUDIT")
            logger.info("=" * 60)

            # 1. LOAD & INFER
            t0 = time.time()
            df = self.loader.load_data()
            if self.target_column not in df.columns:
                raise DataLoadException(f"Target column '{self.target_column}' missing.")

            if self.dev_mode and len(df) > 5000:
                df = df.sample(n=5000, random_state=self.config.preprocessing.random_state)

            metadata = self.loader.basic_info(df)
            self.telemetry['data_loading_seconds'] = time.time() - t0

            # 2. SPLIT
            t0 = time.time()
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            is_classification = self.task_type == "classification" and y.nunique() < 20
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.preprocessing.test_size,
                random_state=self.config.preprocessing.random_state,
                stratify=y if is_classification else None
            )
            self.telemetry['split_seconds'] = time.time() - t0

            # 3. DRIFT, LEAKAGE & VOLATILITY
            t0 = time.time()
            overlap_count, overlap_pct = AdvancedDiagnostics.calculate_row_overlap(X_train, X_test)
            all_psi_scores = AdvancedDiagnostics.compute_all_psi(X_train, X_test)
            drift_warnings = [p for p in all_psi_scores if p["Drift Severity"] in ["MEDIUM", "HIGH"]]
            
            # Feature Volatility (Variance Collapse)
            volatility_warnings = AdvancedDiagnostics.compute_train_test_variance_ratio(X_train, X_test)
            
            self.telemetry['integrity_checks_seconds'] = time.time() - t0

            # 4. DEEP DIAGNOSTICS (TRAIN ONLY)
            t0 = time.time()
            train_df = pd.concat([X_train, y_train], axis=1)
            checks_output = self.checker.run_all_checks(train_df, self.target_column, self.task_type)
            
            # Informative Missingness
            missingness_warnings = AdvancedDiagnostics.compute_informative_missingness(train_df, self.target_column, self.task_type)
            checks_output.setdefault("issues", []).extend(missingness_warnings)
            checks_output.setdefault("issues", []).extend(volatility_warnings)
            
            if overlap_pct > 0.0:
                severity = "critical" if overlap_pct > 5.0 else "high"
                checks_output.setdefault("issues", []).append({
                    "type": "split_overlap", "column": "dataset_wide", "severity": severity,
                    "description": f"Target Leakage: {overlap_pct:.2f}% of test set identically overlaps training set."
                })
            
            for dw in drift_warnings:
                checks_output.setdefault("issues", []).append({
                    "type": "feature_drift", "column": dw["Feature"], "severity": dw["Drift Severity"].lower(),
                    "description": f"Distribution drift between Train/Test (PSI: {dw['PSI Score']:.2f})."
                })

            vif_data = AdvancedDiagnostics.compute_vif(X_train)
            for col, vif in vif_data.items():
                if vif > 5.0:
                    severity = "critical" if vif > 10.0 else "high"
                    checks_output.setdefault("issues", []).append({
                        "type": "multicollinearity", "column": col, "severity": severity,
                        "description": f"Multicollinearity detected via Variance Inflation Factor (VIF: {vif:.1f})."
                    })
            self.telemetry['diagnostics_seconds'] = time.time() - t0

            # 5. RECOMMENDATIONS
            t0 = time.time()
            recommendations = self.recommender.generate(checks_output)
            self.telemetry['recommendations_seconds'] = time.time() - t0

            # 6. MODEL TRAINING
            t0 = time.time()
            feature_names = X_train.columns.tolist()
            try:
                self.model.train(X_train, y_train)
            except Exception as e:
                raise ModelTrainingException(f"Failed to fit model: {str(e)}")
            self.telemetry['training_seconds'] = time.time() - t0

            # 7. EVALUATION & OVERFIT HEURISTICS
            t0 = time.time()
            y_train_pred = self.model.predict(X_train)
            train_metrics = self.model.evaluate(y_train, y_train_pred)
            y_test_pred = self.model.predict(X_test)
            test_metrics = self.model.evaluate(y_test, y_test_pred)
            cv_results = self.model.cross_validate(X_train, y_train)

            primary_metric = "f1" if self.task_type == "classification" else "r2"
            train_score = train_metrics.get(primary_metric, 0)
            test_score = test_metrics.get(primary_metric, 0)
            cv_mean = cv_results.get(f"cv_mean_{primary_metric}", 0)
            cv_std = cv_results.get(f"cv_std_{primary_metric}", 0)
            
            observability_flags = []
            if (train_score - test_score) > 0.15:
                observability_flags.append({"flag": "High Overfit Risk", "detail": f"Train {primary_metric} ({train_score:.3f}) significantly exceeds Test ({test_score:.3f})."})
            if cv_std > 0.08:
                observability_flags.append({"flag": "Fold Instability", "detail": f"Cross-validation variance is high (std: {cv_std:.3f})."})
            if (cv_mean - test_score) > 0.10:
                observability_flags.append({"flag": "Generalization Decay", "detail": f"CV mean ({cv_mean:.3f}) exceeds Holdout ({test_score:.3f}), indicating domain shift or fold leakage."})

            model_metrics = {
                "train": train_metrics,
                "holdout": test_metrics,
                "cv": cv_results,
                "observability_flags": observability_flags
            }
            self.telemetry['evaluation_seconds'] = time.time() - t0

            # 8. FEATURE IMPORTANCE
            t0 = time.time()
            try:
                feature_importance = self.model.feature_importance(feature_names)
            except AttributeError:
                feature_importance = {}
            self.telemetry['importance_seconds'] = time.time() - t0

            # 9. ASSEMBLE REPORT
            self.telemetry['total_pipeline_seconds'] = time.time() - pipeline_start_time

            vif_table = [{"Feature": k, "VIF Score": round(v, 2)} for k, v in vif_data.items() if v > 2.0]
            vif_table = sorted(vif_table, key=lambda x: x["VIF Score"], reverse=True)

            dashboard_report = {
                "dataset": {
                    "rows": metadata.get("num_rows", 0),
                    "columns": metadata.get("num_columns", 0),
                    "target": self.target_column,
                    "task_type": self.task_type,
                    "overlap_pct": overlap_pct
                },
                "feature_audit": metadata.get("feature_audit", []),
                "psi_table": all_psi_scores,
                "vif_table": vif_table,
                "telemetry": self.telemetry,
                "issues": checks_output.get("issues", []),
                "metrics": model_metrics,
                "recommendations": recommendations.get("recommendations", []),
                "critical_issues": recommendations.get("critical_issues", 0),
                "feature_importance": feature_importance,
                "pipeline_status": "success"
            }

            os.makedirs("reports", exist_ok=True)
            with open("reports/report.json", "w") as f:
                json.dump(dashboard_report, f, indent=4, default=str)

            logger.info("PIPELINE SUCCESS")
            return dashboard_report

        except Exception as e:
            logger.error(f"Pipeline Execution Failed: {str(e)}", exc_info=True)
            return self._failure_response(f"System Error: {str(e)}")

    def _failure_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "pipeline_status": "failure", "error": error_msg,
            "dataset": None, "feature_audit": None, "telemetry": None,
            "issues": None, "recommendations": None, "metrics": None,
            "psi_table": None, "vif_table": None, "feature_importance": None
        }