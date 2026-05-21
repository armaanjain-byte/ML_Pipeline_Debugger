from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, Any
import json
import os

from app.pipeline.data_loader import DataLoader
from app.pipeline.model import Model
from app.debugger.data_checks import DataChecks
from app.debugger.recommendations import RecommendationEngine
from app.core.config import PipelineConfig
from app.core.exceptions import (
    DataLoadException,
    PreprocessingException,
    ModelTrainingException
)
from app.utils.logger import get_logger

logger = get_logger()


class PipelineRunner:
    """
    Orchestrates the ML pipeline with zero-leakage architecture.
    """

    def __init__(
        self,
        file_path: str,
        target_column: str,
        task_type: str,
        config: PipelineConfig = None,
        dev_mode: bool = False
    ):

        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type
        self.config = config or PipelineConfig()
        self.dev_mode = dev_mode

        # Components
        self.loader = DataLoader(file_path)

        self.model = Model(
            task_type=self.task_type,
            dev_mode=self.dev_mode
        )

        self.checker = DataChecks()

        self.recommender = RecommendationEngine()

        logger.info(
            f"PipelineRunner initialized: "
            f"task_type={task_type}, "
            f"target={target_column}, "
            f"dev_mode={dev_mode}"
        )

    def run(self) -> Dict[str, Any]:

        try:

            logger.info("=" * 50)
            logger.info("PIPELINE START")
            logger.info("=" * 50)

            # =====================================================
            # STEP 1 — LOAD DATA
            # =====================================================

            logger.info("[Step 1/8] Loading data...")

            df = self.loader.load_data()

            if df is None or df.empty:

                raise DataLoadException(
                    "Dataset is empty or could not be loaded."
                )

            if self.dev_mode and len(df) > 5000:

                logger.warning(
                    "DEV MODE ACTIVE: Downsampling to 5000 rows."
                )

                df = df.sample(
                    n=5000,
                    random_state=42
                )

            metadata = self.loader.basic_info(df)

            logger.info(
                f"✓ Data loaded: {metadata['num_rows']} rows"
            )

            # =====================================================
            # STEP 2 — TRAIN TEST SPLIT
            # =====================================================

            X = df.drop(
                columns=[self.target_column]
            )

            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.preprocessing.test_size,
                random_state=self.config.preprocessing.random_state,
                stratify=y if self.task_type == "classification" else None
            )

            # =====================================================
            # STEP 3 — DIAGNOSTICS
            # =====================================================

            logger.info(
                "[Step 2/8] Running diagnostics..."
            )

            train_df = pd.concat(
                [X_train, y_train],
                axis=1
            )

            checks_output = self.checker.run_all_checks(
                train_df,
                self.target_column,
                self.task_type
            )

            logger.info(
                f"✓ Found "
                f"{len(checks_output.get('issues', []))} "
                f"issues"
            )

            # =====================================================
            # STEP 4 — RECOMMENDATIONS
            # =====================================================

            logger.info(
                "[Step 3/8] Generating recommendations..."
            )

            recommendations = self.recommender.generate(
                checks_output
            )

            # =====================================================
            # STEP 5 — MODEL TRAINING
            # =====================================================

            logger.info(
                "[Step 4/8] Training model..."
            )

            feature_names = X_train.columns.tolist()

            self.model.train(
                X_train,
                y_train
            )

            logger.info(
                f"✓ Model trained on "
                f"{X_train.shape[1]} features"
            )

            # =====================================================
            # STEP 6 — PREDICTIONS
            # =====================================================

            logger.info(
                "[Step 5/8] Predicting..."
            )

            y_pred = self.model.predict(
                X_test
            )

            # =====================================================
            # STEP 7 — EVALUATION
            # =====================================================

            logger.info(
                "[Step 6/8] Evaluating..."
            )

            metrics = self.model.evaluate(
                y_test,
                y_pred
            )

            cv_results = self.model.cross_validate(
                X_train,
                y_train
            )

            metrics.update(
                cv_results
            )

            logger.info(
                f"✓ Metrics: "
                f"{list(metrics.keys())}"
            )

            # =====================================================
            # STEP 8 — FEATURE IMPORTANCE
            # =====================================================

            logger.info(
                "[Step 7/8] Computing feature importance..."
            )

            try:

                feature_importance = (
                    self.model.feature_importance(
                        feature_names
                    )
                )

            except AttributeError:

                logger.warning(
                    "Feature importance method "
                    "not found."
                )

                feature_importance = {}

            # =====================================================
            # DASHBOARD REPORT
            # =====================================================

            dashboard_report = {

                "dataset": {

                    "rows":
                        metadata.get(
                            "num_rows",
                            0
                        ),

                    "columns":
                        metadata.get(
                            "num_columns",
                            0
                        ),

                    "target":
                        self.target_column,

                    "task_type":
                        self.task_type,

                    "numeric_features":
                        int(
                            X_train.select_dtypes(
                                include="number"
                            ).shape[1]
                        ),

                    "categorical_features":
                        int(
                            X_train.select_dtypes(
                                exclude="number"
                            ).shape[1]
                        )
                },

                "issues":
                    checks_output.get(
                        "issues",
                        []
                    ),

                "metrics":
                    metrics,

                "recommendations":
                    recommendations.get(
                        "recommendations",
                        []
                    ),

                "critical_issues":
                    recommendations.get(
                        "critical_issues",
                        0
                    ),

                "total_issues":
                    recommendations.get(
                        "total_issues",
                        0
                    ),

                "feature_importance":
                    feature_importance,

                "pipeline_status":
                    "success"
            }

            # =====================================================
            # SAVE REPORT
            # =====================================================

            os.makedirs(
                "reports",
                exist_ok=True
            )

            with open(
                "reports/report.json",
                "w"
            ) as f:

                json.dump(
                    dashboard_report,
                    f,
                    indent=4,
                    default=str
                )

            logger.info(
                "✓ Dashboard report saved "
                "to reports/report.json"
            )

            logger.info("=" * 50)
            logger.info("PIPELINE SUCCESS")
            logger.info("=" * 50)

            return {

                "status":
                    "success",

                "metadata":
                    metadata,

                "checks":
                    checks_output,

                "recommendations":
                    recommendations.get(
                        "recommendations",
                        []
                    ),

                "critical_issues":
                    recommendations.get(
                        "critical_issues",
                        0
                    ),

                "total_issues":
                    recommendations.get(
                        "total_issues",
                        0
                    ),

                "model_metrics":
                    metrics,

                "feature_importance":
                    feature_importance,

                "error":
                    None
            }

        except DataLoadException as e:

            logger.error(
                f"Data Load Failure: {str(e)}"
            )

            return self._failure_response(
                f"Data Error: {str(e)}"
            )

        except Exception as e:

            logger.error(
                f"Unexpected Pipeline Failure: {str(e)}",
                exc_info=True
            )

            return self._failure_response(
                f"System Error: {str(e)}"
            )

    def _failure_response(
        self,
        error_msg: str
    ) -> Dict[str, Any]:

        return {

            "status":
                "failure",

            "error":
                error_msg,

            "metadata":
                None,

            "checks":
                None,

            "recommendations":
                None,

            "model_metrics":
                None,

            "feature_importance":
                None
        }

    def _apply_auto_fixes(
        self,
        df: pd.DataFrame,
        diagnostics: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Automatically removes problematic features.
        """

        cols_to_drop = set()

        # =====================================================
        # TARGET LEAKAGE
        # =====================================================

        for leak in diagnostics.get(
            "target_leakage",
            []
        ):

            cols_to_drop.add(
                leak["column"]
            )

            logger.warning(
                f"AUTO-FIX: Dropping "
                f"{leak['column']} "
                f"due to target leakage."
            )

        # =====================================================
        # CONSTANT FEATURES
        # =====================================================

        for col in diagnostics.get(
            "constant_features",
            []
        ):

            cols_to_drop.add(col)

            logger.warning(
                f"AUTO-FIX: Dropping "
                f"{col} - constant feature."
            )

        # =====================================================
        # HIGH CORRELATION
        # =====================================================

        for col1, col2, corr in diagnostics.get(
            "high_correlation",
            []
        ):

            if col2 not in cols_to_drop:

                cols_to_drop.add(col2)

                logger.warning(
                    f"AUTO-FIX: Dropping "
                    f"{col2} - highly "
                    f"correlated with {col1}."
                )

        return df.drop(
            columns=list(cols_to_drop),
            errors="ignore"
        )