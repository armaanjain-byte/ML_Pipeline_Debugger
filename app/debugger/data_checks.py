import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest
import logging

from app.core.config import DiagnosticConfig

logger = logging.getLogger(__name__)


class DataChecks:
    """
    Comprehensive data validation engine.
    Performs statistical diagnostics on raw training data.
    """

    def run_all_checks(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification"
    ) -> Dict[str, Any]:

        issues = []

        # =====================================================
        # DATASET OVERVIEW
        # =====================================================

        issues.append({
            "category": "dataset",
            "type": "dataset_shape",
            "column": "dataset_wide",
            "severity": "info",
            "description": (
                f"Dataset contains {df.shape[0]} rows "
                f"and {df.shape[1]} columns"
            )
        })

        # =====================================================
        # MISSING VALUES
        # =====================================================

        missing_counts = df.isnull().sum()

        missing_counts = missing_counts[
            missing_counts > 0
        ]

        for col, count in missing_counts.items():

            pct = round(
                (count / len(df)) * 100,
                2
            )

            severity = (
                "high"
                if pct > 40
                else "medium"
            )

            issues.append({
                "category": "missing_values",
                "type": "missing_data",
                "column": col,
                "severity": severity,
                "description": (
                    f"{count} missing values "
                    f"({pct}%) detected"
                )
            })

        # =====================================================
        # DUPLICATES
        # =====================================================

        duplicate_count = df.duplicated().sum()

        issues.append({
            "category": "duplicates",
            "type": "duplicate_rows",
            "column": "dataset_wide",
            "severity": (
                "medium"
                if duplicate_count > 0
                else "info"
            ),
            "description": (
                f"{duplicate_count} duplicate rows detected"
            )
        })

        # =====================================================
        # CONSTANT FEATURES
        # =====================================================

        for col in df.columns:

            if col == target_column:
                continue

            unique_values = df[col].nunique(dropna=False)

            if unique_values <= 1:

                issues.append({
                    "category": "feature_integrity",
                    "type": "constant_feature",
                    "column": col,
                    "severity": "medium",
                    "description": (
                        "Feature contains only one unique value"
                    )
                })

        # =====================================================
        # HIGH CARDINALITY
        # =====================================================

        for col in df.select_dtypes(include="object").columns:

            if col == target_column:
                continue

            unique_ratio = (
                df[col].nunique() / len(df)
            )

            if unique_ratio > 0.8:

                issues.append({
                    "category": "categorical_features",
                    "type": "high_cardinality",
                    "column": col,
                    "severity": "medium",
                    "description": (
                        f"High cardinality detected "
                        f"({df[col].nunique()} unique values)"
                    )
                })

        # =====================================================
        # CLASS BALANCE
        # =====================================================

        if task_type == "classification":

            class_distribution = (
                df[target_column]
                .value_counts(normalize=True)
            )

            minority_ratio = class_distribution.min()

            severity = (
                "high"
                if minority_ratio < 0.1
                else "medium"
            )

            issues.append({
                "category": "target_distribution",
                "type": "class_balance",
                "column": target_column,
                "severity": severity,
                "description": (
                    f"Minority class ratio: "
                    f"{round(minority_ratio, 3)}"
                )
            })

        # =====================================================
        # CORRELATION CHECKS
        # =====================================================

        numeric_df = df.select_dtypes(include=np.number)

        if len(numeric_df.columns) > 1:

            corr_matrix = numeric_df.corr().abs()

            upper_triangle = corr_matrix.where(
                np.triu(
                    np.ones(corr_matrix.shape),
                    k=1
                ).astype(bool)
            )

            for col in upper_triangle.columns:

                high_corr = upper_triangle[col][
                    upper_triangle[col] > 0.9
                ]

                for idx, corr_value in high_corr.items():

                    issues.append({
                        "category": "correlation",
                        "type": "high_correlation",
                        "column": f"{idx} ↔ {col}",
                        "severity": "medium",
                        "description": (
                            f"Correlation = "
                            f"{round(corr_value, 3)}"
                        )
                    })

        # =====================================================
        # OUTLIER DETECTION
        # =====================================================

        outlier_results = self.check_all_outliers(
            df,
            target_column
        )

        issues.extend(outlier_results)

        return {
            "issues": issues,
            "summary": {
                "total_issues": len(issues)
            }
        }

    def check_all_outliers(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> List[Dict[str, Any]]:

        issues = []

        numeric_df = df.select_dtypes(include=np.number)

        if target_column in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_column])

        if numeric_df.empty:
            return issues

        try:

            iso = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            preds = iso.fit_predict(numeric_df)

            outlier_count = np.sum(preds == -1)

            outlier_pct = round(
                (outlier_count / len(df)) * 100,
                2
            )

            severity = (
                "high"
                if outlier_pct > 20
                else "medium"
            )

            issues.append({
                "category": "outliers",
                "type": "multivariate_outliers",
                "column": "dataset_wide",
                "severity": severity,
                "description": (
                    f"Detected {outlier_count} "
                    f"multivariate outliers "
                    f"({outlier_pct}%)"
                )
            })

        except Exception as e:

            logger.warning(
                f"Outlier detection failed: {str(e)}"
            )

        return issues