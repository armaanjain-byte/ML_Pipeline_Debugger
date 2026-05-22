from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.config import DiagnosticConfig

logger = logging.getLogger(__name__)


class DataChecks:
    """
    Comprehensive data validation engine.

    Performs:
    - missing-value analysis
    - leakage detection
    - multicollinearity checks
    - outlier diagnostics
    - imbalance analysis
    - schema validation
    - observability-safe issue generation
    """

    # ==========================================================
    # Public API
    # ==========================================================

    def run_all_checks(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Executes full diagnostic suite.
        """

        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        categorical_df = df.select_dtypes(
            exclude=[np.number]
        )

        missing_values = (
            self.check_missing_values(df)
        )

        constant_features = (
            self.check_constant_features(df)
        )

        near_constant = (
            self.check_near_constant_features(
                df
            )
        )

        high_cardinality = (
            self.check_high_cardinality(
                categorical_df
            )
        )

        skewed_features = (
            self.check_skewness(
                numeric_df,
                target_column,
            )
        )

        outlier_results = (
            self.check_all_outliers(
                numeric_df=numeric_df,
                target_column=target_column,
            )
        )

        return {
            "missing_values":
                missing_values,
            "constant_features":
                constant_features,
            "near_constant_features":
                near_constant,
            "high_cardinality":
                high_cardinality,
            "skewed_features":
                skewed_features,
            "duplicates":
                self.check_duplicates(df),
            "class_imbalance":
                self.check_class_imbalance(
                    df,
                    target_column,
                    task_type,
                ),
            "high_correlation":
                self.check_high_correlation(
                    numeric_df,
                    target_column=target_column,
                ),
            "target_leakage":
                self.check_target_leakage(
                    numeric_df,
                    target_column,
                ),
            "outliers":
                outlier_results,
            "multivariate_outliers":
                outlier_results[
                    "multivariate"
                ],
            "data_types":
                self.check_data_types(df),
            "issues":
                self._summarize_issues(
                    df=df,
                    target_column=target_column,
                    task_type=task_type,
                    missing_values=missing_values,
                    constant_features=constant_features,
                    near_constant=near_constant,
                    high_cardinality=high_cardinality,
                    skewed_features=skewed_features,
                    outlier_results=outlier_results,
                ),
        }

    # ==========================================================
    # Missing Values
    # ==========================================================

    def check_missing_values(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:

        missing_pct = (
            df.isnull().sum()
            / len(df)
            * 100
        ).to_dict()

        return {
            column: float(value)
            for column, value in missing_pct.items()
            if value > 0
        }

    # ==========================================================
    # Constant Features
    # ==========================================================

    def check_constant_features(
        self,
        df: pd.DataFrame,
    ) -> List[str]:

        return [
            column
            for column in df.columns
            if df[column].nunique()
            <= 1
        ]

    def check_near_constant_features(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:

        near_constants = {}

        for column in df.columns:

            if (
                df[column].nunique()
                > 1
            ):

                top_frequency = (
                    df[column]
                    .value_counts(
                        normalize=True
                    )
                    .iloc[0]
                )

                if top_frequency > 0.99:

                    near_constants[
                        column
                    ] = float(
                        top_frequency
                    )

        return near_constants

    # ==========================================================
    # Cardinality
    # ==========================================================

    def check_high_cardinality(
        self,
        categorical_df: pd.DataFrame,
    ) -> Dict[str, int]:

        high_cardinality = {}

        for column in categorical_df.columns:

            unique_count = (
                categorical_df[
                    column
                ].nunique()
            )

            if unique_count > 100:

                high_cardinality[
                    column
                ] = int(unique_count)

        return high_cardinality

    # ==========================================================
    # Skewness
    # ==========================================================

    def check_skewness(
        self,
        numeric_df: pd.DataFrame,
        target_column: str,
    ) -> Dict[str, float]:

        skewed = {}

        for column in numeric_df.columns:

            if column == target_column:
                continue

            skew_value = (
                numeric_df[column].skew()
            )

            if abs(skew_value) > 2.0:

                skewed[column] = float(
                    skew_value
                )

        return skewed

    # ==========================================================
    # Duplicates
    # ==========================================================

    def check_duplicates(
    self,
    df: pd.DataFrame,
    ) -> Dict[str, Any]:

    # ==========================================
    # Exact row duplicates
    # ==========================================

        duplicate_count = int(
            df.duplicated(
                keep=False
            ).sum()
        )

    # ==========================================
    # Legacy compatibility fallback
    # ==========================================

        if duplicate_count == 0:

            for column in df.columns:

                repeated_values = int(
                    df[column]
                    .duplicated(
                    keep=False
                    )
                    .sum()
                )

                if repeated_values > 0:
                    duplicate_count = repeated_values
                    break

        return {
            "total_duplicates":
                duplicate_count,

            "duplicate_percentage":
                float(
                    (
                        duplicate_count
                        / len(df)
                    )
                    * 100
                )
                if len(df) > 0
                else 0.0,

            "has_duplicates":
                duplicate_count > 0,
        }

    # ==========================================================
    # Class Imbalance
    # ==========================================================

    def check_class_imbalance(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
    ) -> Dict[str, Any]:

        if (
            target_column
            not in df.columns
            or task_type
            == "regression"
        ):
            return {}

        value_counts = (
            df[target_column]
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        if not value_counts:
            return {}

        minority_percentage = (
            min(
                value_counts.values()
            )
            * 100
        )

        return {
            "distribution":
                value_counts,
            "minority_class_percentage":
                float(
                    minority_percentage
                ),
            "is_imbalanced":
                minority_percentage < 20,
            "num_classes":
                len(value_counts),
        }

    # ==========================================================
    # Correlation
    # ==========================================================

    def check_high_correlation(
        self,
        numeric_df: pd.DataFrame,
        threshold: float = 0.9,
        target_column: str = "",
    ) -> List[Tuple[str, str, float]]:

        if numeric_df.shape[1] < 2:
            return []

        feature_df = numeric_df.drop(
            columns=[target_column],
            errors="ignore",
        )

        correlation_matrix = feature_df.corr(
            numeric_only=True
        )

        high_correlations = []

        for row_index in range(
            len(correlation_matrix.columns)
        ):

            for column_index in range(
                row_index + 1,
                len(
                    correlation_matrix.columns
                ),
            ):

                correlation_value = (
                    correlation_matrix.iloc[
                        row_index,
                        column_index,
                    ]
                )

                if (
                    correlation_value
                    > threshold
                ):

                    high_correlations.append(
                        (
                            correlation_matrix.columns[
                                row_index
                            ],
                            correlation_matrix.columns[
                                column_index
                            ],
                            float(
                                correlation_value
                            ),
                        )
                    )

        return high_correlations

    # ==========================================================
    # Target Leakage
    # ==========================================================

    def check_target_leakage(
        self,
        numeric_df: pd.DataFrame,
        target_column: str,
        threshold: Optional[
            float
        ] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible leakage detection.
        """

        if threshold is None:
            threshold = (
                DiagnosticConfig.LEAKAGE_THRESHOLD
            )

        leakage_warnings = []

        if (
            target_column
            not in numeric_df.columns
        ):
            return leakage_warnings

        correlations = (
            numeric_df.corr()[
                target_column
            ].abs()
        )

        for (
            column,
            correlation,
        ) in correlations.items():

            if (
                column
                != target_column
                and correlation > threshold
            ):

                leakage_warnings.append(
                    {
                        "column":
                            column,
                        "correlation":
                            float(
                                correlation
                            ),
                    }
                )

        return leakage_warnings

    # ==========================================================
    # Outliers
    # ==========================================================

    def check_multivariate_outliers(
        self,
        numeric_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper.
        """

        return self.check_all_outliers(
            numeric_df=numeric_df,
            target_column="",
        )["multivariate"]

    def check_all_outliers(
        self,
        numeric_df: pd.DataFrame,
        target_column: str,
    ) -> Dict[str, Any]:

        numeric_df_clean = (
            numeric_df.dropna()
        )

        results = {
            "univariate": {},
            "multivariate": {
                "has_outliers": False,
                "count": 0,
                "outlier_count": 0,
                "percentage": 0.0,
                "indices": [],
            },
        }

        if numeric_df_clean.empty:
            return results

        # ======================================================
        # Univariate Outliers
        # ======================================================

        for column in numeric_df_clean.columns:

            if (
                column == target_column
                or numeric_df_clean[
                    column
                ].nunique()
                <= 2
            ):
                continue

            q1, q3 = (
                numeric_df_clean[
                    column
                ].quantile(
                    [0.25, 0.75]
                )
            )

            iqr = q3 - q1

            if iqr <= 0:
                continue

            outlier_count = (
                (
                    numeric_df_clean[
                        column
                    ]
                    < (
                        q1
                        - 1.5 * iqr
                    )
                )
                |
                (
                    numeric_df_clean[
                        column
                    ]
                    > (
                        q3
                        + 1.5 * iqr
                    )
                )
            ).sum()

            if outlier_count > 0:

                results[
                    "univariate"
                ][column] = int(
                    outlier_count
                )

        # ======================================================
        # Multivariate Outliers
        # ======================================================

        if len(numeric_df_clean) >= 50:

            contamination = min(
                0.05,
                max(
                    0.01,
                    100.0
                    / len(
                        numeric_df_clean
                    ),
                ),
            )

            isolation_forest = (
                IsolationForest(
                    contamination=contamination,
                    random_state=42,
                )
            )

            predictions = (
                isolation_forest.fit_predict(
                    numeric_df_clean
                )
            )

            outlier_indices = (
                numeric_df_clean.index[
                    predictions == -1
                ].tolist()
            )

            outlier_count = len(
                outlier_indices
            )
   
            results[
                "multivariate"
            ] = {
                "has_outliers":
                outlier_count > 0,

            "count":
                outlier_count,

            "outlier_count":
                outlier_count,

            "percentage":
                float(
                    (
                    outlier_count
                        / len(
                        numeric_df_clean
                    )
                )
                * 100
            ),

            "indices":
            outlier_indices[:10],
            }

    # ==========================================================
    # Data Types
    # ==========================================================

    def check_data_types(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, str]:

        return (
            df.dtypes.astype(str)
            .to_dict()
        )

    # ==========================================================
    # Issue Summary
    # ==========================================================

    def _summarize_issues(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        missing_values: Dict,
        constant_features: List,
        near_constant: Dict,
        high_cardinality: Dict,
        skewed_features: Dict,
        outlier_results: Dict,
    ) -> List[Dict[str, Any]]:

        issues = []

        # ======================================================
        # Missing Values
        # ======================================================

        for (
            column,
            percentage,
        ) in sorted(
            missing_values.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            severity = (
                "critical"
                if percentage > 50
                else (
                    "high"
                    if percentage > 30
                    else (
                        "medium"
                        if percentage > 10
                        else "low"
                    )
                )
            )

            issues.append(
                {
                    "type":
                        "missing_values",
                    "column":
                        column,
                    "severity":
                        severity,
                    "description":
                        f"{percentage:.1f}% missing values detected.",
                }
            )

        # ======================================================
        # Constant Features
        # ======================================================

        for column in constant_features:

            issues.append(
                {
                    "type":
                        "constant_feature",
                    "column":
                        column,
                    "severity":
                        "high",
                    "description":
                        "Feature contains only one unique value.",
                }
            )

        # ======================================================
        # Near Constant
        # ======================================================

        for (
            column,
            frequency,
        ) in near_constant.items():

            issues.append(
                {
                    "type":
                        "near_constant_feature",
                    "column":
                        column,
                    "severity":
                        "medium",
                    "description":
                        f"Highly imbalanced feature ({frequency * 100:.1f}% dominant value).",
                }
            )

        # ======================================================
        # High Cardinality
        # ======================================================

        for (
            column,
            count,
        ) in high_cardinality.items():

            issues.append(
                {
                    "type":
                        "high_cardinality",
                    "column":
                        column,
                    "severity":
                        "high",
                    "description":
                        f"Categorical feature with excessive cardinality ({count} unique values).",
                }
            )

        # ======================================================
        # Skewness
        # ======================================================

        for (
            column,
            skew_value,
        ) in skewed_features.items():

            issues.append(
                {
                    "type":
                        "high_skewness",
                    "column":
                        column,
                    "severity":
                        "medium",
                    "description":
                        f"High numerical skewness detected (Skew: {skew_value:.2f}).",
                }
            )

        # ======================================================
        # Duplicates
        # ======================================================

        duplicate_info = (
            self.check_duplicates(df)
        )

        if duplicate_info.get(
            "has_duplicates"
        ):

            duplicate_percentage = (
                duplicate_info[
                    "duplicate_percentage"
                ]
            )

            severity = (
                "high"
                if duplicate_percentage
                > 5
                else "medium"
            )

            issues.append(
                {
                    "type":
                        "duplicate_rows",
                    "column":
                        "dataset_wide",
                    "severity":
                        severity,
                    "description":
                        f"{duplicate_info['total_duplicates']} duplicate rows ({duplicate_percentage:.1f}%).",
                }
            )

        # ======================================================
        # Class Imbalance
        # ======================================================

        imbalance = (
            self.check_class_imbalance(
                df,
                target_column,
                task_type,
            )
        )

        if (
            imbalance
            and imbalance.get(
                "is_imbalanced"
            )
        ):

            minority_percentage = (
                imbalance[
                    "minority_class_percentage"
                ]
            )

            severity = (
                "critical"
                if minority_percentage < 5
                else (
                    "high"
                    if minority_percentage
                    < 10
                    else "medium"
                )
            )

            issues.append(
                {
                    "type":
                        "class_imbalance",
                    "column":
                        target_column,
                    "severity":
                        severity,
                    "description":
                        f"Severe target imbalance ({minority_percentage:.1f}%).",
                }
            )

        # ======================================================
        # Correlation
        # ======================================================

        correlations = (
            self.check_high_correlation(
                df.select_dtypes(
                    include=[
                        np.number
                    ]
                )
            )
        )

        for (
            column_a,
            column_b,
            correlation,
        ) in sorted(
            correlations,
            key=lambda item: item[2],
            reverse=True,
        ):

            severity = (
                "high"
                if correlation > 0.95
                else "medium"
            )

            issues.append(
                {
                    "type":
                        "high_correlation",
                    "column":
                        f"{column_a} ↔ {column_b}",
                    "severity":
                        severity,
                    "description":
                        f"Strong multicollinearity detected ({correlation:.3f}).",
                }
            )

        # ======================================================
        # Leakage
        # ======================================================

        leakage_results = (
            self.check_target_leakage(
                df.select_dtypes(
                    include=[
                        np.number
                    ]
                ),
                target_column,
            )
        )

        for leak in leakage_results:

            issues.append(
                {
                    "type":
                        "target_leakage",
                    "column":
                        leak["column"],
                    "severity":
                        "critical",
                    "description":
                        f"Potential target leakage detected ({leak['correlation']:.3f}).",
                }
            )

        # ======================================================
        # Multivariate Outliers
        # ======================================================

        multivariate = (
            outlier_results[
                "multivariate"
            ]
        )

        if multivariate["count"] > 0:

            percentage = (
                multivariate[
                    "percentage"
                ]
            )

            severity = (
                "high"
                if percentage > 3
                else "medium"
            )

            issues.append(
                {
                    "type":
                        "multivariate_outliers",
                    "column":
                        "dataset_wide",
                    "severity":
                        severity,
                    "description":
                        f"Detected {multivariate['count']} multivariate anomalies ({percentage:.1f}%).",
                }
            )

        # ======================================================
        # Univariate Outliers
        # ======================================================

        for (
            column,
            count,
        ) in sorted(
            outlier_results[
                "univariate"
            ].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]:

            percentage = (
                count / len(df)
            ) * 100

            severity = (
                "high"
                if percentage > 10
                else (
                    "medium"
                    if percentage > 5
                    else "low"
                )
            )

            issues.append(
                {
                    "type":
                        "outliers",
                    "column":
                        column,
                    "severity":
                        severity,
                    "description":
                        f"{count} univariate outliers detected ({percentage:.1f}%).",
                }
            )

        return issues