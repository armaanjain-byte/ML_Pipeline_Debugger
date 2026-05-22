# FILE: app/utils/feature_utils.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from statsmodels.stats.outliers_influence import (variance_inflation_factor)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

EPSILON = 1e-8

DEFAULT_PSI_BINS = 10

MAX_VIF_FEATURES = 100

MAX_CATEGORY_UNIQUES = 50

# ============================================================================
# ADVANCED DIAGNOSTICS
# ============================================================================


class AdvancedDiagnostics:
    """
    Stable statistical diagnostics engine.

    Responsibilities:
    - PSI drift analysis
    - train/test overlap detection
    - variance stability analysis
    - multicollinearity analysis
    - informative missingness detection

    IMPORTANT:
    Preserves analytical semantics while stabilizing:
    - numerical robustness
    - dtype handling
    - NaN propagation
    - divide-by-zero behavior
    - singular matrix failures
    """

    # ==========================================================
    # PSI ANALYSIS
    # ==========================================================

    @staticmethod
    def compute_psi(
        expected: pd.Series,
        actual: pd.Series,
        bins: int = DEFAULT_PSI_BINS,
    ) -> float:
        """
        Numerically stable PSI computation.
        """

        try:
            expected = pd.to_numeric(
                expected,
                errors="coerce",
            ).dropna()

            actual = pd.to_numeric(
                actual,
                errors="coerce",
            ).dropna()

            if (
                len(expected) == 0
                or len(actual) == 0
            ):
                return 0.0

            breakpoints = np.histogram_bin_edges(
                expected,
                bins=bins,
            )

            expected_counts, _ = np.histogram(
                expected,
                bins=breakpoints,
            )

            actual_counts, _ = np.histogram(
                actual,
                bins=breakpoints,
            )

            expected_percents = (
                expected_counts + EPSILON
            ) / (
                np.sum(expected_counts)
                + EPSILON
            )

            actual_percents = (
                actual_counts + EPSILON
            ) / (
                np.sum(actual_counts)
                + EPSILON
            )

            psi_values = (
                expected_percents
                - actual_percents
            ) * np.log(
                expected_percents
                / actual_percents
            )

            psi_score = np.sum(psi_values)

            if np.isnan(psi_score):
                return 0.0

            if np.isinf(psi_score):
                return 999.0

            return float(psi_score)

        except Exception as error:
            logger.warning(
                "PSI computation failed: %s",
                str(error),
            )

            return 0.0

    @staticmethod
    def compute_all_psi(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Stable dataframe-wide PSI analysis.
        """

        psi_results = []

        numeric_columns = (
            train_df.select_dtypes(
                include=[np.number]
            ).columns
        )

        for column in numeric_columns:
            if column not in test_df.columns:
                continue

            try:
                psi_score = (
                    AdvancedDiagnostics.compute_psi(
                        train_df[column],
                        test_df[column],
                    )
                )

                drift_severity = (
                    AdvancedDiagnostics._resolve_psi_severity(
                        psi_score
                    )
                )

                psi_results.append(
                    {
                        "Feature": column,
                        "PSI Score": round(
                            psi_score,
                            4,
                        ),
                        "Drift Severity": (
                            drift_severity
                        ),
                    }
                )

            except Exception as error:
                logger.warning(
                    "PSI analysis failed "
                    "for feature '%s': %s",
                    column,
                    str(error),
                )

        psi_results.sort(
            key=lambda item:
            item["PSI Score"],
            reverse=True,
        )

        return psi_results

    @staticmethod
    def _resolve_psi_severity(
        psi_score: float,
    ) -> str:
        """
        Stable PSI severity mapping.
        """

        if psi_score >= 0.50:
            return "HIGH"

        if psi_score >= 0.25:
            return "MEDIUM"

        return "LOW"

    # ==========================================================
    # ROW OVERLAP ANALYSIS
    # ==========================================================

    @staticmethod
    def calculate_row_overlap(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[int, float]:
        """
        Stable train/test overlap analysis.
        """

        try:
            train_hashes = set()

            for _, row in train_df.iterrows():
                train_hashes.add(
                    hash(
                        tuple(
                            row.astype(str)
                        )
                    )
                )

            overlap_count = 0

            for _, row in test_df.iterrows():
                row_hash = hash(
                    tuple(
                        row.astype(str)
                    )
                )

                if row_hash in train_hashes:
                    overlap_count += 1

            overlap_pct = (
                overlap_count
                / max(len(test_df), 1)
            ) * 100

            return (
                int(overlap_count),
                float(overlap_pct),
            )

        except Exception as error:
            logger.warning(
                "Row overlap analysis failed: %s",
                str(error),
            )

            return 0, 0.0

    # ==========================================================
    # VARIANCE STABILITY
    # ==========================================================

    @staticmethod
    def compute_train_test_variance_ratio(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Stable variance stability analysis.
        """

        issues = []

        numeric_columns = (
            train_df.select_dtypes(
                include=[np.number]
            ).columns
        )

        for column in numeric_columns:
            if column not in test_df.columns:
                continue

            try:
                train_variance = np.var(
                    pd.to_numeric(
                        train_df[column],
                        errors="coerce",
                    ).dropna()
                )

                test_variance = np.var(
                    pd.to_numeric(
                        test_df[column],
                        errors="coerce",
                    ).dropna()
                )

                variance_ratio = (
                    max(
                        train_variance,
                        test_variance,
                    )
                    / (
                        min(
                            train_variance,
                            test_variance,
                        )
                        + EPSILON
                    )
                )

                if variance_ratio > 3:
                    severity = (
                        "critical"
                        if variance_ratio > 10
                        else "high"
                    )

                    issues.append(
                        {
                            "type":
                            "variance_instability",
                            "column":
                            column,
                            "severity":
                            severity,
                            "description": (
                                f"Variance instability "
                                f"detected "
                                f"(ratio="
                                f"{variance_ratio:.2f})."
                            ),
                        }
                    )

            except Exception as error:
                logger.warning(
                    "Variance analysis failed "
                    "for '%s': %s",
                    column,
                    str(error),
                )

        return issues

    # ==========================================================
    # INFORMATIVE MISSINGNESS
    # ==========================================================

    @staticmethod
    def compute_informative_missingness(
        dataframe: pd.DataFrame,
        target_column: str,
        task_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Detects target-correlated missingness.
        """

        issues = []

        if target_column not in dataframe.columns:
            return issues

        target = dataframe[target_column]

        for column in dataframe.columns:
            if column == target_column:
                continue

            try:
                missing_mask = (
                    dataframe[column]
                    .isna()
                    .astype(int)
                )

                if (
                    missing_mask.sum() == 0
                ):
                    continue

                if task_type == "classification":
                    grouped = target.groupby(
                        missing_mask
                    ).mean()

                    if len(grouped) < 2:
                        continue

                    delta = abs(
                        grouped.iloc[0]
                        - grouped.iloc[1]
                    )

                else:
                    grouped = target.groupby(
                        missing_mask
                    ).mean()

                    if len(grouped) < 2:
                        continue

                    delta = abs(
                        grouped.iloc[0]
                        - grouped.iloc[1]
                    )

                if delta > 0.15:
                    severity = (
                        "high"
                        if delta > 0.30
                        else "medium"
                    )

                    issues.append(
                        {
                            "type":
                            "informative_missingness",
                            "column":
                            column,
                            "severity":
                            severity,
                            "description": (
                                f"Missingness appears "
                                f"target-correlated "
                                f"(delta={delta:.3f})."
                            ),
                        }
                    )

            except Exception as error:
                logger.warning(
                    "Missingness analysis failed "
                    "for '%s': %s",
                    column,
                    str(error),
                )

        return issues

    # ==========================================================
    # MULTICOLLINEARITY
    # ==========================================================

    @staticmethod
    def compute_vif(
        dataframe: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Numerically stable VIF analysis.
        """

        try:
            numeric_df = dataframe.select_dtypes(
                include=[np.number]
            ).copy()

            if numeric_df.empty:
                return {}

            numeric_df = numeric_df.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            numeric_df = numeric_df.dropna(
                axis=1,
                how="all",
            )

            numeric_df = numeric_df.fillna(
                numeric_df.median(
                    numeric_only=True
                )
            )

            # Prevent pathological VIF runtime
            if (
                len(numeric_df.columns)
                > MAX_VIF_FEATURES
            ):
                numeric_df = numeric_df.iloc[
                    :,
                    :MAX_VIF_FEATURES,
                ]

            constant_columns = [
                column
                for column in numeric_df.columns
                if numeric_df[column].nunique()
                <= 1
            ]

            numeric_df = numeric_df.drop(
                columns=constant_columns,
                errors="ignore",
            )

            if numeric_df.empty:
                return {}

            vif_scores = {}

            values = numeric_df.values.astype(
                float
            )

            for index, column in enumerate(
                numeric_df.columns
            ):
                try:
                    vif_score = (
                        variance_inflation_factor(
                            values,
                            index,
                        )
                    )

                    if np.isnan(vif_score):
                        continue

                    if np.isinf(vif_score):
                        vif_score = 999.0

                    vif_scores[column] = float(
                        round(vif_score, 4)
                    )

                except Exception:
                    continue

            vif_scores = dict(
                sorted(
                    vif_scores.items(),
                    key=lambda item:
                    item[1],
                    reverse=True,
                )
            )

            return vif_scores

        except Exception as error:
            logger.warning(
                "VIF computation failed: %s",
                str(error),
            )

            return {}

    # ==========================================================
    # HIGH CARDINALITY ANALYSIS
    # ==========================================================

    @staticmethod
    def detect_high_cardinality_features(
        dataframe: pd.DataFrame,
        threshold: int = MAX_CATEGORY_UNIQUES,
    ) -> List[Dict[str, Any]]:
        """
        Stable categorical cardinality analysis.
        """

        issues = []

        categorical_columns = (
            dataframe.select_dtypes(
                include=["object", "category"]
            ).columns
        )

        for column in categorical_columns:
            try:
                unique_count = dataframe[
                    column
                ].nunique(dropna=True)

                if unique_count > threshold:
                    severity = (
                        "high"
                        if unique_count > threshold * 3
                        else "medium"
                    )

                    issues.append(
                        {
                            "type":
                            "high_cardinality",
                            "column":
                            column,
                            "severity":
                            severity,
                            "description": (
                                f"High-cardinality "
                                f"feature detected "
                                f"({unique_count} unique "
                                f"values)."
                            ),
                        }
                    )

            except Exception as error:
                logger.warning(
                    "Cardinality analysis failed "
                    "for '%s': %s",
                    column,
                    str(error),
                )

        return issues
