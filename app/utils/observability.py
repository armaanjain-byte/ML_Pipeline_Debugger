# FILE: app/debugger/observability.py

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

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

DRIFT_THRESHOLDS = {
    "low": 0.1,
    "medium": 0.25,
    "high": 0.5,
}

EPSILON = 1e-8

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ObservabilityFlag:
    """
    Stable observability flag contract.

    Separates:
    - machine-readable identifiers
    - UI-friendly labels
    - operational semantics
    """

    flag_id: str
    severity: str
    title: str
    detail: str
    interpretation: str
    remediation: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flag_id": self.flag_id,
            "severity": self.severity,
            "flag": self.title,
            "detail": self.detail,
            "interpretation": self.interpretation,
            "remediation": self.remediation,
        }


@dataclass
class GeneralizationDiagnostic:
    """
    Encapsulates generalization behavior across:
    - train
    - holdout
    - cross-validation
    """

    train_score: float
    test_score: float
    cv_mean: float
    cv_std: float
    cv_scores: List[float]
    metric_name: str

    @property
    def train_test_gap(self) -> float:
        return float(self.train_score - self.test_score)

    @property
    def cv_test_gap(self) -> float:
        return float(self.cv_mean - self.test_score)

    @property
    def fold_stability_ratio(self) -> float:
        if self.cv_mean <= 0:
            return float("inf")

        return float(
            self.cv_std / (self.cv_mean + EPSILON)
        )

    def get_severity_flags(self) -> List[Dict[str, Any]]:
        """
        Rich observability interpretation layer.
        """

        flags: List[Dict[str, Any]] = []

        # ======================================================
        # Overfitting
        # ======================================================

        if self.train_test_gap > 0.15:
            flags.append(
                {
                    "flag_id": "high_overfit_risk",
                    "flag": "High Overfit Risk",
                    "severity": "high",
                    "detail": (
                        f"Train {self.metric_name.upper()} "
                        f"({self.train_score:.3f}) exceeds "
                        f"test ({self.test_score:.3f}) "
                        f"by {self.train_test_gap:.3f}"
                    ),
                    "interpretation": (
                        "Model appears to memorize training "
                        "distribution and may fail on unseen data."
                    ),
                    "remediation": [
                        "Increase regularization",
                        "Reduce model complexity",
                        "Perform feature selection",
                        "Use early stopping",
                    ],
                }
            )

        elif self.train_test_gap > 0.08:
            flags.append(
                {
                    "flag_id": "mild_overfit",
                    "flag": "Mild Overfit",
                    "severity": "medium",
                    "detail": (
                        f"Train/test gap: "
                        f"{self.train_test_gap:.3f}"
                    ),
                    "interpretation": (
                        "Moderate memorization behavior detected."
                    ),
                    "remediation": [
                        "Monitor production behavior",
                        "Track retraining cadence",
                    ],
                }
            )

        # ======================================================
        # Fold Variance
        # ======================================================

        if self.fold_stability_ratio > 0.10:
            flags.append(
                {
                    "flag_id": "high_fold_variance",
                    "flag": "High Fold Variance",
                    "severity": "high",
                    "detail": (
                        f"CV std={self.cv_std:.3f} "
                        f"(relative std: "
                        f"{self.fold_stability_ratio:.3f})"
                    ),
                    "interpretation": (
                        "Model performance varies heavily "
                        "across validation folds."
                    ),
                    "remediation": [
                        "Investigate feature instability",
                        "Validate stratification logic",
                        "Consider ensemble methods",
                    ],
                }
            )

        # ======================================================
        # Generalization Decay
        # ======================================================

        if self.cv_test_gap > 0.10:
            flags.append(
                {
                    "flag_id": "generalization_decay",
                    "flag": "Generalization Decay",
                    "severity": "high",
                    "detail": (
                        f"CV mean ({self.cv_mean:.3f}) "
                        f"exceeds holdout "
                        f"({self.test_score:.3f}) "
                        f"by {self.cv_test_gap:.3f}"
                    ),
                    "interpretation": (
                        "Potential covariate shift or "
                        "validation leakage detected."
                    ),
                    "remediation": [
                        "Inspect train/test split",
                        "Verify preprocessing isolation",
                        "Analyze PSI drift",
                    ],
                }
            )

        return flags


@dataclass
class MulticollinearityDiagnostic:
    """
    Stable VIF-based multicollinearity analysis.
    """

    vif_scores: Dict[str, float]
    correlation_matrix: Optional[pd.DataFrame] = None

    @property
    def critical_features(self) -> List[str]:
        return [
            feature
            for feature, vif_score in self.vif_scores.items()
            if vif_score > 10.0
        ]

    @property
    def high_vif_features(self) -> List[str]:
        return [
            feature
            for feature, vif_score in self.vif_scores.items()
            if 5.0 < vif_score <= 10.0
        ]

    @property
    def moderate_vif_features(self) -> List[str]:
        return [
            feature
            for feature, vif_score in self.vif_scores.items()
            if 2.0 < vif_score <= 5.0
        ]

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Stable remediation generation.
        """

        recommendations = []

        if self.critical_features:
            recommendations.append(
                {
                    "recommendation_id": "critical_multicollinearity",
                    "severity": "critical",
                    "issue": (
                        f"{len(self.critical_features)} "
                        f"features exceed VIF > 10"
                    ),
                    "features": self.critical_features,
                    "impact": (
                        "Coefficient estimates unstable "
                        "under small perturbations."
                    ),
                    "actions": [
                        "Investigate redundant features",
                        "Apply PCA or feature reduction",
                        "Use L1 regularization",
                    ],
                }
            )

        if self.high_vif_features:
            recommendations.append(
                {
                    "recommendation_id": "high_multicollinearity",
                    "severity": "high",
                    "issue": (
                        f"{len(self.high_vif_features)} "
                        f"features exceed VIF > 5"
                    ),
                    "features": self.high_vif_features,
                    "impact": (
                        "Model predictions may become unstable."
                    ),
                    "actions": [
                        "Consider Ridge regression",
                        "Document feature correlations",
                    ],
                }
            )

        return recommendations


@dataclass
class DriftDiagnostic:
    """
    PSI-based drift assessment.
    """

    psi_scores: Dict[str, float]

    @property
    def low_drift_features(self) -> List[str]:
        return [
            feature
            for feature, psi in self.psi_scores.items()
            if psi < DRIFT_THRESHOLDS["low"]
        ]

    @property
    def medium_drift_features(self) -> List[str]:
        return [
            feature
            for feature, psi in self.psi_scores.items()
            if (
                DRIFT_THRESHOLDS["low"]
                <= psi
                < DRIFT_THRESHOLDS["medium"]
            )
        ]

    @property
    def high_drift_features(self) -> List[str]:
        return [
            feature
            for feature, psi in self.psi_scores.items()
            if (
                DRIFT_THRESHOLDS["medium"]
                <= psi
                < DRIFT_THRESHOLDS["high"]
            )
        ]

    @property
    def critical_drift_features(self) -> List[str]:
        return [
            feature
            for feature, psi in self.psi_scores.items()
            if psi >= DRIFT_THRESHOLDS["high"]
        ]

    def get_impact_assessment(self) -> Dict[str, str]:
        """
        Stable deployment-oriented drift interpretation.
        """

        severe_feature_count = (
            len(self.critical_drift_features)
            + len(self.high_drift_features)
        )

        if severe_feature_count > 5:
            return {
                "severity": "critical",
                "message": (
                    "Extensive distribution shift detected."
                ),
                "implication": (
                    "Model trained on significantly "
                    "different distribution."
                ),
                "action": "Hold deployment immediately.",
            }

        if severe_feature_count > 2:
            return {
                "severity": "high",
                "message": (
                    "Significant feature drift detected."
                ),
                "implication": (
                    "Prediction quality may degrade."
                ),
                "action": (
                    "Deploy only with enhanced monitoring."
                ),
            }

        return {
            "severity": "medium",
            "message": "Moderate drift detected.",
            "implication": (
                "Monitor feature stability in production."
            ),
            "action": "Track PSI during inference.",
        }


# ============================================================================
# OBSERVABILITY ENGINE
# ============================================================================


class ObservabilityEngine:
    """
    Advanced observability utilities.
    """

    @staticmethod
    def compute_calibration_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Stable calibration analysis.
        """

        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if y_pred_proba.ndim > 1:
            y_pred_proba = np.max(
                y_pred_proba,
                axis=1,
            )

        y_pred_proba = np.clip(
            y_pred_proba,
            0,
            1,
        )

        bin_edges = np.linspace(0, 1, n_bins + 1)

        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)

        for prediction, target in zip(
            y_pred_proba,
            y_true,
        ):
            bin_index = min(
                int(prediction * n_bins),
                n_bins - 1,
            )

            bin_sums[bin_index] += prediction
            bin_counts[bin_index] += 1
            bin_true[bin_index] += target

        predicted_probs = bin_sums / (
            bin_counts + EPSILON
        )

        actual_freqs = bin_true / (
            bin_counts + EPSILON
        )

        mask = bin_counts > 0

        ece = np.mean(
            np.abs(
                predicted_probs[mask]
                - actual_freqs[mask]
            )
        )

        return {
            "expected_calibration_error": float(ece),
            "is_well_calibrated": bool(ece < 0.05),
            "predicted_probs": predicted_probs.tolist(),
            "actual_frequencies": actual_freqs.tolist(),
        }

    @staticmethod
    def compute_confidence_spread(
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Stable confidence distribution analysis.
        """

        y_pred_proba = np.asarray(y_pred_proba)

        if y_pred_proba.ndim > 1:
            max_probs = np.max(
                y_pred_proba,
                axis=1,
            )
        else:
            max_probs = y_pred_proba

        high_confidence = np.mean(max_probs > 0.8)
        medium_confidence = np.mean(
            (0.5 <= max_probs)
            & (max_probs <= 0.8)
        )
        low_confidence = np.mean(max_probs < 0.5)

        return {
            "high_confidence_pct": float(
                high_confidence * 100
            ),
            "medium_confidence_pct": float(
                medium_confidence * 100
            ),
            "low_confidence_pct": float(
                low_confidence * 100
            ),
            "mean_confidence": float(
                np.mean(max_probs)
            ),
            "confidence_std": float(
                np.std(max_probs)
            ),
        }

    @staticmethod
    def compute_feature_stability_ranking(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        KL-divergence-based feature stability ranking.
        """

        numeric_features = (
            X_train.select_dtypes(
                include=[np.number]
            ).columns
        )

        rankings = []

        for feature in numeric_features:
            if feature not in X_test.columns:
                continue

            train_values = (
                X_train[feature]
                .dropna()
                .astype(float)
            )

            test_values = (
                X_test[feature]
                .dropna()
                .astype(float)
            )

            if (
                len(train_values) == 0
                or len(test_values) == 0
            ):
                continue

            try:
                bins = np.histogram_bin_edges(
                    np.concatenate(
                        [train_values, test_values]
                    ),
                    bins=30,
                )

                p, _ = np.histogram(
                    train_values,
                    bins=bins,
                )

                q, _ = np.histogram(
                    test_values,
                    bins=bins,
                )

                p = (p + EPSILON) / (
                    np.sum(p) + EPSILON
                )

                q = (q + EPSILON) / (
                    np.sum(q) + EPSILON
                )

                kl_divergence = (
                    np.sum(p * np.log(p / q))
                    + np.sum(q * np.log(q / p))
                ) / 2.0

                stability_score = np.exp(
                    -kl_divergence
                )

                rankings.append(
                    {
                        "feature": feature,
                        "stability_score": float(
                            stability_score
                        ),
                        "kl_divergence": float(
                            kl_divergence
                        ),
                    }
                )

            except Exception as error:
                logger.warning(
                    "Feature stability failed for '%s': %s",
                    feature,
                    str(error),
                )

        rankings.sort(
            key=lambda item: item[
                "stability_score"
            ],
            reverse=True,
        )

        for index, entry in enumerate(rankings):
            entry["rank"] = index + 1

        return rankings

    @staticmethod
    def compute_leakage_risk_score(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Stable leakage-risk analysis.
        """

        leakage_signals = []

        train_hashes = set()

        for _, row in X_train.iterrows():
            row_hash = hash(
                tuple(row.astype(str))
            )
            train_hashes.add(row_hash)

        overlap_count = 0

        for _, row in X_test.iterrows():
            row_hash = hash(
                tuple(row.astype(str))
            )

            if row_hash in train_hashes:
                overlap_count += 1

        overlap_pct = (
            overlap_count / max(len(X_test), 1)
        ) * 100

        if overlap_pct > 0.1:
            leakage_signals.append(
                {
                    "signal_id": "row_overlap",
                    "severity": (
                        "critical"
                        if overlap_pct > 1.0
                        else "high"
                    ),
                    "overlap_pct": float(
                        overlap_pct
                    ),
                }
            )

        return {
            "leakage_signals": leakage_signals,
            "row_overlap_pct": float(
                overlap_pct
            ),
            "is_high_leakage_risk": bool(
                overlap_pct > 1.0
            ),
        }


# ============================================================================
# DEPLOYMENT READINESS
# ============================================================================


class DeploymentReadinessAssessment:
    """
    Stable deployment-readiness synthesis layer.
    """

    @staticmethod
    def generate_deployment_report(
        generalization: GeneralizationDiagnostic,
        multicollinearity: MulticollinearityDiagnostic,
        drift: DriftDiagnostic,
        performance_metric: float,
        issues_by_severity: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Multi-dimensional deployment assessment.
        """

        score = 100.0

        blocking_issues = []
        warnings = []

        # ======================================================
        # Generalization
        # ======================================================

        generalization_flags = (
            generalization.get_severity_flags()
        )

        if any(
            flag["severity"] == "high"
            for flag in generalization_flags
        ):
            score -= 25

            blocking_issues.append(
                "High generalization risk detected."
            )

        elif generalization_flags:
            score -= 10

            warnings.append(
                "Moderate generalization concerns."
            )

        # ======================================================
        # Multicollinearity
        # ======================================================

        if multicollinearity.critical_features:
            score -= 30

            blocking_issues.append(
                "Critical multicollinearity detected."
            )

        elif multicollinearity.high_vif_features:
            score -= 15

            warnings.append(
                "High multicollinearity detected."
            )

        # ======================================================
        # Drift
        # ======================================================

        drift_assessment = (
            drift.get_impact_assessment()
        )

        if (
            drift_assessment["severity"]
            == "critical"
        ):
            score -= 40

            blocking_issues.append(
                "Critical feature drift detected."
            )

        elif (
            drift_assessment["severity"]
            == "high"
        ):
            score -= 20

            warnings.append(
                "Significant feature drift detected."
            )

        # ======================================================
        # Performance
        # ======================================================

        if performance_metric < 0.65:
            score -= 30

            blocking_issues.append(
                "Performance below acceptable threshold."
            )

        elif performance_metric < 0.75:
            score -= 15

            warnings.append(
                "Model performance is modest."
            )

        # ======================================================
        # Issue Accumulation
        # ======================================================

        total_critical = issues_by_severity.get(
            "critical",
            0,
        )

        total_high = issues_by_severity.get(
            "high",
            0,
        )

        if total_critical > 0:
            score -= min(
                30,
                total_critical * 10,
            )

        if total_high > 3:
            score -= (
                total_high - 3
            ) * 5

        score = float(
            np.clip(score, 0, 100)
        )

        return {
            "readiness_score": round(score, 2),
            "can_deploy": bool(
                score >= 75
                and len(blocking_issues) == 0
            ),
            "recommendation": (
                "green"
                if score >= 85
                else "yellow"
                if score >= 70
                else "red"
            ),
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "approval_rationale": (
                _generate_rationale(
                    score,
                    blocking_issues,
                    warnings,
                )
            ),
        }


# ============================================================================
# RATIONALE GENERATION
# ============================================================================


def _generate_rationale(
    score: float,
    blocking_issues: List[str],
    warnings: List[str],
) -> str:
    """
    Stable deployment rationale generation.
    """

    if blocking_issues:
        return (
            "DEPLOYMENT BLOCKED: "
            f"{', '.join(blocking_issues)}"
        )

    if warnings:
        return (
            "Conditional deployment approved. "
            "Enhanced monitoring recommended."
        )

    if score >= 85:
        return (
            "Model passed deployment readiness audit."
        )

    return (
        "Model satisfies minimum deployment criteria."
    )
