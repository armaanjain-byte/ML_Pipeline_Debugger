# FILE: app/debugger/recommendations.py

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd


class RecommendationEngine:
    """
    Stable recommendation orchestration layer.

    Goals:
    - deterministic recommendation generation
    - normalized severity handling
    - machine-readable recommendation identifiers
    - dashboard-safe recommendation contracts
    - stable issue taxonomy
    - compatibility-safe schema behavior

    IMPORTANT:
    This implementation preserves the original analytical
    meaning while stabilizing semantics and output contracts.
    """

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

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        task: str,
    ):
        self.df = df
        self.target = target
        self.task = task.lower().strip()

    # ==========================================================
    # Public API
    # ==========================================================

    def generate(
        self,
        checks_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Stable recommendation generation entrypoint.
        """

        issues = checks_output.get("issues", [])

        normalized_issues = self._normalize_issues(issues)

        grouped_issues = self._group_issues(normalized_issues)

        recommendations = self._build_recommendations(
            grouped_issues
        )

        recommendations = self._sort_recommendations(
            recommendations
        )

        critical_issues = self._count_critical_issues(
            normalized_issues
        )

        return {
            "recommendations": recommendations,
            "critical_issues": critical_issues,
        }

    # ==========================================================
    # Issue Normalization
    # ==========================================================

    def _normalize_issues(
        self,
        issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Stabilizes issue contracts.

        Goals:
        - consistent severity values
        - machine-safe identifiers
        - dashboard-safe schemas
        - stable optional handling
        """

        normalized = []

        for raw_issue in issues:
            normalized_issue = {
                "type": self._normalize_issue_type(
                    raw_issue.get("type", "unknown_issue")
                ),
                "column": self._normalize_column_name(
                    raw_issue.get("column", "unknown")
                ),
                "severity": self._normalize_severity(
                    raw_issue.get("severity", "medium")
                ),
                "description": str(
                    raw_issue.get(
                        "description",
                        "No description provided.",
                    )
                ),
            }

            normalized.append(normalized_issue)

        return normalized

    def _normalize_issue_type(
        self,
        issue_type: str,
    ) -> str:
        """
        Converts issue types into deterministic identifiers.
        """

        issue_type = str(issue_type).strip().lower()

        replacements = {
            " ": "_",
            "-": "_",
            "/": "_",
        }

        for source, target in replacements.items():
            issue_type = issue_type.replace(source, target)

        return issue_type

    def _normalize_column_name(
        self,
        column_name: str,
    ) -> str:
        """
        Prevents unstable null/None contracts.
        """

        if column_name is None:
            return "unknown"

        return str(column_name)

    def _normalize_severity(
        self,
        severity: str,
    ) -> str:
        """
        Stable severity normalization.
        """

        normalized = str(severity).strip().lower()

        severity_aliases = {
            "warn": "medium",
            "warning": "medium",
            "severe": "critical",
            "danger": "critical",
        }

        normalized = severity_aliases.get(
            normalized,
            normalized,
        )

        if normalized not in self.VALID_SEVERITIES:
            return "medium"

        return normalized

    # ==========================================================
    # Issue Grouping
    # ==========================================================

    def _group_issues(
        self,
        issues: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Stable issue grouping layer.
        """

        grouped = defaultdict(list)

        for issue in issues:
            grouped[issue["type"]].append(issue)

        return dict(grouped)

    # ==========================================================
    # Recommendation Construction
    # ==========================================================

    def _build_recommendations(
        self,
        grouped_issues: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Deterministic recommendation generation.
        """

        recommendations = []

        for issue_type, issue_group in grouped_issues.items():
            recommendation = self._build_single_recommendation(
                issue_type=issue_type,
                issue_group=issue_group,
            )

            if recommendation:
                recommendations.append(recommendation)

        return recommendations

    def _build_single_recommendation(
        self,
        issue_type: str,
        issue_group: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Stable recommendation construction.
        """

        severity = self._resolve_group_severity(
            issue_group
        )

        affected_columns = sorted(
            {
                issue["column"]
                for issue in issue_group
            }
        )

        recommendation_text = self._generate_recommendation_text(
            issue_type=issue_type,
            severity=severity,
            affected_columns=affected_columns,
        )

        return {
            "recommendation_id": issue_type,
            "severity": severity,
            "affected_columns": affected_columns,
            "issue_count": len(issue_group),
            "recommendation": recommendation_text,
            "issues": issue_group,
        }

    # ==========================================================
    # Recommendation Text
    # ==========================================================

    def _generate_recommendation_text(
        self,
        issue_type: str,
        severity: str,
        affected_columns: List[str],
    ) -> str:
        """
        Stable recommendation language layer.

        IMPORTANT:
        Preserves analytical meaning while normalizing wording.
        """

        column_preview = ", ".join(
            affected_columns[:5]
        )

        recommendation_templates = {
            "missing_values": (
                "Investigate missing-value patterns and "
                "consider imputation improvements for: "
                f"{column_preview}."
            ),
            "feature_drift": (
                "Detected train/test distribution drift. "
                "Review feature stability and data collection "
                f"for: {column_preview}."
            ),
            "multicollinearity": (
                "High multicollinearity detected. "
                "Consider feature selection or dimensionality "
                f"reduction for: {column_preview}."
            ),
            "split_overlap": (
                "Potential train/test leakage detected. "
                "Review dataset splitting strategy and "
                "deduplication safeguards."
            ),
            "class_imbalance": (
                "Class imbalance detected. "
                "Consider rebalancing techniques, "
                "class weighting, or threshold tuning."
            ),
            "outliers": (
                "Outlier-heavy distributions detected. "
                "Consider robust scaling, clipping, or "
                "distribution-aware preprocessing."
            ),
            "high_cardinality": (
                "High-cardinality categorical features detected. "
                "Review encoding strategy and memory impact."
            ),
        }

        fallback_message = (
            "Review detected issues and validate "
            "data quality, preprocessing stability, "
            "and model robustness."
        )

        message = recommendation_templates.get(
            issue_type,
            fallback_message,
        )

        if severity == "critical":
            message = (
                "Immediate attention recommended. "
                + message
            )

        return message

    # ==========================================================
    # Severity Resolution
    # ==========================================================

    def _resolve_group_severity(
        self,
        issue_group: List[Dict[str, Any]],
    ) -> str:
        """
        Resolves highest severity deterministically.
        """

        if not issue_group:
            return "medium"

        resolved_severity = "info"

        for issue in issue_group:
            current = issue["severity"]

            if (
                self.SEVERITY_PRIORITY[current]
                > self.SEVERITY_PRIORITY[
                    resolved_severity
                ]
            ):
                resolved_severity = current

        return resolved_severity

    def _count_critical_issues(
        self,
        issues: List[Dict[str, Any]],
    ) -> int:
        """
        Stable critical-issue counting.
        """

        return sum(
            1
            for issue in issues
            if issue["severity"] == "critical"
        )

    # ==========================================================
    # Sorting
    # ==========================================================

    def _sort_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deterministic recommendation ordering.
        """

        return sorted(
            recommendations,
            key=lambda recommendation: (
                -self.SEVERITY_PRIORITY[
                    recommendation["severity"]
                ],
                -recommendation["issue_count"],
                recommendation["recommendation_id"],
            ),
        )
