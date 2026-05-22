# FILE: dashboard.py

from __future__ import annotations

import json
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ML Reliability Platform",
    page_icon="▢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.8rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 1600px;
    }

    h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    h2 {
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 0.4rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    h3 {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
        color: #34495e;
    }

    .deployment-banner {
        padding: 1.2rem;
        border-radius: 6px;
        color: white;
        font-weight: bold;
        margin-bottom: 1.2rem;
        text-align: center;
        font-size: 1.15rem;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3498db;
        margin-bottom: 0.8rem;
    }

    .severity-critical {
        color: #c0392b;
        font-weight: 700;
    }

    .severity-high {
        color: #d35400;
        font-weight: 700;
    }

    .severity-medium {
        color: #f39c12;
        font-weight: 600;
    }

    .severity-low {
        color: #27ae60;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SCHEMA HELPERS
# ============================================================================


class SchemaAdapter:
    """
    Centralized schema compatibility layer.

    Prevents:
    - nested get() chaos
    - old/new schema incompatibility
    - dashboard rendering crashes
    - missing-field instability
    """

    @staticmethod
    def get_dataset(result: Dict[str, Any]) -> Dict[str, Any]:
        return result.get("dataset", {}) or {}

    @staticmethod
    def get_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
        metrics = result.get("metrics", {})

        if "metrics" in metrics:
            return metrics.get("metrics", {})

        return metrics or {}

    @staticmethod
    def get_train_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
        return SchemaAdapter.get_metrics(result).get(
            "train",
            {},
        )

    @staticmethod
    def get_holdout_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
        return SchemaAdapter.get_metrics(result).get(
            "holdout",
            {},
        )

    @staticmethod
    def get_cv_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
        return SchemaAdapter.get_metrics(result).get(
            "cv",
            {},
        )

    @staticmethod
    def get_observability_flags(
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        metrics = SchemaAdapter.get_metrics(result)

        flags = metrics.get(
            "observability_flags",
            [],
        )

        if not isinstance(flags, list):
            return []

        return flags

    @staticmethod
    def get_issues(
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        issues = result.get("issues", [])

        if not isinstance(issues, list):
            return []

        return issues

    @staticmethod
    def get_feature_importance(
        result: Dict[str, Any],
    ) -> Dict[str, float]:
        importance = result.get(
            "feature_importance",
            {},
        )

        if not isinstance(importance, dict):
            return {}

        return importance

    @staticmethod
    def get_recommendations(
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        recommendations = result.get(
            "recommendations",
            [],
        )

        if not isinstance(recommendations, list):
            return []

        return recommendations

    @staticmethod
    def get_psi_table(
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        psi_table = result.get("psi_table", [])

        if not isinstance(psi_table, list):
            return []

        return psi_table

    @staticmethod
    def get_vif_table(
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        vif_table = result.get("vif_table", [])

        if not isinstance(vif_table, list):
            return []

        return vif_table

    @staticmethod
    def get_feature_audit(
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        audit = result.get("feature_audit", [])

        if not isinstance(audit, list):
            return []

        return audit

    @staticmethod
    def get_telemetry(
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        telemetry = result.get("telemetry", {})

        if not isinstance(telemetry, dict):
            return {}

        return telemetry


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================


class Normalizer:
    @staticmethod
    def normalize_severity(
        severity: str,
    ) -> str:
        if severity is None:
            return "medium"

        severity = str(severity).strip().lower()

        aliases = {
            "warn": "medium",
            "warning": "medium",
            "severe": "critical",
        }

        severity = aliases.get(
            severity,
            severity,
        )

        valid = {
            "critical",
            "high",
            "medium",
            "low",
            "info",
        }

        if severity not in valid:
            return "medium"

        return severity

    @staticmethod
    def safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            if value is None:
                return default

            if isinstance(value, str):
                value = value.strip()

            return float(value)

        except Exception:
            return default


# ============================================================================
# RELIABILITY AUDITOR
# ============================================================================


class ReliabilityAuditor:
    @staticmethod
    def compute_deployability_score(
        issues: List[Dict],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = 100.0

        component_scores = {}

        critical_issues = [
            issue
            for issue in issues
            if Normalizer.normalize_severity(
                issue.get("severity")
            )
            == "critical"
        ]

        high_issues = [
            issue
            for issue in issues
            if Normalizer.normalize_severity(
                issue.get("severity")
            )
            == "high"
        ]

        # ======================================================
        # DATA INTEGRITY
        # ======================================================

        integrity_score = 100.0

        if any(
            "leakage" in issue.get("type", "")
            for issue in critical_issues
        ):
            integrity_score -= 40

        if any(
            "overlap" in issue.get("type", "")
            for issue in critical_issues
        ):
            integrity_score -= 35

        drift_issues = [
            issue
            for issue in critical_issues
            if "drift" in issue.get("type", "")
        ]

        if len(drift_issues) > 2:
            integrity_score -= 25

        if len(high_issues) > 5:
            integrity_score -= 15

        integrity_score = max(
            integrity_score,
            0,
        )

        component_scores["Data Integrity"] = {
            "score": integrity_score,
            "rationale": (
                f"{len(critical_issues)} critical, "
                f"{len(high_issues)} high issues"
            ),
        }

        # ======================================================
        # STABILITY
        # ======================================================

        stability_score = 100.0

        obs_flags = SchemaAdapter.get_observability_flags(
            result
        )

        overfit_flags = [
            flag
            for flag in obs_flags
            if "overfit"
            in str(
                flag.get(
                    "flag",
                    "",
                )
            ).lower()
        ]

        fold_flags = [
            flag
            for flag in obs_flags
            if "fold"
            in str(
                flag.get(
                    "flag",
                    "",
                )
            ).lower()
        ]

        decay_flags = [
            flag
            for flag in obs_flags
            if "decay"
            in str(
                flag.get(
                    "flag",
                    "",
                )
            ).lower()
        ]

        if overfit_flags:
            stability_score -= 20

        if fold_flags:
            stability_score -= 15

        if decay_flags:
            stability_score -= 25

        stability_score = max(
            stability_score,
            0,
        )

        component_scores["Statistical Stability"] = {
            "score": stability_score,
            "rationale": (
                f"{len(obs_flags)} stability warnings"
            ),
        }

        # ======================================================
        # FEATURE QUALITY
        # ======================================================

        feature_score = 100.0

        vif_issues = [
            issue
            for issue in issues
            if issue.get("type")
            == "multicollinearity"
        ]

        card_issues = [
            issue
            for issue in issues
            if issue.get("type")
            == "high_cardinality"
        ]

        if len(vif_issues) > 3:
            feature_score -= 20

        elif len(vif_issues) > 0:
            feature_score -= 10

        if len(card_issues) > 2:
            feature_score -= 15

        feature_score = max(
            feature_score,
            0,
        )

        component_scores["Feature Quality"] = {
            "score": feature_score,
            "rationale": (
                f"{len(vif_issues)} collinearity, "
                f"{len(card_issues)} cardinality risks"
            ),
        }

        # ======================================================
        # MODEL PERFORMANCE
        # ======================================================

        perf_score = 100.0

        holdout_metrics = (
            SchemaAdapter.get_holdout_metrics(
                result
            )
        )

        primary_metric_name = (
            ReliabilityAuditor.get_primary_metric_name(
                holdout_metrics
            )
        )

        primary_metric_value = (
            holdout_metrics.get(
                primary_metric_name,
                0.0,
            )
        )

        primary_metric_value = (
            Normalizer.safe_float(
                primary_metric_value
            )
        )

        if primary_metric_name in {
            "rmse",
            "mae",
        }:
            if primary_metric_value > 1.0:
                perf_score = 50
            elif primary_metric_value > 0.5:
                perf_score = 75
            else:
                perf_score = 95

        else:
            if primary_metric_value < 0.65:
                perf_score = 50
            elif primary_metric_value < 0.75:
                perf_score = 75
            else:
                perf_score = 95

        component_scores["Model Performance"] = {
            "score": perf_score,
            "rationale": (
                f"{primary_metric_name}: "
                f"{primary_metric_value:.3f}"
            ),
        }

        # ======================================================
        # FINAL SCORE
        # ======================================================

        score = (
            integrity_score * 0.35
            + stability_score * 0.35
            + feature_score * 0.20
            + perf_score * 0.10
        )

        return {
            "overall_score": round(
                score,
                1,
            ),
            "components": component_scores,
            "recommendation": (
                ReliabilityAuditor._get_recommendation(
                    score
                )
            ),
        }

    @staticmethod
    def get_primary_metric_name(
        holdout_metrics: Dict[str, Any],
    ) -> str:
        metric_priority = [
            "f1",
            "accuracy",
            "precision",
            "recall",
            "r2",
            "rmse",
            "mae",
        ]

        for metric_name in metric_priority:
            if metric_name in holdout_metrics:
                return metric_name

        return next(
            iter(holdout_metrics.keys()),
            "metric",
        )

    @staticmethod
    def _get_recommendation(
        score: float,
    ) -> Dict[str, str]:
        if score >= 85:
            return {
                "status": "✓ DEPLOYMENT READY",
                "color": "#27ae60",
                "message": (
                    "Pipeline passed reliability audit."
                ),
                "action": (
                    "Deploy with standard monitoring."
                ),
            }

        elif score >= 70:
            return {
                "status": "⚠ CONDITIONAL DEPLOYMENT",
                "color": "#f39c12",
                "message": (
                    "Pipeline has moderate concerns."
                ),
                "action": (
                    "Resolve high-severity issues."
                ),
            }

        elif score >= 50:
            return {
                "status": "✗ DEPLOYMENT BLOCKED",
                "color": "#e67e22",
                "message": (
                    "Critical reliability gaps detected."
                ),
                "action": (
                    "Investigate critical issues."
                ),
            }

        return {
            "status": "✗✗ SEVERE ISSUES",
            "color": "#c0392b",
            "message": (
                "Pipeline exhibits severe failures."
            ),
            "action": (
                "Rebuild validation workflow."
            ),
        }

    @staticmethod
    def get_score_color(
        score: float,
    ) -> str:
        if score >= 85:
            return "#27ae60"

        if score >= 70:
            return "#f39c12"

        if score >= 50:
            return "#e67e22"

        return "#c0392b"


# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================


class AdvancedVisualizationEngine:
    @staticmethod
    def create_drift_severity_heatmap(
        psi_scores: List[Dict],
    ) -> Optional[go.Figure]:
        if not psi_scores:
            return None

        try:
            df = pd.DataFrame(psi_scores)

            required_columns = {
                "Feature",
                "PSI Score",
                "Drift Severity",
            }

            if not required_columns.issubset(
                set(df.columns)
            ):
                return None

            colors_map = {
                "LOW": "#27ae60",
                "MEDIUM": "#f39c12",
                "HIGH": "#e67e22",
                "CRITICAL": "#c0392b",
            }

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df["Feature"],
                        y=df["PSI Score"],
                        marker=dict(
                            color=df[
                                "Drift Severity"
                            ].map(colors_map),
                        ),
                        text=df[
                            "PSI Score"
                        ].round(2),
                        textposition="outside",
                    )
                ]
            )

            fig.update_layout(
                height=350,
                xaxis={
                    "tickangle": -45
                },
                title=(
                    "Feature Drift Severity "
                    "(Population Stability Index)"
                ),
                yaxis_title="PSI Score",
                showlegend=False,
            )

            return fig

        except Exception:
            return None


# ============================================================================
# DASHBOARD STATE
# ============================================================================


class DashboardState:
    @staticmethod
    def get_session_state():
        if (
            "pipeline_result"
            not in st.session_state
        ):
            st.session_state.pipeline_result = None

        if (
            "run_timestamp"
            not in st.session_state
        ):
            st.session_state.run_timestamp = None

        return st.session_state


# ============================================================================
# SIDEBAR
# ============================================================================


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown(
        "### Audit Configuration"
    )

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Dataset (CSV)",
            type=["csv"],
        )

        target_column = st.text_input(
            "Target Variable",
            value="",
            placeholder="Column name",
        )

        col1, col2 = st.columns(2)

        with col1:
            task_type = st.selectbox(
                "Task",
                [
                    "classification",
                    "regression",
                ],
            )

        with col2:
            dev_mode = st.checkbox(
                "Fast Mode",
                value=False,
            )

        st.markdown("---")

        run_button = st.button(
            "Run Audit",
            use_container_width=True,
            type="primary",
        )

        return {
            "uploaded_file": uploaded_file,
            "target_column": target_column,
            "task_type": task_type,
            "dev_mode": dev_mode,
            "run_button": run_button,
        }


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================


def render_executive_summary(
    auditor: ReliabilityAuditor,
    result: Dict[str, Any],
    tab,
) -> None:
    with tab:
        issues = SchemaAdapter.get_issues(
            result
        )

        audit_result = (
            auditor.compute_deployability_score(
                issues,
                result,
            )
        )

        overall_score = audit_result[
            "overall_score"
        ]

        components = audit_result[
            "components"
        ]

        recommendation = audit_result[
            "recommendation"
        ]

        st.markdown(
            f"""
            <div class='deployment-banner'
            style='background-color:
            {recommendation["color"]}'>
            {recommendation["status"]}<br>
            <span style='font-size:0.9rem;
            font-weight:normal;'>
            {recommendation["message"]}
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(
            [1, 2]
        )

        with col1:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={
                        "text":
                        "Reliability Index"
                    },
                    gauge={
                        "axis": {
                            "range":
                            [0, 100]
                        },
                        "bar": {
                            "color":
                            ReliabilityAuditor.get_score_color(
                                overall_score
                            )
                        },
                    },
                )
            )

            fig.update_layout(
                height=250,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with col2:
            st.markdown(
                "**Reliability Components**"
            )

            for (
                component_name,
                component_data,
            ) in components.items():
                score = component_data[
                    "score"
                ]

                color = (
                    "green"
                    if score >= 80
                    else "orange"
                    if score >= 60
                    else "red"
                )

                st.markdown(
                    f"""
                    <div class='metric-card'>
                    <b>{component_name}</b>
                    <span style='color:{color};
                    font-weight:bold;'>
                    {score:.0f}/100
                    </span><br>
                    <span style='font-size:0.85rem;
                    color:#7f8c8d;'>
                    {component_data["rationale"]}
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        telemetry = (
            SchemaAdapter.get_telemetry(
                result
            )
        )

        telemetry_columns = st.columns(6)

        telemetry_metrics = [
            (
                "Total Runtime",
                "total_pipeline_seconds",
            ),
            (
                "Data Load",
                "data_loading_seconds",
            ),
            (
                "Integrity",
                "integrity_checks_seconds",
            ),
            (
                "Diagnostics",
                "diagnostics_seconds",
            ),
            (
                "Training",
                "training_seconds",
            ),
            (
                "Evaluation",
                "evaluation_seconds",
            ),
        ]

        for (
            column,
            metric_info,
        ) in zip(
            telemetry_columns,
            telemetry_metrics,
        ):
            label, key = metric_info

            value = Normalizer.safe_float(
                telemetry.get(key, 0.0)
            )

            column.metric(
                label,
                f"{value:.2f}s",
            )


# ============================================================================
# DATA QUALITY
# ============================================================================


def render_data_quality_deep_dive(
    result: Dict[str, Any],
    tab,
) -> None:
    with tab:
        issues = SchemaAdapter.get_issues(
            result
        )

        dataset_info = (
            SchemaAdapter.get_dataset(
                result
            )
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Rows",
            dataset_info.get("rows", 0),
        )

        col2.metric(
            "Columns",
            dataset_info.get(
                "columns",
                0,
            ),
        )

        overlap_pct = (
            Normalizer.safe_float(
                dataset_info.get(
                    "overlap_pct",
                    0.0,
                )
            )
        )

        col3.metric(
            "Overlap",
            f"{overlap_pct:.2f}%",
        )

        col4.metric(
            "Issues",
            len(issues),
        )

        st.markdown("---")

        issue_types: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        for issue in issues:
            issue_type = issue.get(
                "type",
                "unknown",
            )

            issue_types.setdefault(
                issue_type,
                [],
            ).append(issue)

        for (
            issue_type,
            issue_list,
        ) in sorted(
            issue_types.items()
        ):
            with st.expander(
                f"{issue_type.replace('_', ' ').title()} "
                f"({len(issue_list)})"
            ):
                for issue in issue_list:
                    severity = (
                        Normalizer.normalize_severity(
                            issue.get(
                                "severity"
                            )
                        )
                    )

                    st.markdown(
                        f"""
                        <div class='severity-{severity}'>
                        [{severity.upper()}]
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"""
                        **Column:** {issue.get("column", "dataset")}  
                        **Description:** {issue.get("description", "")}
                        """
                    )


# ============================================================================
# STABILITY
# ============================================================================


def render_statistical_stability(
    result: Dict[str, Any],
    tab,
) -> None:
    with tab:
        metrics = SchemaAdapter.get_metrics(
            result
        )

        obs_flags = (
            SchemaAdapter.get_observability_flags(
                result
            )
        )

        if obs_flags:
            st.warning(
                "Generalization issues detected."
            )

            for flag in obs_flags:
                st.markdown(
                    f"""
                    **{flag.get("flag", "Flag")}**  
                    {flag.get("detail", "")}
                    """
                )

        else:
            st.success(
                "Model exhibits stable generalization."
            )

        st.markdown("---")

        train_metrics = (
            SchemaAdapter.get_train_metrics(
                result
            )
        )

        holdout_metrics = (
            SchemaAdapter.get_holdout_metrics(
                result
            )
        )

        primary_metric = (
            ReliabilityAuditor.get_primary_metric_name(
                holdout_metrics
            )
        )

        train_val = (
            Normalizer.safe_float(
                train_metrics.get(
                    primary_metric,
                    0.0,
                )
            )
        )

        holdout_val = (
            Normalizer.safe_float(
                holdout_metrics.get(
                    primary_metric,
                    0.0,
                )
            )
        )

        gap = train_val - holdout_val

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Train",
            f"{train_val:.4f}",
        )

        col2.metric(
            "Holdout",
            f"{holdout_val:.4f}",
        )

        col3.metric(
            "Gap",
            f"{gap:.4f}",
        )

        st.markdown("---")

        metrics_df = pd.DataFrame(
            {
                "Metric":
                list(
                    holdout_metrics.keys()
                ),
                "Holdout":
                list(
                    holdout_metrics.values()
                ),
                "Train": [
                    train_metrics.get(
                        metric,
                        np.nan,
                    )
                    for metric
                    in holdout_metrics.keys()
                ],
            }
        )

        if not metrics_df.empty:
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================================
# FEATURE ANALYSIS
# ============================================================================


def render_feature_analysis(
    result: Dict[str, Any],
    tab,
) -> None:
    with tab:
        psi_data = (
            SchemaAdapter.get_psi_table(
                result
            )
        )

        vif_data = (
            SchemaAdapter.get_vif_table(
                result
            )
        )

        feature_audit = (
            SchemaAdapter.get_feature_audit(
                result
            )
        )

        feature_importance = (
            SchemaAdapter.get_feature_importance(
                result
            )
        )

        feat_tab1, feat_tab2, feat_tab3, feat_tab4 = st.tabs(
            [
                "Drift",
                "Multicollinearity",
                "Schema Audit",
                "Importance",
            ]
        )

        with feat_tab1:
            if psi_data:
                fig = (
                    AdvancedVisualizationEngine.create_drift_severity_heatmap(
                        psi_data
                    )
                )

                if fig:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                st.dataframe(
                    pd.DataFrame(psi_data),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "No drift analysis available."
                )

        with feat_tab2:
            if vif_data:
                st.dataframe(
                    pd.DataFrame(vif_data),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.success(
                    "No major multicollinearity detected."
                )

        with feat_tab3:
            if feature_audit:
                st.dataframe(
                    pd.DataFrame(
                        feature_audit
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "No schema audit available."
                )

        with feat_tab4:
            if feature_importance:
                sorted_importance = dict(
                    sorted(
                        feature_importance.items(),
                        key=lambda item:
                        item[1],
                        reverse=True,
                    )[:15]
                )

                fig = px.bar(
                    x=list(
                        sorted_importance.values()
                    ),
                    y=list(
                        sorted_importance.keys()
                    ),
                    orientation="h",
                )

                fig.update_layout(
                    height=450,
                    yaxis={
                        "categoryorder":
                        "total ascending"
                    },
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:
                st.info(
                    "No feature importance available."
                )


# ============================================================================
# RECOMMENDATIONS
# ============================================================================


def render_recommendations_audit(
    result: Dict[str, Any],
    tab,
) -> None:
    with tab:
        recommendations = (
            SchemaAdapter.get_recommendations(
                result
            )
        )

        if not recommendations:
            st.success(
                "No major recommendations."
            )
            return

        severity_filter = st.selectbox(
            "Severity Filter",
            [
                "All",
                "Critical",
                "High",
                "Medium",
                "Low",
            ],
        )

        filtered_recommendations = (
            recommendations
        )

        if severity_filter != "All":
            filtered_recommendations = [
                recommendation
                for recommendation
                in recommendations
                if Normalizer.normalize_severity(
                    recommendation.get(
                        "severity"
                    )
                )
                == severity_filter.lower()
            ]

        for (
            index,
            recommendation,
        ) in enumerate(
            filtered_recommendations,
            start=1,
        ):
            severity = (
                Normalizer.normalize_severity(
                    recommendation.get(
                        "severity"
                    )
                )
            )

            with st.expander(
                f"[{severity.upper()}] "
                f"Recommendation {index}"
            ):
                if (
                    "recommendation"
                    in recommendation
                ):
                    st.markdown(
                        recommendation[
                            "recommendation"
                        ]
                    )

                if (
                    "affected_columns"
                    in recommendation
                ):
                    st.markdown(
                        "**Affected Columns:**"
                    )

                    st.write(
                        recommendation[
                            "affected_columns"
                        ]
                    )

                if (
                    "issues"
                    in recommendation
                ):
                    st.markdown(
                        "**Supporting Issues:**"
                    )

                    for issue in recommendation[
                        "issues"
                    ]:
                        st.markdown(
                            f"- {issue.get('description', '')}"
                        )


# ============================================================================
# MAIN
# ============================================================================


def main():
    state = (
        DashboardState.get_session_state()
    )

    controls = render_sidebar()

    if (
        not controls["uploaded_file"]
        or not controls["target_column"]
    ):
        st.markdown(
            """
            # ML Reliability Platform

            Enterprise-grade ML audit and diagnostics system.

            ## Core Capabilities
            - leakage detection
            - drift analysis
            - multicollinearity analysis
            - stability diagnostics
            - feature auditing
            - deployment readiness scoring
            """
        )

        return

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv",
    ) as temp_file:
        temp_file.write(
            controls[
                "uploaded_file"
            ].getbuffer()
        )

        temp_file_path = temp_file.name

    if controls["run_button"]:
        with st.spinner(
            "Executing audit..."
        ):
            try:
                from app.pipeline.pipeline_runner import (
                    PipelineRunner,
                )

                runner = PipelineRunner(
                    file_path=temp_file_path,
                    target_column=controls[
                        "target_column"
                    ],
                    task_type=controls[
                        "task_type"
                    ],
                    dev_mode=controls[
                        "dev_mode"
                    ],
                )

                state.pipeline_result = (
                    runner.run()
                )

                state.run_timestamp = (
                    datetime.now()
                )

            except Exception as error:
                st.error(
                    f"Execution failed: {str(error)}"
                )

                st.exception(error)

                return

    if state.pipeline_result:
        result = state.pipeline_result

        if (
            result.get(
                "pipeline_status"
            )
            == "failure"
        ):
            st.error(
                result.get(
                    "error",
                    "Unknown pipeline failure.",
                )
            )

            return

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Executive Summary",
                "Data Quality",
                "Statistical Stability",
                "Feature Analysis",
                "Recommendations",
            ]
        )

        auditor = ReliabilityAuditor()

        render_executive_summary(
            auditor,
            result,
            tab1,
        )

        render_data_quality_deep_dive(
            result,
            tab2,
        )

        render_statistical_stability(
            result,
            tab3,
        )

        render_feature_analysis(
            result,
            tab4,
        )

        render_recommendations_audit(
            result,
            tab5,
        )

    try:
        if os.path.exists(
            temp_file_path
        ):
            os.unlink(
                temp_file_path
            )

    except Exception:
        pass


if __name__ == "__main__":
    main()