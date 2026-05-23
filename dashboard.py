from __future__ import annotations

import html
import json
import os
import re
import tempfile
import warnings
from datetime import datetime
from html import unescape
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
    page_title="ML Pipeline Debugger",
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
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #f9fafc 100%);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e6ed;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 2000px;
        margin: 0 auto;
    }

    h1 {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
        color: #0f172a;
        line-height: 1.05;
    }

    h2 {
        font-size: 1.5rem;
        font-weight: 800;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.6rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
        color: #0f172a;
        letter-spacing: -0.3px;
    }

    h3 {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
        color: #0f172a;
        letter-spacing: -0.2px;
    }

    h4 {
        font-size: 0.98rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.1px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .header-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2.5rem 2.2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 2.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 0.4rem;
        color: #0f172a;
        letter-spacing: -1px;
    }

    .header-subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        line-height: 1.5;
    }

    .deployment-banner {
        padding: 1.6rem;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        margin-bottom: 1.6rem;
        text-align: center;
        font-size: 1.2rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.15);
        border-left: 6px solid rgba(255, 255, 255, 0.3);
    }

    .metric-card {
        background: #ffffff;
        padding: 1.1rem 1.2rem;
        border-radius: 11px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid #e2e8f0;
    }

    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transform: translateY(-3px);
        border-color: #cbd5e1;
    }

    .metric-card-label {
        font-weight: 800;
        font-size: 0.8rem;
        color: #1e293b;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card-value {
        font-size: 1.85rem;
        font-weight: 900;
        margin: 0.3rem 0;
        color: #0f172a;
        letter-spacing: -0.5px;
    }

    .metric-card-rationale {
        font-size: 0.82rem;
        color: #64748b;
        font-style: italic;
        margin-top: 0.4rem;
        line-height: 1.5;
    }

    .severity-critical {
        color: #7f1d1d;
        font-weight: 800;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        display: inline-block;
        font-size: 0.78rem;
        margin-bottom: 0;
        letter-spacing: 0.4px;
        border: 1.5px solid #fca5a5;
    }

    .severity-high {
        color: #7c2d12;
        font-weight: 800;
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        display: inline-block;
        font-size: 0.78rem;
        margin-bottom: 0;
        letter-spacing: 0.4px;
        border: 1.5px solid #fb923c;
    }

    .severity-medium {
        color: #78350f;
        font-weight: 800;
        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        display: inline-block;
        font-size: 0.78rem;
        margin-bottom: 0;
        letter-spacing: 0.4px;
        border: 1.5px solid #f59e0b;
    }

    .severity-low {
        color: #166534;
        font-weight: 800;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        display: inline-block;
        font-size: 0.78rem;
        margin-bottom: 0;
        letter-spacing: 0.4px;
        border: 1.5px solid #86efac;
    }

    .severity-info {
        color: #164e63;
        font-weight: 800;
        background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        display: inline-block;
        font-size: 0.78rem;
        margin-bottom: 0;
        letter-spacing: 0.4px;
        border: 1.5px solid #67e8f9;
    }

    .section-divider {
        margin: 1.5rem 0;
        border: none;
        border-top: 2px solid #e2e8f0;
    }

    .tab-content {
        padding: 1.2rem 0;
    }

    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 2px solid #e2e8f0;
        gap: 0.5rem;
    }

    [data-testid="stTabs"] [role="tab"] {
        padding: 0.75rem 1.4rem;
        font-weight: 700;
        font-size: 0.95rem;
        color: #64748b;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
        border-radius: 6px 6px 0 0;
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: #0f172a;
        background-color: #f1f5f9;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #1e40af;
        border-bottom-color: #3b82f6;
        font-weight: 800;
        background-color: #f0f4f8;
    }

    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        overflow: hidden;
    }

    [data-testid="stExpander"] [role="button"] {
        padding: 0.9rem 1.1rem;
        font-weight: 700;
        color: #1e293b;
        background-color: #f8fafc;
        border-radius: 9px;
        font-size: 0.93rem;
    }

    [data-testid="stExpander"] [role="button"]:hover {
        background-color: #f1f5f9;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    [data-testid="stDataFrame"] table {
        background-color: #ffffff;
    }

    [data-testid="stDataFrame"] th {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #0f172a;
        font-weight: 800;
        border-bottom: 2px solid #cbd5e1;
        padding: 0.95rem;
        text-align: left;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    [data-testid="stDataFrame"] td {
        padding: 0.85rem 0.95rem;
        border-bottom: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.92rem;
    }

    [data-testid="stDataFrame"] tr:hover {
        background-color: #f8fafc;
    }

    .plotly-container {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 1.4rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin: 1.2rem 0;
    }

    [data-testid="stPlotlyChart"] {
        background-color: transparent;
    }

    .column-container {
        background-color: transparent;
    }

    .stSuccess {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%) !important;
        color: #166534 !important;
        border: 1.5px solid #bbf7d0 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1) !important;
    }

    .stWarning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef9e7 100%) !important;
        color: #78350f !important;
        border: 1.5px solid #fcd34d !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.1) !important;
    }

    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
        color: #7f1d1d !important;
        border: 1.5px solid #fca5a5 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 6px rgba(220, 38, 38, 0.1) !important;
    }

    .stInfo {
        background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%) !important;
        color: #164e63 !important;
        border: 1.5px solid #67e8f9 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.1) !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.8rem !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 0.98rem !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    .sidebar-section {
        padding: 0.8rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 0.3rem;
    }

    .sidebar-section:last-child {
        border-bottom: none;
    }

    .sidebar-label {
        font-weight: 800;
        font-size: 0.8rem;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        display: block;
    }

    .component-spacing {
        margin-bottom: 1.5rem;
    }

    .reliability-gauge {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 1.4rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .component-grid {
        display: grid;
        gap: 1rem;
        margin-top: 1rem;
    }

    .recommendation-item {
        background-color: #ffffff;
        border-left: 6px solid #3b82f6;
        padding: 1.2rem 1.4rem;
        border-radius: 11px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    .recommendation-item:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }

    .recommendation-item.critical {
        border-left-color: #dc2626;
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
        border-color: #fecaca;
    }

    .recommendation-item.high {
        border-left-color: #ea580c;
        background: linear-gradient(135deg, #ffffff 0%, #fffbf0 100%);
        border-color: #fdba74;
    }

    .recommendation-item.medium {
        border-left-color: #eab308;
        background: linear-gradient(135deg, #ffffff 0%, #fffef5 100%);
        border-color: #fcd34d;
    }

    .recommendation-item.low {
        border-left-color: #16a34a;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        border-color: #bbf7d0;
    }

    .recommendation-item.info {
        border-left-color: #0ea5e9;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border-color: #a5f3fc;
    }

    .recommendation-title {
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0;
        font-size: 0.98rem;
        letter-spacing: -0.2px;
    }

    .recommendation-content {
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.6;
        margin: 0;
    }

    .recommendation-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
    }

    .recommendation-section-title {
        font-weight: 800;
        color: #0f172a;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 0.5rem;
    }

    .recommendation-section ul {
        margin: 0.4rem 0;
        padding-left: 1.6rem;
    }

    .recommendation-section li {
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.6;
        margin-bottom: 0.35rem;
    }

    .issue-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .issue-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }

    .issue-severity-badge {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: 0.3px;
    }

    .issue-column {
        font-family: 'Monaco', 'Courier New', monospace;
        background: #f1f5f9;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #1e293b;
    }

    .issue-description {
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.6;
        margin-top: 0.35rem;
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
                "status": "DEPLOYMENT READY",
                "color": "#10b981",
                "message": (
                    "Pipeline passed reliability audit."
                ),
                "action": (
                    "Deploy with standard monitoring."
                ),
            }

        elif score >= 70:
            return {
                "status": "CONDITIONAL DEPLOYMENT",
                "color": "#f59e0b",
                "message": (
                    "Pipeline has moderate concerns."
                ),
                "action": (
                    "Resolve high-severity issues."
                ),
            }

        elif score >= 50:
            return {
                "status": "DEPLOYMENT BLOCKED",
                "color": "#ef5350",
                "message": (
                    "Critical reliability gaps detected."
                ),
                "action": (
                    "Investigate critical issues."
                ),
            }

        return {
            "status": "SEVERE ISSUES DETECTED",
            "color": "#c62828",
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
            return "#10b981"

        if score >= 70:
            return "#f59e0b"

        if score >= 50:
            return "#ef5350"

        return "#c62828"


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
                "LOW": "#10b981",
                "MEDIUM": "#f59e0b",
                "HIGH": "#ef5350",
                "CRITICAL": "#c62828",
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
                            line=dict(
                                color="rgba(0,0,0,0.08)",
                                width=0.5,
                            ),
                        ),
                        text=df[
                            "PSI Score"
                        ].round(2),
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>PSI Score: %{y:.3f}<extra></extra>",
                    )
                ]
            )

            fig.update_layout(
                height=380,
                xaxis={
                    "tickangle": -45,
                    "showgrid": False,
                },
                yaxis={
                    "showgrid": True,
                    "gridwidth": 1,
                    "gridcolor": "rgba(0,0,0,0.03)",
                },
                title={
                    "text": "Feature Drift Severity (Population Stability Index)",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 13, "color": "#1e293b", "family": "sans-serif"},
                },
                yaxis_title="PSI Score",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "sans-serif", "color": "#475569"},
                margin={"b": 100, "l": 60, "r": 40, "t": 50},
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
        "<div class='sidebar-label'>Audit Configuration</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Dataset (CSV)",
            type=["csv"],
            help="Upload a CSV file for analysis",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        target_column = st.text_input(
            "Target Variable",
            value="",
            placeholder="Enter column name",
            help="The target variable for analysis",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            task_type = st.selectbox(
                "Task Type",
                [
                    "classification",
                    "regression",
                ],
                help="Select the ML task type",
            )

        with col2:
            dev_mode = st.checkbox(
                "Fast Mode",
                value=False,
                help="Run faster analysis",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        run_button = st.button(
            "Run Audit",
            use_container_width=True,
            type="primary",
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
            style='background: linear-gradient(135deg, {recommendation["color"]} 0%, {recommendation["color"]}dd 100%);'>
            <div style='font-size: 1.25rem; font-weight: 800; margin-bottom: 0.3rem;'>{recommendation["status"]}</div>
            <div style='font-size: 0.95rem; font-weight: 500; opacity: 0.95;'>
            {recommendation["message"]}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1.15, 1.85])

        with col1:
            st.markdown(
                "<div class='reliability-gauge'>",
                unsafe_allow_html=True,
            )
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={
                        "text": "Reliability Index",
                        "font": {"size": 15, "color": "#1e293b"},
                    },
                    number={"font": {"size": 32, "color": "#0f172a"}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#cbd5e1",
                        },
                        "bar": {
                            "color": ReliabilityAuditor.get_score_color(
                                overall_score
                            ),
                            "thickness": 0.28,
                        },
                        "bgcolor": "#f1f5f9",
                        "borderwidth": 0,
                        "steps": [
                            {
                                "range": [0, 50],
                                "color": "#fee2e2",
                            },
                            {
                                "range": [50, 70],
                                "color": "#fed7aa",
                            },
                            {
                                "range": [70, 85],
                                "color": "#fef3c7",
                            },
                            {
                                "range": [85, 100],
                                "color": "#dcfce7",
                            },
                        ],
                    },
                )
            )

            fig.update_layout(
                height=300,
                margin={"t": 20, "b": 10, "l": 10, "r": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                "<div style='margin-bottom: 0.8rem;'><h3 style='margin: 0; color: #0f172a;'>Reliability Components</h3></div>",
                unsafe_allow_html=True,
            )

            for (
                component_name,
                component_data,
            ) in components.items():
                score = component_data["score"]

                if score >= 80:
                    color = "#10b981"
                    bg_color = "#f0fdf4"
                elif score >= 60:
                    color = "#f59e0b"
                    bg_color = "#fffbf0"
                else:
                    color = "#dc2626"
                    bg_color = "#fee2e2"

                st.markdown(
                    f"""
                    <div class='metric-card' style='border-left-color: {color}; background-color: {bg_color};'>
                    <div class='metric-card-label' style='color: {color};'>{component_name}</div>
                    <div class='metric-card-value' style='color: {color};'>{score:.0f}/100</div>
                    <div class='metric-card-rationale'>{component_data["rationale"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='margin: 1.2rem 0 1rem 0; color: #0f172a; border: none; padding: 0;'>Execution Telemetry</h2>",
            unsafe_allow_html=True,
        )

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

        st.markdown(
            "<h2 style='margin: 0 0 1rem 0; color: #0f172a; border: none; padding: 0;'>Dataset Overview</h2>",
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Rows",
            f"{dataset_info.get('rows', 0):,}",
        )

        col2.metric(
            "Total Columns",
            f"{dataset_info.get('columns', 0)}",
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
            "Overlap %",
            f"{overlap_pct:.2f}%",
        )

        col4.metric(
            "Total Issues",
            f"{len(issues)}",
        )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='margin: 0 0 1rem 0; color: #0f172a; border: none; padding: 0;'>Issue Breakdown</h2>",
            unsafe_allow_html=True,
        )

        if not issues:
            st.success("No data quality issues detected.")
        else:
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
                    f"{issue_type.replace('_', ' ').title()} ({len(issue_list)})",
                    expanded=False,
                ):
                    for idx, issue in enumerate(issue_list, 1):
                        severity = (
                            Normalizer.normalize_severity(
                                issue.get(
                                    "severity"
                                )
                            )
                        )

                        column = issue.get(
                            "column",
                            "dataset",
                        )

                        description = issue.get(
                            "description",
                            "N/A",
                        )

                        st.markdown(
                            f"""
                            <div class='issue-card'>
                                <div class='issue-severity-badge severity-{severity}'>{severity.upper()}</div>
                                <div style='margin-top: 0.35rem;'>
                                    <span style='color: #64748b; font-size: 0.85rem;'>Column:</span>
                                    <span class='issue-column'>{column}</span>
                                </div>
                                <div class='issue-description'><strong>Description:</strong> {description}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
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
                f"Detected {len(obs_flags)} generalization concern(s)"
            )

            for idx, flag in enumerate(obs_flags, 1):
                with st.expander(
                    f"Concern {idx}: {flag.get('flag', 'Issue')}",
                    expanded=False,
                ):
                    st.markdown(
                        f"{flag.get('detail', 'No additional details.')}"
                    )

        else:
            st.success(
                "Model exhibits stable generalization across all checks."
            )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='margin: 0 0 1rem 0; color: #0f172a; border: none; padding: 0;'>Metric Comparison</h2>",
            unsafe_allow_html=True,
        )

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

        gap = abs(train_val - holdout_val)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            f"Train {primary_metric.upper()}",
            f"{train_val:.4f}",
        )

        col2.metric(
            f"Holdout {primary_metric.upper()}",
            f"{holdout_val:.4f}",
        )

        gap_status = "OK" if gap < 0.1 else "WARNING"
        col3.metric(
            "Train-Holdout Gap",
            f"{gap:.4f}",
            delta=gap_status,
        )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='margin: 0 0 1rem 0; color: #0f172a; border: none; padding: 0;'>All Metrics</h2>",
            unsafe_allow_html=True,
        )

        metrics_df = pd.DataFrame(
            {
                "Metric": list(
                    holdout_metrics.keys()
                ),
                "Holdout": [
                    f"{Normalizer.safe_float(v):.4f}"
                    for v in holdout_metrics.values()
                ],
                "Train": [
                    f"{Normalizer.safe_float(train_metrics.get(metric, 0.0)):.4f}"
                    for metric in holdout_metrics.keys()
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
                "Drift Analysis",
                "Multicollinearity",
                "Schema Audit",
                "Feature Importance",
            ]
        )

        with feat_tab1:
            if psi_data:
                st.markdown(
                    "<h3 style='margin: 0 0 1rem 0; color: #0f172a;'>Population Stability Index</h3>",
                    unsafe_allow_html=True,
                )

                fig = (
                    AdvancedVisualizationEngine.create_drift_severity_heatmap(
                        psi_data
                    )
                )

                if fig:
                    st.markdown(
                        "<div class='plotly-container'>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    "<h3 style='margin: 1.4rem 0 0.8rem 0; color: #0f172a;'>Detailed Results</h3>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(psi_data),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "No drift analysis data available."
                )

        with feat_tab2:
            if vif_data:
                st.markdown(
                    "<h3 style='margin: 0 0 1rem 0; color: #0f172a;'>Variance Inflation Factor (VIF)</h3>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(vif_data),
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    "VIF > 10 typically indicates high multicollinearity. Consider feature engineering or selection."
                )

            else:
                st.success(
                    "No major multicollinearity issues detected."
                )

        with feat_tab3:
            if feature_audit:
                st.markdown(
                    "<h3 style='margin: 0 0 1rem 0; color: #0f172a;'>Feature Schema Audit</h3>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(
                        feature_audit
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "No schema audit data available."
                )

        with feat_tab4:
            if feature_importance:
                st.markdown(
                    "<h3 style='margin: 0 0 1rem 0; color: #0f172a;'>Top 15 Features by Importance</h3>",
                    unsafe_allow_html=True,
                )

                sorted_importance = dict(
                    sorted(
                        feature_importance.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:15]
                )

                fig = px.bar(
                    x=list(sorted_importance.values()),
                    y=list(sorted_importance.keys()),
                    orientation="h",
                    labels={
                        "x": "Importance Score",
                        "y": "Feature",
                    },
                )

                fig.update_traces(
                    marker=dict(
                        color="#3b82f6",
                        line=dict(
                            color="rgba(0,0,0,0.08)",
                            width=0.5,
                        ),
                    )
                )

                fig.update_layout(
                    height=450,
                    yaxis={"categoryorder": "total ascending"},
                    xaxis={
                        "showgrid": True,
                        "gridwidth": 1,
                        "gridcolor": "rgba(0,0,0,0.03)",
                    },
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"family": "sans-serif", "color": "#475569"},
                    margin={"l": 150, "b": 40, "t": 20, "r": 20},
                )

                st.markdown(
                    "<div class='plotly-container'>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.info(
                    "No feature importance data available."
                )


# ============================================================================
# RECOMMENDATIONS
# ============================================================================


def _safe_render_content(content: str) -> str:
    """
    Clean malformed recommendation payloads.

    Handles:
    1. Plain text recommendations
    2. Legacy HTML recommendation blobs
    3. Escaped HTML payloads
    4. Corrupted nested recommendation cards
    """

    if content is None:
        return "No recommendation provided."

    content = str(content)

    # Decode escaped HTML entities
    content = html.unescape(content)

    # Remove markdown fences
    content = re.sub(
        r"```(?:html)?",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = content.replace(
        "```",
        "",
    )

    # =====================================================
    # Legacy full-card HTML extraction
    # =====================================================

    if (
        "recommendation-content" in content
        or "recommendation-item" in content
    ):

        match = re.search(
            r"<div[^>]*class=['\"]recommendation-content['\"][^>]*>(.*?)</div>",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match:
            content = match.group(1).strip()

        else:
            content = re.sub(
                r"<[^>]+>",
                " ",
                content,
            )

    # Remove remaining HTML tags
    content = re.sub(
        r"<[^>]+>",
        " ",
        content,
    )

    # Normalize whitespace
    content = re.sub(
        r"\s+",
        " ",
        content,
    ).strip()

    if not content:
        return "No recommendation provided."

    return content


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
                "No recommendations at this time. Pipeline is in good shape."
            )
            return

        # =====================================================
        # Severity Counts
        # =====================================================

        severity_counts = {}

        for recommendation in recommendations:

            severity = (
                Normalizer.normalize_severity(
                    recommendation.get(
                        "severity"
                    )
                )
            )

            severity_counts[severity] = (
                severity_counts.get(
                    severity,
                    0,
                )
                + 1
            )

        metric_cols = st.columns(5)

        metric_cols[0].metric(
            "Critical",
            severity_counts.get(
                "critical",
                0,
            ),
        )

        metric_cols[1].metric(
            "High",
            severity_counts.get(
                "high",
                0,
            ),
        )

        metric_cols[2].metric(
            "Medium",
            severity_counts.get(
                "medium",
                0,
            ),
        )

        metric_cols[3].metric(
            "Low",
            severity_counts.get(
                "low",
                0,
            ),
        )

        metric_cols[4].metric(
            "Info",
            severity_counts.get(
                "info",
                0,
            ),
        )

        st.markdown(
            "<div class='section-divider'></div>",
            unsafe_allow_html=True,
        )

        # =====================================================
        # Filter
        # =====================================================

        severity_filter = st.selectbox(
            "Filter by Severity",
            [
                "All",
                "Critical",
                "High",
                "Medium",
                "Low",
                "Info",
            ],
        )

        filtered_recommendations = (
            recommendations
        )

        if severity_filter != "All":

            filtered_recommendations = [
                recommendation
                for recommendation in recommendations
                if (
                    Normalizer.normalize_severity(
                        recommendation.get(
                            "severity"
                        )
                    )
                    == severity_filter.lower()
                )
            ]

        st.markdown(
            f"""
            <h2 style='margin-top: 1.5rem;'>
                Recommendations ({len(filtered_recommendations)})
            </h2>
            """,
            unsafe_allow_html=True,
        )

        # =====================================================
        # Render Cards
        # =====================================================

        for (
            idx,
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

            raw_payload = recommendation.get(
                "recommendation",
                "No recommendation available.",
            )

            recommendation_text = _safe_render_content(
                raw_payload
            )

            if (
                recommendation_text.strip().startswith("<div")
                or "recommendation-item" in recommendation_text
            ):
                recommendation_text = re.sub(
                r"<[^>]+>",
                " ",
                recommendation_text,
                )

                recommendation_text = re.sub(
                    r"\s+",
                    " ",
                    recommendation_text,
                ).strip()

            affected_columns = (
                recommendation.get(
                    "affected_columns",
                    [],
                )
            )

            issues = recommendation.get(
                "issues",
                [],
            )

            severity_html = (
                f"<span class='severity-{severity}'>"
                f"{severity.upper()}"
                f"</span>"
            )

            card_html = f"""
            <div class='recommendation-item {severity}'>

                <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;'>

                    <div class='recommendation-title'>
                        Recommendation {idx}
                    </div>

                    {severity_html}

                </div>

                <div class='recommendation-content'>
                    {recommendation_text}
                </div>
            """

            # =================================================
            # Affected Columns
            # =================================================

            if affected_columns:

                columns_html = "".join(
                    [
                        f'''
                        <span
                            style="
                                display:inline-block;
                                background:#e2e8f0;
                                color:#0f172a;
                                padding:0.3rem 0.7rem;
                                border-radius:999px;
                                margin-right:0.5rem;
                                margin-top:0.5rem;
                                font-size:0.82rem;
                                font-weight:700;
                            "
                        >
                            {column}
                        </span>
                        '''
                        for column in affected_columns
                    ]
                )

                card_html += f"""
                <div class='recommendation-section'>

                    <div class='recommendation-section-title'>
                        Affected Columns
                    </div>

                    <div>
                        {columns_html}
                    </div>

                </div>
                """

            # =================================================
            # Supporting Issues
            # =================================================

            if issues:

                issues_html = ""

                for issue in issues:

                    issues_html += f"""
                    <li style='margin-bottom:0.45rem;'>
                        {issue.get("description", "Issue detected.")}
                    </li>
                    """

                card_html += f"""
                <div class='recommendation-section'>

                    <div class='recommendation-section-title'>
                        Supporting Issues
                    </div>

                    <ul style='padding-left:1.5rem;margin-top:0.7rem;'>
                        {issues_html}
                    </ul>

                </div>
                """

            card_html += "</div>"

            st.markdown(
                card_html,
                unsafe_allow_html=True,
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
            <div class='header-section'>
            <div class='header-title'>ML Reliability Platform</div>
            <div class='header-subtitle'>Enterprise-grade ML audit and diagnostics system</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("## Core Capabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **Data Integrity Diagnostics**
                - Data leakage detection
                - Training-test set overlap analysis
                - Population drift monitoring
                """
            )

        with col2:
            st.markdown(
                """
                **Model Reliability Auditing**
                - Stability and overfitting diagnostics
                - Multicollinearity analysis
                - Feature cardinality assessment
                """
            )

        st.markdown("---")

        st.markdown(
            """
            **Getting Started:**

            1. Upload your CSV dataset using the sidebar
            2. Specify the target variable column
            3. Select your task type (classification or regression)
            4. Click "Run Audit" to begin analysis
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
            "Executing audit pipeline..."
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
                    f"Audit execution failed: {str(error)}"
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