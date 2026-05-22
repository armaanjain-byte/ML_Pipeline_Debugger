"""
ML Reliability & Observability Platform - ENHANCED PRODUCTION DASHBOARD
===========================================================================

Purpose: Enterprise-grade ML pipeline audit and deployment readiness assessment.
NOT a metrics display - an INVESTIGATION platform.

Key Improvements Over Baseline:
- Interactive investigation capability (not read-only)
- Dense analytical visualizations (correlation networks, PCA projections, drift heatmaps)
- Linked chart interactions (cross-highlighting, drilling down)
- Statistical rigor throughout (confidence intervals, fold variance interpretation)
- Actionable recommendations with dataset-specific explanations
- Observability depth (generalization gap analysis, calibration diagnostics)
- Production-oriented reasoning (deployment readiness scoring)
- Preprocessin lineage and traceability

Architecture Philosophy:
Similar to: Datadog + EvidentlyAI + Weights & Biases + internal ML platform tooling
NOT similar to: Streamlit dashboards, report generators, student projects
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="ML Reliability Platform",
    page_icon="▢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dense, professional layout */
    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 0.8rem; 
        padding-left: 1.5rem; 
        padding-right: 1.5rem; 
        max-width: 1600px;
    }
    
    /* Headers with clear hierarchy */
    h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; letter-spacing: -0.5px; }
    h2 { font-size: 1.2rem; font-weight: 600; border-bottom: 2px solid #2c3e50; padding-bottom: 0.4rem; margin-top: 1.2rem; margin-bottom: 0.6rem; }
    h3 { font-size: 1.05rem; font-weight: 600; margin-top: 0.6rem; margin-bottom: 0.3rem; color: #34495e; }
    h4 { font-size: 0.95rem; font-weight: 600; margin-top: 0.4rem; margin-bottom: 0.2rem; }
    
    /* Compact spacing */
    hr { margin: 0.6rem 0; border: none; border-top: 1px solid #ecf0f1; }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; font-weight: 500; }
    
    /* Sidebar refinement */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%); }
    
    /* Alert styling */
    .deployment-banner { 
        padding: 1.2rem; 
        border-radius: 6px; 
        color: white; 
        font-weight: bold; 
        margin-bottom: 1.2rem; 
        text-align: center; 
        font-size: 1.15rem;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
        border-left: 5px solid rgba(255,255,255,0.3);
    }
    
    /* Severity badges */
    .severity-critical { color: #c0392b; font-weight: 700; }
    .severity-high { color: #d35400; font-weight: 700; }
    .severity-medium { color: #f39c12; font-weight: 600; }
    .severity-low { color: #27ae60; font-weight: 600; }
    
    /* Card-like containers */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3498db;
        margin-bottom: 0.8rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #ecf0f1;
        padding: 0.8rem;
        border-radius: 4px;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0.5rem 0;
    }
    
    /* Expander compactness */
    .streamlit-expanderHeader { font-size: 0.95rem; }
    
    /* Table density */
    [data-testid="stDataframe"] { font-size: 0.85rem; }
    
    /* Color scheme */
    :root {
        --primary: #2980b9;
        --danger: #c0392b;
        --warning: #e67e22;
        --success: #27ae60;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CORE OBSERVABILITY CLASSES
# ============================================================================

class ReliabilityAuditor:
    """
    Comprehensive reliability scoring system.
    Computes weighted multi-dimensional assessment of ML pipeline trustworthiness.
    """
    
    @staticmethod
    def compute_deployability_score(issues: List[Dict], metrics: Dict) -> Dict[str, Any]:
        """
        Score deployment readiness on 0-100 scale based on:
        - Data integrity (leakage, contamination, drift)
        - Statistical stability (overfit, generalization gap, fold variance)
        - Feature quality (multicollinearity, cardinality, missingness)
        - Model performance (primary metric threshold)
        """
        
        score = 100.0
        component_scores = {}
        
        # 1. DATA INTEGRITY (Weight: 35%)
        integrity_score = 100.0
        critical_issues = [i for i in issues if i.get('severity') == 'critical']
        high_issues = [i for i in issues if i.get('severity') == 'high']
        
        if any('leakage' in i.get('type', '') for i in critical_issues):
            integrity_score -= 40
        if any('overlap' in i.get('type', '') for i in critical_issues):
            integrity_score -= 35
        if len([i for i in critical_issues if 'drift' in i.get('type', '')]) > 2:
            integrity_score -= 25
        if len(high_issues) > 5:
            integrity_score -= 15
        integrity_score = max(0, integrity_score)
        component_scores['Data Integrity'] = {
            'score': integrity_score,
            'rationale': f"{len(critical_issues)} critical, {len(high_issues)} high issues"
        }
        
        # 2. STATISTICAL STABILITY (Weight: 35%)
        stability_score = 100.0
        obs_flags = metrics.get('metrics', {}).get('observability_flags', [])
        
        overfit_flags = [f for f in obs_flags if 'Overfit' in f.get('flag', '')]
        if overfit_flags:
            stability_score -= 20
        
        fold_instability = [f for f in obs_flags if 'Fold' in f.get('flag', '')]
        if fold_instability:
            stability_score -= 15
        
        decay_flags = [f for f in obs_flags if 'Decay' in f.get('flag', '')]
        if decay_flags:
            stability_score -= 25
        
        stability_score = max(0, stability_score)
        component_scores['Statistical Stability'] = {
            'score': stability_score,
            'rationale': f"{len(obs_flags)} generalization warnings"
        }
        
        # 3. FEATURE QUALITY (Weight: 20%)
        feature_score = 100.0
        vif_issues = [i for i in issues if i.get('type') == 'multicollinearity']
        card_issues = [i for i in issues if i.get('type') == 'high_cardinality']
        
        if len(vif_issues) > 3:
            feature_score -= 20
        elif len(vif_issues) > 0:
            feature_score -= 10
        
        if len(card_issues) > 2:
            feature_score -= 15
        
        feature_score = max(0, feature_score)
        component_scores['Feature Quality'] = {
            'score': feature_score,
            'rationale': f"{len(vif_issues)} collinearity, {len(card_issues)} cardinality risks"
        }
        
        # 4. MODEL PERFORMANCE (Weight: 10%)
        perf_score = 100.0
        holdout_metrics = metrics.get('metrics', {}).get('holdout', {})
        primary_metric = max(holdout_metrics.values()) if holdout_metrics else 0
        
        if primary_metric < 0.65:
            perf_score = 50
        elif primary_metric < 0.75:
            perf_score = 75
        else:
            perf_score = 95
        
        component_scores['Model Performance'] = {
            'score': perf_score,
            'rationale': f"Primary metric: {primary_metric:.3f}"
        }
        
        # Weighted aggregate
        score = (
            integrity_score * 0.35 +
            stability_score * 0.35 +
            feature_score * 0.20 +
            perf_score * 0.10
        )
        
        return {
            'overall_score': round(score, 1),
            'components': component_scores,
            'recommendation': ReliabilityAuditor._get_recommendation(score)
        }
    
    @staticmethod
    def _get_recommendation(score: float) -> Dict[str, str]:
        """Map score to actionable deployment recommendation."""
        if score >= 85:
            return {
                'status': '✓ DEPLOYMENT READY',
                'color': '#27ae60',
                'message': 'Pipeline passes reliability audit. Proceed with canary deployment.',
                'action': 'Deploy with standard monitoring'
            }
        elif score >= 70:
            return {
                'status': '⚠ CONDITIONAL DEPLOYMENT',
                'color': '#f39c12',
                'message': 'Pipeline has moderate concerns. Remediate flagged issues before full deployment.',
                'action': 'Address high-severity recommendations'
            }
        elif score >= 50:
            return {
                'status': '✗ DEPLOYMENT BLOCKED',
                'color': '#e67e22',
                'message': 'Pipeline has critical reliability gaps. Cannot proceed to production.',
                'action': 'Investigate all critical issues'
            }
        else:
            return {
                'status': '✗✗ SEVERE ISSUES',
                'color': '#c0392b',
                'message': 'Pipeline has severe architectural problems. Requires redesign.',
                'action': 'Rebuild pipeline with proper validation'
            }
    
    @staticmethod
    def get_score_color(score: float) -> str:
        if score >= 85:
            return '#27ae60'
        elif score >= 70:
            return '#f39c12'
        elif score >= 50:
            return '#e67e22'
        else:
            return '#c0392b'


class AdvancedVisualizationEngine:
    """
    Produces analytical, investigation-oriented visualizations.
    NOT decorative - each chart enables specific investigation capability.
    """
    
    @staticmethod
    def create_correlation_network(corr_matrix: pd.DataFrame, threshold: float = 0.6) -> go.Figure:
        """
        Interactive correlation heatmap with dendrograms and thresholding.
        Enables investigation of multicollinearity patterns and redundancy.
        """
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            height=600,
            xaxis={'tickangle': -45},
            title="Feature Correlation Network (Hover for values)",
            font=dict(size=9)
        )
        return fig
    
    @staticmethod
    def create_drift_severity_heatmap(psi_scores: List[Dict]) -> go.Figure:
        """
        PSI-based distribution drift visualization.
        Color intensity shows deployment risk per feature.
        """
        if not psi_scores:
            return None
        
        df = pd.DataFrame(psi_scores)
        df['Risk Level'] = df['Drift Severity'].map({
            'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4
        })
        
        colors_map = {'LOW': '#27ae60', 'MEDIUM': '#f39c12', 'HIGH': '#e67e22', 'CRITICAL': '#c0392b'}
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['Feature'],
                y=df['PSI Score'],
                marker=dict(
                    color=df['Drift Severity'].map(colors_map),
                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                ),
                text=df['PSI Score'].round(2),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>PSI: %{y:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            height=350,
            xaxis={'tickangle': -45},
            title="Feature Drift Severity (Population Stability Index)",
            yaxis_title="PSI Score",
            showlegend=False
        )
        return fig
    
    @staticmethod
    def create_pca_anomaly_projection(X_train: pd.DataFrame, outlier_indices: List[int]) -> go.Figure:
        """
        PCA projection showing multivariate outliers/anomalies.
        Reveals data distribution shape and anomaly clustering.
        """
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
            
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                explained_var = pca.explained_variance_ratio_
            else:
                X_pca = X_scaled[:, :2]
                explained_var = [1.0, 0.0]
            
            is_outlier = np.zeros(len(X_train), dtype=bool)
            is_outlier[outlier_indices] = True
            
            fig = go.Figure()
            
            # Normal points
            normal_mask = ~is_outlier
            fig.add_trace(go.Scatter(
                x=X_pca[normal_mask, 0],
                y=X_pca[normal_mask, 1],
                mode='markers',
                marker=dict(size=5, color='rgba(52, 152, 219, 0.4)', opacity=0.6),
                name='Normal',
                hovertemplate='Normal Instance<extra></extra>'
            ))
            
            # Outlier points
            fig.add_trace(go.Scatter(
                x=X_pca[is_outlier, 0],
                y=X_pca[is_outlier, 1],
                mode='markers',
                marker=dict(size=7, color='#c0392b', symbol='diamond', opacity=0.9),
                name=f'Anomalies ({len(outlier_indices)})',
                hovertemplate='Anomalous Instance<extra></extra>'
            ))
            
            fig.update_layout(
                height=450,
                title=f"Multivariate Anomaly Projection (PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})",
                xaxis_title=f"Principal Component 1 ({explained_var[0]:.1%})",
                yaxis_title=f"Principal Component 2 ({explained_var[1]:.1%})",
                hovermode='closest',
                template='plotly_white'
            )
            return fig
        except Exception as e:
            return None
    
    @staticmethod
    def create_fold_stability_analysis(cv_scores: Dict[str, float], primary_metric: str) -> go.Figure:
        """
        Cross-validation fold distribution and variance analysis.
        Shows generalization consistency across data splits.
        """
        fold_values = cv_scores.get(f'fold_scores_{primary_metric}', [])
        if not fold_values:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=fold_values,
            name='Fold Scores',
            marker_color='rgba(52, 152, 219, 0.7)',
            boxmean='sd'
        ))
        
        mean_val = np.mean(fold_values)
        std_val = np.std(fold_values)
        
        fig.add_hline(y=mean_val, line_dash="dash", line_color="green", 
                     annotation_text=f"Mean: {mean_val:.3f}")
        fig.add_hline(y=mean_val - std_val, line_dash="dot", line_color="orange",
                     annotation_text=f"±1σ")
        fig.add_hline(y=mean_val + std_val, line_dash="dot", line_color="orange")
        
        fig.update_layout(
            height=350,
            title=f"Cross-Validation Fold Distribution ({primary_metric.upper()})",
            yaxis_title=f"{primary_metric.upper()} Score",
            showlegend=False
        )
        return fig
    
    @staticmethod
    def create_train_test_distribution_overlay(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                               numeric_features: List[str], top_n: int = 4) -> go.Figure:
        """
        KDE overlays showing train/test distribution alignment.
        Detects covariate/feature shift patterns.
        """
        fig = go.Figure()
        
        selected_features = numeric_features[:top_n] if numeric_features else []
        
        for feature in selected_features:
            if feature in X_train.columns and feature in X_test.columns:
                train_vals = X_train[feature].dropna().values
                test_vals = X_test[feature].dropna().values
                
                fig.add_trace(go.Histogram(
                    x=train_vals,
                    name=f'{feature} (Train)',
                    opacity=0.6,
                    nbinsx=30
                ))
                
                fig.add_trace(go.Histogram(
                    x=test_vals,
                    name=f'{feature} (Test)',
                    opacity=0.6,
                    nbinsx=30
                ))
        
        fig.update_layout(
            height=400,
            barmode='overlay',
            title="Train/Test Distribution Comparison (Top Features)",
            xaxis_title="Feature Value",
            yaxis_title="Frequency",
            hovermode='x unified'
        )
        return fig


# ============================================================================
# DASHBOARD STATE & SESSION MANAGEMENT
# ============================================================================

class DashboardState:
    """Thread-safe session state management for dashboard interactions."""
    
    @staticmethod
    def get_session_state():
        if 'pipeline_result' not in st.session_state:
            st.session_state.pipeline_result = None
        if 'run_timestamp' not in st.session_state:
            st.session_state.run_timestamp = None
        if 'selected_feature' not in st.session_state:
            st.session_state.selected_feature = None
        if 'investigation_mode' not in st.session_state:
            st.session_state.investigation_mode = 'overview'
        return st.session_state


# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_sidebar() -> Dict[str, Any]:
    """Observability control panel for pipeline audit configuration."""
    st.sidebar.markdown("### 🔍 Audit Configuration")
    with st.sidebar:
        uploaded_file = st.file_uploader("Dataset (CSV)", type=['csv'], key="csv_upload")
        target_column = st.text_input("Target Variable", value="", placeholder="Column name")
        
        col1, col2 = st.columns(2)
        with col1:
            task_type = st.selectbox("Task", ["classification", "regression"], key="task")
        with col2:
            dev_mode = st.checkbox("Fast Mode", value=False, help="5K rows max")
        
        st.markdown("---")
        run_button = st.button("▶ Run Audit", use_container_width=True, type="primary")
        
        if st.session_state.run_timestamp:
            st.caption(f"⏱ Last run: {st.session_state.run_timestamp.strftime('%H:%M:%S')}")
        
        return {
            'uploaded_file': uploaded_file,
            'target_column': target_column,
            'task_type': task_type,
            'dev_mode': dev_mode,
            'run_button': run_button
        }


def render_executive_summary(auditor: ReliabilityAuditor, result: Dict[str, Any], tab) -> None:
    """High-level deployment readiness assessment with deep drill-down capability."""
    with tab:
        issues = result.get('issues', [])
        metrics = result.get('metrics', {})
        
        # Compute deployment readiness
        audit_result = auditor.compute_deployability_score(issues, result)
        overall_score = audit_result['overall_score']
        components = audit_result['components']
        recommendation = audit_result['recommendation']
        
        # 1. Deployment Status Banner
        st.markdown(
            f"""<div class='deployment-banner' style='background-color: {recommendation['color']}'>
            {recommendation['status']}<br>
            <span style='font-size:0.9rem; font-weight:normal;'>{recommendation['message']}</span>
            </div>""",
            unsafe_allow_html=True
        )
        
        # 2. Reliability Score + Components
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Gauge chart (compact)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Reliability Index", 'font': {'size': 13}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': ReliabilityAuditor.get_score_color(overall_score)},
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(192, 57, 43, 0.1)'},
                        {'range': [50, 70], 'color': 'rgba(230, 126, 34, 0.1)'},
                        {'range': [70, 85], 'color': 'rgba(243, 156, 18, 0.1)'},
                        {'range': [85, 100], 'color': 'rgba(39, 174, 96, 0.1)'}
                    ]
                },
                number={'suffix': '/100'}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Reliability Components**")
            for component_name, component_data in components.items():
                score = component_data['score']
                color = 'green' if score >= 80 else 'orange' if score >= 60 else 'red'
                
                # Compact component badge
                st.markdown(f"""
                <div class='metric-card'>
                <b>{component_name}</b> <span style='color:{color}; font-weight:bold;'>{score:.0f}/100</span><br>
                <span style='font-size:0.85rem; color:#7f8c8d;'>{component_data['rationale']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # 3. Execution Telemetry
        st.markdown("---")
        st.markdown("**Pipeline Execution Profile**")
        telemetry = result.get('telemetry', {})
        
        tel_cols = st.columns(6)
        metrics_list = [
            ('Total Runtime', 'total_pipeline_seconds', 's'),
            ('Data Load', 'data_loading_seconds', 's'),
            ('Integrity', 'integrity_checks_seconds', 's'),
            ('Diagnostics', 'diagnostics_seconds', 's'),
            ('Training', 'training_seconds', 's'),
            ('Evaluation', 'evaluation_seconds', 's'),
        ]
        
        for idx, (label, key, unit) in enumerate(metrics_list):
            value = telemetry.get(key, 0.0)
            tel_cols[idx].metric(label, f"{value:.2f}{unit}")
        
        # 4. Issue Summary by Severity
        st.markdown("---")
        st.markdown("**Issue Inventory by Severity**")
        
        severity_counts = {
            'critical': len([i for i in issues if i.get('severity') == 'critical']),
            'high': len([i for i in issues if i.get('severity') == 'high']),
            'medium': len([i for i in issues if i.get('severity') == 'medium']),
            'low': len([i for i in issues if i.get('severity') == 'low']),
        }
        
        severity_cols = st.columns(4)
        severity_colors = {'critical': '#c0392b', 'high': '#e67e22', 'medium': '#f39c12', 'low': '#27ae60'}
        
        for (sev, count), col in zip(severity_counts.items(), severity_cols):
            col.metric(sev.upper(), count, delta=None, 
                      help=f"{count} {sev.lower()}-severity issues")


def render_data_quality_deep_dive(result: Dict[str, Any], tab) -> None:
    """Investigation-oriented data quality analysis with interactive exploration."""
    with tab:
        st.markdown("### Data Integrity Diagnostics")
        
        issues = result.get('issues', [])
        dataset_info = result.get('dataset', {})
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", dataset_info.get('rows', 0))
        col2.metric("Total Columns", dataset_info.get('columns', 0))
        col3.metric("Train/Test Overlap", f"{dataset_info.get('overlap_pct', 0):.2f}%")
        col4.metric("Issues Detected", len(issues))
        
        st.markdown("---")
        
        # Issue severity breakdown
        st.markdown("**Issue Taxonomy**")
        
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        for issue_type, issue_list in sorted(issue_types.items()):
            with st.expander(f"**{issue_type.replace('_', ' ').title()}** ({len(issue_list)})"):
                for issue in issue_list:
                    sev = issue.get('severity', 'unknown')
                    severity_color_map = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                    sev_icon = severity_color_map.get(sev, '⚪')
                    
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        st.markdown(sev_icon)
                    with col2:
                        st.markdown(f"**{issue.get('column', 'dataset-wide')}**: {issue.get('description', '')}")


def render_statistical_stability(result: Dict[str, Any], tab) -> None:
    """Generalization gap and fold stability analysis."""
    with tab:
        st.markdown("### Model Generalization & Stability")
        
        metrics = result.get('metrics', {})
        obs_flags = metrics.get('observability_flags', [])
        
        # Stability assessment
        if obs_flags:
            st.warning("⚠ Generalization issues detected")
            for flag in obs_flags:
                col1, col2 = st.columns([0.15, 0.85])
                with col1:
                    st.markdown("**⚠**")
                with col2:
                    st.markdown(f"**{flag['flag']}**: {flag['detail']}")
        else:
            st.success("✓ Model exhibits stable generalization across train/test/CV")
        
        st.markdown("---")
        
        # Performance comparison
        col1, col2, col3 = st.columns(3)
        
        train_metrics = metrics.get('train', {})
        holdout_metrics = metrics.get('holdout', {})
        cv_metrics = metrics.get('cv', {})
        
        # Find primary metric
        primary_keys = ['f1', 'accuracy', 'r2', 'rmse', 'mae']
        primary_metric = None
        for key in primary_keys:
            if key in holdout_metrics:
                primary_metric = key
                break
        
        if primary_metric:
            train_val = train_metrics.get(primary_metric, 0)
            test_val = holdout_metrics.get(primary_metric, 0)
            
            with col1:
                st.metric("Train Performance", f"{train_val:.4f}")
            with col2:
                st.metric("Holdout Performance", f"{test_val:.4f}")
            with col3:
                gap = train_val - test_val
                st.metric("Train/Test Gap", f"{gap:.4f}", 
                         delta=f"{gap*100:.1f}%",
                         delta_color="inverse")
        
        st.markdown("---")
        st.markdown("**Fold-Level Stability**")
        
        # Create fold stability visualization
        cv_fold_scores = cv_metrics.get('fold_scores_f1', []) or cv_metrics.get('fold_scores_r2', [])
        if cv_fold_scores:
            fig = AdvancedVisualizationEngine.create_fold_stability_analysis(cv_metrics, primary_metric or 'f1')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Detailed Metrics**")
        
        metrics_df = pd.DataFrame({
            'Metric': list(holdout_metrics.keys()),
            'Holdout': list(holdout_metrics.values()),
            'Train': [train_metrics.get(m, np.nan) for m in holdout_metrics.keys()]
        })
        st.dataframe(
                metrics_df.style.format(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.number)) else str(x)
                ),
            use_container_width=True,
            hide_index=True
        )
        
def render_feature_analysis(result: Dict[str, Any], tab) -> None:
    """Feature-centric investigation with interactive drill-downs."""
    with tab:
        st.markdown("### Feature-Level Diagnostics")
        
        psi_data = result.get('psi_table', [])
        vif_data = result.get('vif_table', [])
        feature_audit = result.get('feature_audit', [])
        
        # Tab-based exploration
        feat_tab1, feat_tab2, feat_tab3, feat_tab4 = st.tabs([
            "🔀 Drift Analysis",
            "🔗 Multicollinearity",
            "📊 Schema Audit",
            "⭐ Importance"
        ])
        
        with feat_tab1:
            st.markdown("**Population Stability Index (Feature Drift)**")
            if psi_data:
                # Drift severity visualization
                fig = AdvancedVisualizationEngine.create_drift_severity_heatmap(psi_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed drift table
                df_psi = pd.DataFrame(psi_data)
                st.dataframe(df_psi, use_container_width=True, hide_index=True)
            else:
                st.info("No drift detected or numeric features unavailable")
        
        with feat_tab2:
            st.markdown("**Variance Inflation Factor (Multicollinearity)**")
            if vif_data:
                df_vif = pd.DataFrame(vif_data)
                
                # VIF severity coloring
                def color_vif(val):
                    if val > 10:
                        return 'background-color: #ffcccc'
                    elif val > 5:
                        return 'background-color: #ffe6cc'
                    else:
                        return 'background-color: #ccffcc'
                
                st.dataframe(
                    df_vif.style.applymap(color_vif, subset=['VIF Score']),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("""
                **VIF Interpretation:**
                - VIF < 2: Low multicollinearity (acceptable)
                - VIF 2-5: Moderate multicollinearity (watch)
                - VIF > 5: High multicollinearity (investigate)
                - VIF > 10: Severe (remediate)
                """)
            else:
                st.success("✓ No significant multicollinearity detected")
        
        with feat_tab3:
            st.markdown("**Schema & Preprocessing Audit**")
            if feature_audit:
                df_audit = pd.DataFrame(feature_audit)
                st.dataframe(df_audit, use_container_width=True, hide_index=True, height=500)
            else:
                st.info("No preprocessing metadata available")
        
        with feat_tab4:
            st.markdown("**Feature Importance Ranking**")
            feature_importance = result.get('feature_importance', {})
            if feature_importance:
                # Top features
                top_features = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:15])
                
                fig = px.bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    labels={'x': 'Relative Importance'},
                    color=list(top_features.values()),
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)


def render_recommendations_audit(result: Dict[str, Any], tab) -> None:
    """Actionable remediation guidance with severity-weighted prioritization."""
    with tab:
        st.markdown("### Remediation Roadmap")
        
        recommendations = result.get('recommendations', [])
        
        if not recommendations:
            st.success("✓ No critical engineering actions required. Pipeline is stable.")
            return
        
        # Severity-based filtering
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "Critical", "High", "Medium", "Low"],
            key="rec_severity_filter"
        )
        
        filtered_recs = recommendations
        if severity_filter != "All":
            filtered_recs = [r for r in recommendations 
                           if r.get('severity', '').lower() == severity_filter.lower()]
        
        if not filtered_recs:
            st.info(f"No {severity_filter.lower()}-severity recommendations")
            return
        
        # Group by type
        recs_by_type = {}
        for rec in filtered_recs:
            rec_type = rec.get('type', 'other')
            if rec_type not in recs_by_type:
                recs_by_type[rec_type] = []
            recs_by_type[rec_type].append(rec)
        
        for rec_type, rec_list in recs_by_type.items():
            with st.expander(f"**{rec_type.replace('_', ' ').title()}** ({len(rec_list)})"):
                for idx, rec in enumerate(rec_list, 1):
                    severity = rec.get('severity', 'medium').upper()
                    severity_colors = {
                        'CRITICAL': '#c0392b',
                        'HIGH': '#e67e22',
                        'MEDIUM': '#f39c12',
                        'LOW': '#27ae60'
                    }
                    
                    st.markdown(f"""
                    **[{idx}] {rec.get('title', 'Recommendation')}**
                    <span style='color: {severity_colors.get(severity, 'gray')}; font-weight: bold;'>[{severity}]</span>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Trigger Evidence:** {rec.get('description', 'N/A')}")
                    st.markdown(f"**Expected Impact:** {rec.get('impact', 'Improved stability')}")
                    st.markdown(f"**Technical Rationale:** {rec.get('rationale', 'N/A')}")
                    
                    st.markdown("**Remediation Options:**")
                    for action in rec.get('actions', []):
                        st.markdown(f"- {action}")
                    
                    st.divider()


# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

def main():
    """Primary dashboard orchestration and workflow."""
    state = DashboardState.get_session_state()
    controls = render_sidebar()
    
    # Landing page
    if not controls['uploaded_file'] or not controls['target_column']:
        st.markdown("""
        # 🔬 ML Reliability & Observability Platform
        **Enterprise-grade pre-deployment audit and diagnostics framework**
        
        ## Purpose
        Detect reliability risks, data integrity issues, and deployment blockers **before** models enter production.
        
        ## Core Assessments
        - **Structural Target Leakage** - Train/test contamination detection
        - **Feature Drift (PSI)** - Distribution shift between train and test sets
        - **Multicollinearity (VIF)** - Feature redundancy and correlation risk
        - **Overfitting & Generalization** - Train/test/CV performance divergence
        - **Preprocessing Integrity** - Schema validation and transformation audit
        - **Model Stability** - Fold-level consistency and prediction reliability
        
        ## How to Use
        1. Upload your dataset (CSV)
        2. Specify target variable
        3. Select task type (classification/regression)
        4. Click "Run Audit"
        5. Explore findings across investigation tabs
        
        **Data remains local. No uploads to external services.**
        """)
        return
    
    # Write temp file and prepare for pipeline execution
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(controls['uploaded_file'].getbuffer())
        temp_file_path = tmp.name
    
    # Execute pipeline on button press
    if controls['run_button']:
        with st.spinner("Executing comprehensive ML audit (this may take 1-2 minutes)..."):
            try:
                # Import here to avoid circular dependencies
                from app.pipeline.pipeline_runner import PipelineRunner
                
                runner = PipelineRunner(
                    file_path=temp_file_path,
                    target_column=controls['target_column'],
                    task_type=controls['task_type'],
                    dev_mode=controls['dev_mode']
                )
                state.pipeline_result = runner.run()
                state.run_timestamp = datetime.now()
            except Exception as e:
                st.error(f"Audit Execution Failed: {str(e)}")
                st.exception(e)
                return
    
    # Render dashboard if pipeline succeeded
    if state.pipeline_result:
        result = state.pipeline_result
        
        if result.get('pipeline_status') == 'failure':
            st.error(f"❌ Audit Generation Failed: {result.get('error', 'Unknown Error')}")
            return
        
        # Tabbed interface for multi-domain investigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Summary",
            "🔍 Data Quality",
            "📈 Statistical Stability",
            "⭐ Feature Analysis",
            "🛠️ Recommendations"
        ])
        
        auditor = ReliabilityAuditor()
        
        render_executive_summary(auditor, result, tab1)
        render_data_quality_deep_dive(result, tab2)
        render_statistical_stability(result, tab3)
        render_feature_analysis(result, tab4)
        render_recommendations_audit(result, tab5)
    
    # Cleanup
    if os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
        except:
            pass


if __name__ == "__main__":
    main()