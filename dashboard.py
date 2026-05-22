"""
ML Pipeline Debugger Dashboard - Stabilized Edition
Professional MLOps observability and diagnostics platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional

import plotly.graph_objects as go
import plotly.express as px

from app.pipeline.pipeline_runner import PipelineRunner
from app.utils.feature_utils import FeatureNameCleaner
from app.utils.observability import ReliabilityScorer, PipelineObserver

# Page Configuration
st.set_page_config(
    page_title="ML Pipeline Debugger",
    page_icon="▢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional minimalist styling
st.markdown("""
<style>
    /* Reduce excessive whitespace */
    .main { padding-top: 1rem; }
    .block-container { padding: 1.5rem 2rem; }
    
    /* Professional color scheme */
    :root {
        --primary: #2c3e50;
        --secondary: #34495e;
        --border: #bdc3c7;
        --bg-light: #ecf0f1;
        --critical: #c0392b;
        --high: #e67e22;
        --medium: #f39c12;
        --low: #27ae60;
    }
    
    /* Section styling - cleaner typography */
    h1, h2, h3 { margin-top: 1.5rem; margin-bottom: 0.75rem; color: #2c3e50; }
    h1 { font-size: 1.8rem; font-weight: 600; }
    h2 { font-size: 1.4rem; font-weight: 600; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.5rem; }
    h3 { font-size: 1.1rem; font-weight: 600; }
    
    /* Compact metric cards */
    .metric-card {
        background: white;
        border: 1px solid #bdc3c7;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    
    /* Severity indicators - no emojis, subtle */
    .severity-critical { color: #c0392b; font-weight: 600; }
    .severity-high { color: #e67e22; font-weight: 600; }
    .severity-medium { color: #f39c12; font-weight: 600; }
    .severity-low { color: #27ae60; font-weight: 600; }
    
    /* Reduce dividers */
    hr { margin: 1rem 0; border: 1px solid #ecf0f1; }
    
    /* Tighter sidebar */
    [data-testid="stSidebar"] {
        width: 280px;
        background: #f8f9fa;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Compact form elements */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        padding: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Table styling */
    .dataframe { font-size: 0.9rem; }
    
    /* Reduce Plotly margins */
    .plotly-graph-div { margin: 0; }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manages dashboard state across reruns."""
    
    @staticmethod
    def get_session_state():
        if 'pipeline_result' not in st.session_state:
            st.session_state.pipeline_result = None
        if 'run_timestamp' not in st.session_state:
            st.session_state.run_timestamp = None
        return st.session_state


def render_sidebar() -> Dict[str, Any]:
    """Render compact sidebar with pipeline controls."""
    st.sidebar.markdown("### ML Pipeline Debugger")
    
    with st.sidebar:
        st.markdown("Configuration")
        
        uploaded_file = st.file_uploader("CSV Dataset", type=['csv'], key="csv_upload")
        target_column = st.text_input("Target Column", value="", placeholder="e.g., churn")
        
        col1, col2 = st.columns(2)
        with col1:
            task_type = st.selectbox("Task", ["classification", "regression"], key="task")
        with col2:
            dev_mode = st.checkbox("Dev Mode", value=False, help="5K rows")
        
        st.markdown("---")
        run_button = st.button("Run Analysis", use_container_width=True, type="primary")
        
        if st.session_state.run_timestamp:
            ts = st.session_state.run_timestamp.strftime('%H:%M:%S')
            st.caption(f"Last run: {ts}")
        
        return {
            'uploaded_file': uploaded_file,
            'target_column': target_column,
            'task_type': task_type,
            'dev_mode': dev_mode,
            'run_button': run_button
        }


def render_header(metadata: Dict[str, Any]) -> None:
    """Render compact header metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{metadata['num_rows']:,}")
    with col2:
        st.metric("Features", metadata['num_columns'])
    with col3:
        memory_mb = metadata['memory_usage_mb']
        st.metric("Memory", f"{memory_mb:.1f} MB")
    with col4:
        numeric_cols = sum(1 for dt in metadata['dtypes'].values() if 'int' in dt or 'float' in dt)
        st.metric("Numeric", numeric_cols)


def render_reliability_score(scorer: ReliabilityScorer, checks: Dict[str, Any]) -> None:
    """Render reliability score with clear breakdown."""
    st.markdown("## Reliability Assessment")
    
    overall_score = scorer.compute_overall_score(checks)
    component_scores = scorer.compute_component_scores(checks)
    
    # Main score with interpretation
    status_text = scorer.get_health_status(overall_score)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': scorer.get_score_color(overall_score)},
                'steps': [
                    {'range': [0, 30], 'color': '#ffebee'},
                    {'range': [30, 50], 'color': '#fff3e0'},
                    {'range': [50, 70], 'color': '#fffde7'},
                    {'range': [70, 85], 'color': '#e8f5e9'},
                    {'range': [85, 100], 'color': '#c8e6c9'}
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"**Score:** {overall_score:.1f}")
        st.markdown(f"**Status:** {status_text}")
        st.markdown("**Components:**")
        for component, weight in [
            ('Data Quality', 0.25),
            ('Leakage Risk', 0.30),
            ('Class Balance', 0.20),
            ('Outlier Health', 0.15),
            ('Missing Data', 0.10)
        ]:
            score = component_scores.get(component.lower().replace(' ', '_'), 0)
            st.text(f"{component:15} {score:5.1f}%")


def render_issues_section(checks: Dict[str, Any]) -> None:
    """Render data quality issues in compact format."""
    st.markdown("## Data Quality Issues")
    
    issues = checks.get('issues', [])
    
    if not issues:
        st.info("No data quality issues detected.")
        return
    
    # Severity summary
    critical = sum(1 for i in issues if i.get('severity') == 'critical')
    high = sum(1 for i in issues if i.get('severity') == 'high')
    medium = sum(1 for i in issues if i.get('severity') == 'medium')
    low = sum(1 for i in issues if i.get('severity') == 'low')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Critical", critical)
    with col2:
        st.metric("High", high)
    with col3:
        st.metric("Medium", medium)
    with col4:
        st.metric("Low", low)
    
    st.markdown("---")
    
    # Issues table
    if critical > 0 or high > 0:
        st.markdown("### Actionable Issues")
        critical_high = [i for i in issues if i.get('severity') in ['critical', 'high']]
        
        for issue in critical_high:
            severity = issue.get('severity', 'medium').upper()
            col1, col2, col3 = st.columns([2, 1.5, 1])
            
            with col1:
                st.markdown(f"**{issue['type'].replace('_', ' ').title()}**")
                st.caption(issue['description'])
            with col2:
                st.caption(f"Column: `{issue['column']}`")
            with col3:
                if severity == 'CRITICAL':
                    st.markdown('<span class="severity-critical">CRITICAL</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="severity-high">HIGH</span>', unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Medium/Low issues collapsible
    other_issues = [i for i in issues if i.get('severity') not in ['critical', 'high']]
    if other_issues:
        with st.expander(f"Other Issues ({len(other_issues)})"):
            for issue in other_issues:
                severity = issue.get('severity', 'low').upper()
                st.markdown(f"**{issue['type'].replace('_', ' ')}** - {issue['description']}")
                st.caption(f"Column: `{issue['column']}`")
                st.markdown("---")


def render_feature_importance(importance_dict: Dict[str, float]) -> None:
    """Render top features with clean design."""
    st.markdown("## Feature Importance")
    
    if not importance_dict:
        st.info("Feature importance not available for this model type.")
        return
    
    cleaner = FeatureNameCleaner()
    top_n = 15
    top_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    if not top_features:
        st.info("No features to display.")
        return
    
    display_names = [cleaner.clean_feature_name(name) for name in top_features.keys()]
    importances = list(top_features.values())
    
    fig = go.Figure(data=[go.Bar(
        y=display_names,
        x=importances,
        orientation='h',
        marker=dict(color=importances, colorscale='Blues', showscale=False),
        text=[f'{imp:.4f}' for imp in importances],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.6f}<extra></extra>'
    )])
    
    fig.update_layout(
        height=300,
        margin=dict(l=200, r=100, t=20, b=20),
        xaxis_title="Importance",
        yaxis_title=None,
        showlegend=False
    )
    fig.update_yaxes(autorange="reversed")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 summary
    st.markdown("**Top Features:**")
    for idx, (feat, imp) in enumerate(list(top_features.items())[:5], 1):
        clean_name = cleaner.clean_feature_name(feat)
        st.text(f"{idx}. {clean_name:40} {imp:.6f}")


def render_model_performance(metrics: Dict[str, float], task_type: str) -> None:
    """Render model metrics in compact table."""
    st.markdown("## Model Performance")
    
    if not metrics:
        st.info("Model metrics not available.")
        return
    
    # Separate holdout and cross-validation metrics
    holdout = {k: v for k, v in metrics.items() if not k.startswith('cv_')}
    cv = {k: v for k, v in metrics.items() if k.startswith('cv_')}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Holdout Set**")
        for metric, value in sorted(holdout.items()):
            st.text(f"{metric:20} {value:.6f}")
    
    with col2:
        st.markdown("**Cross-Validation**")
        for metric, value in sorted(cv.items()):
            st.text(f"{metric:20} {value:.6f}")


def _validate_recommendations(recommendations: Any) -> Dict[str, Any]:
    """
    Validate and standardize recommendation structure.
    Handles mixed-type returns and edge cases.
    
    Returns canonical schema:
    {
        "recommendations": [...],
        "total_issues": int,
        "critical_issues": int,
        "severity_breakdown": {...}
    }
    """
    # Handle None/empty
    if not recommendations:
        return {
            "recommendations": [],
            "total_issues": 0,
            "critical_issues": 0,
            "severity_breakdown": {}
        }
    
    # If it's a list, assume it came from somewhere unexpected
    if isinstance(recommendations, list):
        return {
            "recommendations": recommendations,
            "total_issues": len(recommendations),
            "critical_issues": sum(1 for r in recommendations if r.get('severity') == 'critical'),
            "severity_breakdown": {}
        }
    
    # If it's a dict, validate structure
    if isinstance(recommendations, dict):
        # Ensure recommendations key exists and is a list
        recs = recommendations.get('recommendations', [])
        if not isinstance(recs, list):
            recs = []
        
        return {
            "recommendations": recs,
            "total_issues": recommendations.get('total_issues', len(recs)),
            "critical_issues": recommendations.get('critical_issues', sum(1 for r in recs if r.get('severity') == 'critical')),
            "severity_breakdown": recommendations.get('severity_breakdown', {})
        }
    
    # Fallback for unexpected types
    return {
        "recommendations": [],
        "total_issues": 0,
        "critical_issues": 0,
        "severity_breakdown": {}
    }


def render_recommendations(recommendations_input: Any) -> None:
    """Render actionable recommendations with robust validation."""
    st.markdown("## Recommendations")
    
    # Validate and standardize input
    recommendations = _validate_recommendations(recommendations_input)
    recs = recommendations.get('recommendations', [])
    total_issues = recommendations.get('total_issues', 0)
    critical_issues = recommendations.get('critical_issues', 0)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Issues", total_issues)
    with col2:
        status = "Critical" if critical_issues > 0 else "Healthy"
        st.metric("Critical", critical_issues)
    with col3:
        st.metric("Recommendations", len(recs))
    
    if not recs:
        st.info("No recommendations at this time.")
        return
    
    st.markdown("---")
    
    # Group by type for organization
    grouped = {}
    for rec in recs:
        issue_type = rec.get('type', 'unknown').replace('_', ' ').title()
        if issue_type not in grouped:
            grouped[issue_type] = []
        grouped[issue_type].append(rec)
    
    # Render grouped recommendations
    for issue_type, type_recs in grouped.items():
        with st.expander(f"{issue_type} ({len(type_recs)})"):
            for rec in type_recs:
                # Severity indicator
                severity = rec.get('severity', 'medium').upper()
                severity_class = f'severity-{rec.get("severity", "medium")}'
                
                # Header with severity
                st.markdown(f"**{rec.get('title', issue_type)}**")
                st.markdown(f'<span class="{severity_class}">{severity}</span>', unsafe_allow_html=True)
                st.caption(rec.get('description', ''))
                
                # Actions
                actions = rec.get('recommendations', rec.get('actions', []))
                if actions:
                    st.markdown("**Actions:**")
                    for action in actions:
                        st.text(f"• {action}")
                
                # Rationale
                rationale = rec.get('rationale', '')
                if rationale:
                    with st.expander("Rationale"):
                        st.text(rationale)
                
                st.markdown("---")


def render_data_sample(df: pd.DataFrame) -> None:
    """Render data sample and statistics."""
    st.markdown("## Data Sample")
    
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.slider("Rows", 5, 50, 10, key="sample_slider")
    with col2:
        show_stats = st.checkbox("Show statistics", key="show_stats")
    
    st.dataframe(df.head(n_rows), use_container_width=True, height=300)
    
    if show_stats:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe(), use_container_width=True)


def main():
    """Main dashboard application."""
    state = DashboardState.get_session_state()
    controls = render_sidebar()
    
    # Check for required inputs
    if not controls['uploaded_file'] or not controls['target_column']:
        st.markdown("""
        # ML Pipeline Debugger
        
        Upload a CSV dataset to analyze data quality, detect issues, and get actionable recommendations.
        
        **Features:**
        • Comprehensive data quality diagnostics
        • Reliability scoring (0-100 scale)
        • Feature importance analysis
        • Model performance evaluation
        • Actionable recommendations
        
        **Getting started:** Use the sidebar to upload your data and configure analysis.
        """)
        return
    
    # Save uploaded file temporarily
    if controls['uploaded_file']:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(controls['uploaded_file'].getbuffer())
            temp_file_path = tmp.name
    else:
        return
    
    # Run pipeline if requested
    if controls['run_button']:
        with st.spinner("Running analysis..."):
            try:
                runner = PipelineRunner(
                    file_path=temp_file_path,
                    target_column=controls['target_column'],
                    task_type=controls['task_type'],
                    dev_mode=controls['dev_mode']
                )
                
                result = runner.run()
                state.pipeline_result = result
                state.run_timestamp = datetime.now()
                
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                return
    
    # Display results if available
    if state.pipeline_result:
        result = state.pipeline_result
        
        if result['status'] == 'failure':
            st.error(f"Analysis failed: {result['error']}")
            return
        
        # Render dashboard sections
        render_header(result['metadata'])
        st.markdown("---")
        
        scorer = ReliabilityScorer()
        render_reliability_score(scorer, result.get('checks', {}))
        st.markdown("---")
        
        render_issues_section(result.get('checks', {}))
        st.markdown("---")
        
        render_feature_importance(result.get('feature_importance', {}))
        st.markdown("---")
        
        render_model_performance(result.get('model_metrics', {}), controls['task_type'])
        st.markdown("---")
        
        # Robust recommendations rendering
        render_recommendations(result.get('recommendations'))
        st.markdown("---")
        
        # Data sample
        df = pd.read_csv(temp_file_path)
        render_data_sample(df)
        
        # Export section
        st.markdown("## Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            json_str = json.dumps(result, indent=2, default=str)
            st.download_button(
                "Download JSON Report",
                data=json_str,
                file_name=f"pipeline_{state.run_timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            issues_df = pd.DataFrame(result.get('checks', {}).get('issues', []))
            if not issues_df.empty:
                csv_str = issues_df.to_csv(index=False)
                st.download_button(
                    "Download Issues CSV",
                    data=csv_str,
                    file_name=f"issues_{state.run_timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Cleanup temp file
    if os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
        except:
            pass


if __name__ == "__main__":
    main()