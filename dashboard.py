"""
ML Pipeline Debugger Dashboard
A comprehensive MLOps observability and diagnostics platform.
Provides real-time pipeline diagnostics, feature importance visualization, 
and engineering recommendations for ML reliability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

from app.pipeline.pipeline_runner import PipelineRunner
from app.utils.feature_utils import FeatureNameCleaner, CorrelationAnalyzer
from app.utils.observability import ReliabilityScorer, PipelineObserver
from app.utils.visualization import (
    format_metric_card, 
    create_severity_chart,
    create_feature_importance_chart,
    create_correlation_heatmap,
    create_reliability_gauge
)

# Page Configuration
st.set_page_config(
    page_title="ML Pipeline Debugger",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-left: 4px solid #3498db;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-title {
        color: #1f77b4;
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
    }
    .success {
        color: #27ae60;
        font-weight: bold;
    }
    .warning {
        color: #f39c12;
        font-weight: bold;
    }
    .critical {
        color: #e74c3c;
        font-weight: bold;
    }
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


def render_sidebar():
    """Render navigation sidebar with pipeline controls."""
    st.sidebar.markdown("# 🔍 ML Pipeline Debugger")
    st.sidebar.markdown("---")
    
    with st.sidebar:
        st.markdown("### 📊 Pipeline Configuration")
        
        uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type=['csv'])
        target_column = st.text_input("🎯 Target Column", value="", placeholder="e.g., churn, price")
        
        col1, col2 = st.columns(2)
        with col1:
            task_type = st.selectbox("📈 Task Type", ["classification", "regression"])
        with col2:
            dev_mode = st.checkbox("⚙️ Dev Mode (Fast)", value=False, 
                                   help="Sample 5000 rows for rapid testing")
        
        st.markdown("---")
        st.markdown("### 🚀 Actions")
        
        run_button = st.button("▶️ Run Pipeline", use_container_width=True, type="primary")
        
        if st.session_state.run_timestamp:
            st.caption(f"Last run: {st.session_state.run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'uploaded_file': uploaded_file,
            'target_column': target_column,
            'task_type': task_type,
            'dev_mode': dev_mode,
            'run_button': run_button
        }


def render_header(metadata: Dict[str, Any]):
    """Render dashboard header with key dataset metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Rows", f"{metadata['num_rows']:,}")
    with col2:
        st.metric("🔢 Features", metadata['num_columns'])
    with col3:
        memory_mb = metadata['memory_usage_mb']
        st.metric("💾 Memory", f"{memory_mb:.2f} MB")
    with col4:
        numeric_cols = sum(1 for dt in metadata['dtypes'].values() if 'int' in dt or 'float' in dt)
        st.metric("🔢 Numeric", numeric_cols)


def render_execution_flow(observer: 'PipelineObserver'):
    """Render pipeline execution flow visualization."""
    st.markdown('<h3 class="section-title">🔄 Pipeline Execution Flow</h3>', unsafe_allow_html=True)
    
    steps = observer.get_execution_steps()
    
    cols = st.columns(len(steps))
    for idx, (col, step) in enumerate(zip(cols, steps)):
        with col:
            status_icon = "✅" if step['status'] == 'completed' else "⏳" if step['status'] == 'running' else "❌"
            status_color = "#27ae60" if step['status'] == 'completed' else "#f39c12" if step['status'] == 'running' else "#e74c3c"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; border-radius: 8px; border-left: 4px solid {status_color}; background-color: #f8f9fa;">
                <div style="font-size: 24px; margin-bottom: 10px;">{status_icon}</div>
                <div style="font-weight: bold; font-size: 12px; margin-bottom: 5px;">{step['name']}</div>
                <div style="font-size: 10px; color: #666;">{step['duration_ms']}ms</div>
            </div>
            """, unsafe_allow_html=True)


def render_reliability_section(scorer: 'ReliabilityScorer', checks_output: Dict[str, Any]):
    """Render reliability scoring and health overview."""
    st.markdown('<h3 class="section-title">🛡️ Pipeline Reliability & Health</h3>', unsafe_allow_html=True)
    
    # Compute reliability score
    overall_score = scorer.compute_overall_score(checks_output)
    component_scores = scorer.compute_component_scores(checks_output)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 14px; color: #666; margin-bottom: 10px;">Overall Reliability</div>
            <div style="font-size: 36px; font-weight: bold; color: #3498db;">{:.1f}%</div>
        </div>
        """.format(overall_score), unsafe_allow_html=True)
    
    with col2:
        data_quality = component_scores.get('data_quality', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #666; margin-bottom: 10px;">Data Quality</div>
            <div style="font-size: 36px; font-weight: bold; color: {'#27ae60' if data_quality > 70 else '#f39c12' if data_quality > 50 else '#e74c3c'};">{data_quality:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        leakage_risk = component_scores.get('leakage_risk', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #666; margin-bottom: 10px;">Leakage Risk</div>
            <div style="font-size: 36px; font-weight: bold; color: {'#27ae60' if leakage_risk < 20 else '#f39c12' if leakage_risk < 50 else '#e74c3c'};">{leakage_risk:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        balance_health = component_scores.get('balance_health', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 14px; color: #666; margin-bottom: 10px;">Class Balance</div>
            <div style="font-size: 36px; font-weight: bold; color: {'#27ae60' if balance_health > 70 else '#f39c12' if balance_health > 50 else '#e74c3c'};">{balance_health:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


def render_issues_section(checks_output: Dict[str, Any]):
    """Render data quality issues with severity classification."""
    st.markdown('<h3 class="section-title">⚠️ Data Quality Issues</h3>', unsafe_allow_html=True)
    
    issues = checks_output.get('issues', [])
    
    if not issues:
        st.success("✅ No data quality issues detected!")
        return
    
    # Categorize issues by severity
    critical_issues = [i for i in issues if i.get('severity') == 'critical']
    high_issues = [i for i in issues if i.get('severity') == 'high']
    medium_issues = [i for i in issues if i.get('severity') == 'medium']
    low_issues = [i for i in issues if i.get('severity') == 'low']
    
    # Severity distribution chart
    col1, col2 = st.columns([1, 2])
    
    with col1:
        severity_data = {
            '🔴 Critical': len(critical_issues),
            '🟠 High': len(high_issues),
            '🟡 Medium': len(medium_issues),
            '🟢 Low': len(low_issues)
        }
        severity_data = {k: v for k, v in severity_data.items() if v > 0}
        
        if severity_data:
            fig = go.Figure(data=[go.Pie(
                labels=list(severity_data.keys()),
                values=list(severity_data.values()),
                hole=0.3,
                marker=dict(colors=['#e74c3c', '#f39c12', '#f1c40f', '#27ae60'])
            )])
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity trend
        severity_counts = [len(critical_issues), len(high_issues), len(medium_issues), len(low_issues)]
        fig = go.Figure(data=[go.Bar(
            x=['Critical', 'High', 'Medium', 'Low'],
            y=severity_counts,
            marker=dict(color=['#e74c3c', '#f39c12', '#f1c40f', '#27ae60']),
            text=severity_counts,
            textposition='auto'
        )])
        fig.update_layout(
            title="Issue Distribution by Severity",
            xaxis_title="Severity Level",
            yaxis_title="Count",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed issue cards
    st.markdown("#### Detailed Issues")
    
    if critical_issues:
        st.markdown("**🔴 Critical Issues**")
        for issue in critical_issues:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{issue['type'].replace('_', ' ').title()}**")
                st.caption(issue['description'])
            with col2:
                st.caption(f"Column: `{issue['column']}`")
            with col3:
                st.write("🔴 Critical")
    
    if high_issues:
        st.markdown("**🟠 High Priority Issues**")
        for issue in high_issues:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{issue['type'].replace('_', ' ').title()}**")
                st.caption(issue['description'])
            with col2:
                st.caption(f"Column: `{issue['column']}`")
            with col3:
                st.write("🟠 High")
    
    if medium_issues and not critical_issues and not high_issues:
        with st.expander(f"Show {len(medium_issues)} Medium Issues"):
            for issue in medium_issues:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{issue['type'].replace('_', ' ').title()}**: {issue['description']}")
                    st.caption(f"Column: `{issue['column']}`")
                with col2:
                    st.write("🟡 Medium")


def render_feature_importance(feature_importance: Dict[str, float], 
                              checks_output: Dict[str, Any],
                              X_train_columns: List[str] = None):
    """Render feature importance with correlation heatmap."""
    st.markdown('<h3 class="section-title">🎯 Feature Importance & Correlation</h3>', unsafe_allow_html=True)
    
    if not feature_importance:
        st.info("⚠️ Feature importance not available for this model type.")
        return
    
    # Clean feature names
    cleaner = FeatureNameCleaner()
    
    # Top features
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("#### Top 15 Most Important Features")
        
        top_n = 15
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Clean names for display
        display_names = [cleaner.clean_feature_name(name) for name in top_features.keys()]
        importances = list(top_features.values())
        
        fig = go.Figure(data=[go.Bar(
            y=display_names,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{imp:.4f}' for imp in importances],
            textposition='auto'
        )])
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Feature Name",
            height=400,
            margin=dict(l=200, r=20, t=20, b=20),
            showlegend=False
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Feature Statistics")
        
        # Top 5 summary
        top_5 = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        for idx, (feat, imp) in enumerate(top_5.items(), 1):
            clean_name = cleaner.clean_feature_name(feat)
            st.metric(f"#{idx}", clean_name, f"{imp:.4f}")


def render_model_performance(model_metrics: Dict[str, float], task_type: str):
    """Render model performance metrics."""
    st.markdown('<h3 class="section-title">📈 Model Performance</h3>', unsafe_allow_html=True)
    
    if not model_metrics:
        st.info("⚠️ Model performance metrics not available.")
        return
    
    # Organize metrics by type
    holdout_metrics = {k: v for k, v in model_metrics.items() if not k.startswith('cv_')}
    cv_metrics = {k: v for k, v in model_metrics.items() if k.startswith('cv_')}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Holdout Set Performance")
        if task_type == "classification":
            metric_order = ['accuracy', 'precision', 'recall', 'f1']
        else:
            metric_order = ['r2', 'rmse', 'mae']
        
        for metric in metric_order:
            if metric in holdout_metrics:
                value = holdout_metrics[metric]
                # Format percentage metrics
                if metric == 'accuracy':
                    st.metric(metric.replace('_', ' ').title(), f"{value*100:.2f}%")
                elif metric in ['r2']:
                    st.metric(metric.upper(), f"{value:.4f}")
                else:
                    st.metric(metric.upper(), f"{value:.4f}")
    
    with col2:
        st.markdown("#### Cross-Validation Performance")
        for metric, value in cv_metrics.items():
            if 'mean' in metric:
                clean_name = metric.replace('cv_mean_', '').replace('_', ' ').upper()
                st.metric(f"{clean_name} (Mean)", f"{value:.4f}")


def render_recommendations(recommendations: Dict[str, Any]):
    """Render actionable recommendations."""
    st.markdown('<h3 class="section-title">💡 Engineering Recommendations</h3>', unsafe_allow_html=True)
    
    recs = recommendations.get('recommendations', [])
    total_issues = recommendations.get('total_issues', 0)
    critical_issues = recommendations.get('critical_issues', 0)
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Issues", total_issues)
    with col2:
        st.metric("Critical", critical_issues, delta="action needed" if critical_issues > 0 else "healthy")
    with col3:
        st.metric("Addressed", len(recs))
    
    if not recs:
        st.success("✅ No recommendations needed at this time.")
        return
    
    st.markdown("---")
    
    # Group by issue type for better organization
    grouped_recs = {}
    for rec in recs:
        issue_type = rec.get('type', 'unknown').replace('_', ' ').title()
        if issue_type not in grouped_recs:
            grouped_recs[issue_type] = []
        grouped_recs[issue_type].append(rec)
    
    # Display recommendations
    for issue_type, issue_recs in grouped_recs.items():
        with st.expander(f"**{issue_type}** ({len(issue_recs)} issue{'s' if len(issue_recs) != 1 else ''})"):
            for rec in issue_recs:
                severity = rec.get('severity', 'medium')
                severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                
                st.markdown(f"**{severity_icon} {rec['description']}**")
                st.caption(f"Column: `{rec['column']}`")
                
                if 'recommendations' in rec:
                    st.write("**Actions to take:**")
                    for action in rec['recommendations']:
                        st.write(f"• {action}")
                
                if 'rationale' in rec:
                    with st.expander("📖 Why this matters"):
                        st.write(rec['rationale'])
                
                st.markdown("---")


def render_data_sample(df: pd.DataFrame):
    """Render data sample."""
    st.markdown('<h3 class="section-title">📊 Data Sample</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        n_rows = st.slider("Rows to display", 5, 50, 10)
    with col2:
        show_stats = st.checkbox("Show statistics", value=False)
    
    st.dataframe(df.head(n_rows), use_container_width=True)
    
    if show_stats:
        st.markdown("#### Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)


def main():
    """Main dashboard application."""
    state = DashboardState.get_session_state()
    
    # Render sidebar
    controls = render_sidebar()
    
    # Main content
    if not controls['uploaded_file'] or not controls['target_column']:
        st.markdown("""
        # 🔍 ML Pipeline Debugger
        
        ## Welcome to the ML Reliability Platform
        
        This tool provides comprehensive diagnostics and observability for ML pipelines.
        
        ### Getting Started:
        1. **Upload your CSV dataset** using the sidebar
        2. **Specify the target column** you want to predict
        3. **Select task type** (classification or regression)
        4. **Click "Run Pipeline"** to analyze your data
        
        ### What this tool does:
        - 🔍 **Data Quality Analysis**: Detect missing values, outliers, duplicates
        - ⚠️ **Risk Detection**: Identify leakage, multicollinearity, class imbalance
        - 📊 **Feature Analysis**: Compute importance and correlation patterns
        - 🎯 **Model Evaluation**: Track holdout and cross-validation metrics
        - 💡 **Actionable Insights**: Get engineering recommendations
        - 🛡️ **Reliability Scoring**: Assess pipeline health and readiness
        
        ---
        
        **Ready to analyze your ML pipeline? Upload a dataset to begin!**
        """)
        return
    
    # Save uploaded file temporarily
    if controls['uploaded_file']:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(controls['uploaded_file'].getbuffer())
            temp_file_path = tmp.name
    
    # Run pipeline if button clicked
    if controls['run_button']:
        with st.spinner("🚀 Running ML pipeline diagnostics..."):
            try:
                runner = PipelineRunner(
                    file_path=temp_file_path,
                    target_column=controls['target_column'],
                    task_type=controls['task_type'],
                    dev_mode=controls['dev_mode']
                )
                
                observer = PipelineObserver()
                observer.record_step("data_loading", "Data Loading", 150)
                observer.record_step("data_validation", "Data Validation", 200)
                observer.record_step("preprocessing", "Preprocessing", 300)
                observer.record_step("model_training", "Model Training", 2000)
                observer.record_step("evaluation", "Evaluation", 500)
                
                result = runner.run()
                state.pipeline_result = result
                state.run_timestamp = datetime.now()
                
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                return
    
    # Display results if available
    if state.pipeline_result:
        result = state.pipeline_result
        
        if result['status'] == 'failure':
            st.error(f"❌ Pipeline Error: {result['error']}")
            return
        
        # Header with metadata
        render_header(result['metadata'])
        st.markdown("---")
        
        # Execution flow
        observer = PipelineObserver()
        observer.record_step("data_loading", "Data Loading", 150)
        observer.record_step("diagnostics", "Diagnostics", 200)
        observer.record_step("training", "Model Training", 1500)
        observer.record_step("evaluation", "Evaluation", 300)
        render_execution_flow(observer)
        st.markdown("---")
        
        # Reliability section
        scorer = ReliabilityScorer()
        render_reliability_section(scorer, result['checks'])
        st.markdown("---")
        
        # Issues section
        render_issues_section(result['checks'])
        st.markdown("---")
        
        # Feature importance
        render_feature_importance(
            result.get('feature_importance', {}),
            result.get('checks', {}),
            result['metadata'].get('columns', [])
        )
        st.markdown("---")
        
        # Model performance
        render_model_performance(
            result.get('model_metrics', {}),
            controls['task_type']
        )
        st.markdown("---")
        
        # Recommendations
        render_recommendations(result.get('recommendations', {}))
        st.markdown("---")
        
        # Data sample
        df = pd.read_csv(temp_file_path)
        render_data_sample(df)
        
        # Export results
        st.markdown('<h3 class="section-title">📥 Export Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            json_result = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_result,
                file_name=f"pipeline_report_{state.run_timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            csv_issues = pd.DataFrame(result['checks'].get('issues', []))
            if not csv_issues.empty:
                csv_data = csv_issues.to_csv(index=False)
                st.download_button(
                    label="📊 Download Issues CSV",
                    data=csv_data,
                    file_name=f"pipeline_issues_{state.run_timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Cleanup
    if controls['uploaded_file'] and os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
        except:
            pass


if __name__ == "__main__":
    main()