"""
ML Pipeline Debugger - Enterprise Observability Dashboard
Provides dense, tabbed analytics for pre-deployment ML audits.
"""

import streamlit as st
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Any

import plotly.graph_objects as go
import plotly.express as px

from app.pipeline.pipeline_runner import PipelineRunner
from app.utils.feature_utils import FeatureNameCleaner
from app.utils.observability import ReliabilityScorer

st.set_page_config(
    page_title="ML Pipeline Debugger",
    page_icon="▢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise UI Styling - Focus on extreme density and readability
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }
    h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.2rem; }
    h2 { font-size: 1.2rem; font-weight: 600; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.3rem; margin-top: 1rem; margin-bottom: 0.5rem; }
    h3 { font-size: 1.0rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.2rem; }
    hr { margin: 0.8rem 0; border: 1px solid #ecf0f1; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stSidebar"] { width: 320px; background: #f8f9fa; }
    .deployment-banner { padding: 1rem; border-radius: 4px; color: white; font-weight: bold; margin-bottom: 1rem; text-align: center; font-size: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .severity-critical { color: #c0392b; font-weight: bold; }
    .severity-high { color: #e67e22; font-weight: bold; }
    .severity-medium { color: #f39c12; font-weight: bold; }
    .severity-low { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    @staticmethod
    def get_session_state():
        if 'pipeline_result' not in st.session_state:
            st.session_state.pipeline_result = None
        if 'run_timestamp' not in st.session_state:
            st.session_state.run_timestamp = None
        return st.session_state


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown("### Observability Control Panel")
    with st.sidebar:
        uploaded_file = st.file_uploader("Audit Dataset (CSV)", type=['csv'], key="csv_upload")
        target_column = st.text_input("Target Column", value="", placeholder="Target Variable")
        
        col1, col2 = st.columns(2)
        with col1:
            task_type = st.selectbox("Task Objective", ["classification", "regression"], key="task")
        with col2:
            st.write("") 
            st.write("")
            dev_mode = st.checkbox("Fast Subsample", value=False, help="Downsample to 5K rows")
            
        st.markdown("---")
        run_button = st.button("Execute Pipeline Audit", use_container_width=True, type="primary")
        
        if st.session_state.run_timestamp:
            st.caption(f"Last audit generated at: {st.session_state.run_timestamp.strftime('%H:%M:%S')}")
            
        return {
            'uploaded_file': uploaded_file, 'target_column': target_column,
            'task_type': task_type, 'dev_mode': dev_mode, 'run_button': run_button
        }


def render_executive_summary(scorer: ReliabilityScorer, result: Dict[str, Any], tab: st.delta_generator) -> None:
    with tab:
        issues = result.get('issues', [])
        metrics = result.get('metrics', {})
        
        score_data = scorer.compute_overall_score(issues)
        overall_score = score_data["score"]
        components = score_data["components"]
        
        # 1. Executive Banner
        readiness = scorer.get_deployment_readiness(overall_score, metrics)
        st.markdown(f"<div class='deployment-banner' style='background-color: {readiness['color']}'>{readiness['status']}<br><span style='font-size:0.95rem; font-weight:normal;'>{readiness['message']}</span></div>", unsafe_allow_html=True)
        
        # 2. Gauge & Components
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=overall_score, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Reliability Index", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': scorer.get_score_color(overall_score)},
                    'steps': [
                        {'range': [0, 50], 'color': '#ffebee'},
                        {'range': [50, 75], 'color': '#fff3e0'},
                        {'range': [75, 90], 'color': '#e8f5e9'},
                        {'range': [90, 100], 'color': '#c8e6c9'}
                    ]
                }
            ))
            fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("**Sub-system Health Penalities**")
            for name, data in components.items():
                color = "green" if data['score'] == 100 else "orange" if data['score'] >= 70 else "red"
                st.markdown(f"**{name}** (:{color}[{data['score']:.0f}/100]): {data['rationale']}")
                
        # 3. Telemetry
        st.markdown("---")
        st.markdown("**Pipeline Execution Telemetry**")
        telemetry = result.get('telemetry', {})
        cols = st.columns(6)
        cols[0].metric("Total Run", f"{telemetry.get('total_pipeline_seconds', 0.0):.2f}s")
        cols[1].metric("Load/Infer", f"{telemetry.get('data_loading_seconds', 0.0):.2f}s")
        cols[2].metric("Integrity", f"{telemetry.get('integrity_checks_seconds', 0.0):.2f}s")
        cols[3].metric("Diagnostics", f"{telemetry.get('diagnostics_seconds', 0.0):.2f}s")
        cols[4].metric("Train", f"{telemetry.get('training_seconds', 0.0):.2f}s")
        cols[5].metric("Evaluate", f"{telemetry.get('evaluation_seconds', 0.0):.2f}s")


def render_model_generalization(result: Dict[str, Any], tab: st.delta_generator) -> None:
    with tab:
        metrics = result.get('metrics', {})
        if not metrics:
            st.info("Model metrics not available.")
            return
            
        flags = metrics.get('observability_flags', [])
        if flags:
            for flag in flags:
                st.error(f"**{flag['flag']}**: {flag['detail']}")
        else:
            st.success("Model exhibits stable generalization. No severe overfit or CV variance detected.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Holdout Performance (Test Set)**")
            for m, v in metrics.get('holdout', {}).items():
                st.text(f"{m.upper():20} {v:.4f}")
        with col2:
            st.markdown("**Cross-Validation (Fold Stability)**")
            for m, v in metrics.get('cv', {}).items():
                st.text(f"{m.upper():20} {v:.4f}")
                
        st.markdown("---")
        st.markdown("**Feature Importance Attributions**")
        importance = result.get('feature_importance', {})
        if importance:
            cleaner = FeatureNameCleaner()
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
            names = [cleaner.clean_feature_name(n) for n in top_features.keys()]
            vals = list(top_features.values())
            
            fig = px.bar(x=vals, y=names, orientation='h', labels={'x': 'Relative Importance', 'y': ''})
            fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


def render_feature_health(result: Dict[str, Any], tab: st.delta_generator) -> None:
    with tab:
        psi_data = result.get('psi_table', [])
        vif_data = result.get('vif_table', [])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Population Stability Index (Feature Drift)**")
            if psi_data:
                # Color code severities
                df_psi = pd.DataFrame(psi_data)
                st.dataframe(df_psi, use_container_width=True, hide_index=True)
            else:
                st.info("No drift detected or numeric features unavailable.")
                
        with col2:
            st.markdown("**Variance Inflation Factor (Multicollinearity)**")
            if vif_data:
                df_vif = pd.DataFrame(vif_data)
                st.dataframe(df_vif, use_container_width=True, hide_index=True)
            else:
                st.success("No significant multicollinearity detected (VIF < 2.0).")


def render_preprocessing_traceability(result: Dict[str, Any], tab: st.delta_generator) -> None:
    with tab:
        audit_data = result.get('feature_audit', [])
        if not audit_data:
            st.info("No preprocessing metadata available.")
            return
            
        st.markdown("**Raw Schema & Implicit Coercions**")
        df_audit = pd.DataFrame(audit_data)
        st.dataframe(df_audit, use_container_width=True, hide_index=True, height=500)


def render_actionable_recommendations(result: Dict[str, Any], tab: st.delta_generator) -> None:
    with tab:
        recs = result.get('recommendations', [])
        if not recs:
            st.success("No critical engineering actions required. Pipeline is stable.")
            return
            
        for rec in recs:
            sev_color = "red" if rec['severity'] == 'critical' else "orange" if rec['severity'] == 'high' else "green"
            with st.expander(f":{sev_color}[[{rec.get('severity', 'Medium').upper()}]] {rec.get('title', 'Recommendation')} — `{rec.get('column', '')}`"):
                st.markdown(f"**Trigger Evidence:** {rec.get('description', '')}")
                st.markdown(f"**Expected Impact:** {rec.get('impact', '')}")
                st.markdown(f"**Technical Rationale:** {rec.get('rationale', '')}")
                st.markdown("**Remediation Options:**")
                for action in rec.get('actions', []):
                    st.markdown(f"- {action}")


def main():
    state = DashboardState.get_session_state()
    controls = render_sidebar()
    
    if not controls['uploaded_file'] or not controls['target_column']:
        st.markdown("""
        # ML Pipeline Debugger
        **Enterprise Observability & Pre-Deployment Audit Platform**
        
        Upload a dataset in the sidebar to generate a comprehensive reliability report evaluating:
        - Structural Target Leakage
        - Feature Drift (PSI) & Volatility
        - Multicollinearity (VIF)
        - Overfitting & Generalization Decay
        - Preprocessing Integrity
        """)
        return
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(controls['uploaded_file'].getbuffer())
        temp_file_path = tmp.name
    
    if controls['run_button']:
        with st.spinner("Executing deep diagnostic audit..."):
            try:
                runner = PipelineRunner(
                    file_path=temp_file_path,
                    target_column=controls['target_column'],
                    task_type=controls['task_type'],
                    dev_mode=controls['dev_mode']
                )
                state.pipeline_result = runner.run()
                state.run_timestamp = datetime.now()
            except Exception as e:
                st.error(f"Pipeline Audit Failed: {str(e)}")
                return
    
    if state.pipeline_result:
        result = state.pipeline_result
        if result.get('pipeline_status') == 'failure':
            st.error(f"Audit Generation Failed: {result.get('error', 'Unknown Error')}")
            return
            
        # Build the Grafana-style Tabbed Interface
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Summary", 
            "🧠 Model Generalization", 
            "📈 Feature Health", 
            "⚙️ Preprocessing Traceability", 
            "🛠️ Actionable Alerts"
        ])
        
        render_executive_summary(ReliabilityScorer(), result, tab1)
        render_model_generalization(result, tab2)
        render_feature_health(result, tab3)
        render_preprocessing_traceability(result, tab4)
        render_actionable_recommendations(result, tab5)
        
    if os.path.exists(temp_file_path):
        try: os.unlink(temp_file_path)
        except: pass

if __name__ == "__main__":
    main()