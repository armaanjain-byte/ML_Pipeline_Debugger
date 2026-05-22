"""
Visualization utilities for ML Pipeline Debugger.
Focused on high-value charts that communicate actual insight.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def create_severity_distribution(issues: List[Dict[str, Any]]) -> go.Figure:
    if not issues:
        return None
    
    severity_counts = {}
    for issue in issues:
        severity = issue.get('severity', 'medium')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    if len(severity_counts) == 0:
        return None
    
    fig = go.Figure(data=[go.Bar(
        x=list(severity_counts.keys()),
        y=list(severity_counts.values()),
        marker=dict(
            color=['#c0392b', '#e67e22', '#f39c12', '#27ae60'],
            colorscale='RdYlGn_r'
        ),
        text=list(severity_counts.values()),
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Issue Distribution",
        xaxis_title="Severity",
        yaxis_title="Count",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    
    return fig


def create_feature_importance_bars(
    importance_dict: Dict[str, float],
    top_n: int = 15
) -> go.Figure:
    if not importance_dict:
        return None
    
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_features:
        return None
    
    names = [str(feat) for feat, _ in sorted_features]
    values = [float(imp) for _, imp in sorted_features]
    
    fig = go.Figure(data=[go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker=dict(color=values, colorscale='Blues', showscale=False),
        text=[f'{v:.4f}' for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>'
    )])
    
    fig.update_layout(
        height=300,
        margin=dict(l=200, r=80, t=0, b=20),
        xaxis_title="Importance Score (Normalized)",
        yaxis_title=None,
        showlegend=False,
        font=dict(size=11)
    )
    
    fig.update_yaxes(autorange="reversed")
    return fig


def create_correlation_heatmap(df: pd.DataFrame, max_features: int = 20) -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return None
    
    if numeric_df.shape[1] > max_features:
        top_features = numeric_df.var().nlargest(max_features).index
        numeric_df = numeric_df[top_features]
    
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 9},
        hoverongaps=False
    ))
    
    size = min(400, 250 + numeric_df.shape[1] * 15)
    
    fig.update_layout(
        title="Feature Correlations", height=size, width=size,
        margin=dict(l=100, r=50, t=50, b=100),
        xaxis=dict(tickangle=-45), font=dict(size=10)
    )
    return fig


def create_missing_values_chart(missing_pct: Dict[str, float], threshold: float = 0.05) -> go.Figure:
    if not missing_pct:
        return None
    
    notable = {k: v for k, v in missing_pct.items() if v >= threshold}
    if not notable:
        return None
    
    sorted_items = sorted(notable.items(), key=lambda x: x[1], reverse=True)[:10]
    if not sorted_items:
        return None
    
    columns, percentages = zip(*sorted_items)
    
    fig = go.Figure(data=[go.Bar(
        x=list(columns), y=list(percentages),
        marker=dict(color='#e67e22'),
        text=[f'{p:.1f}%' for p in percentages],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="Column", yaxis_title="Missing %",
        height=250, margin=dict(l=0, r=0, t=30, b=80),
        showlegend=False, xaxis=dict(tickangle=-45), font=dict(size=10)
    )
    return fig