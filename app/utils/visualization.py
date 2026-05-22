"""
Visualization utilities for ML Pipeline Debugger.
Focused on high-value charts that communicate actual insight.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def create_severity_distribution(issues: List[Dict[str, Any]]) -> go.Figure:
    """
    Create severity distribution for non-trivial datasets only.
    Returns None if data is uninformative.
    """
    if not issues:
        return None
    
    severity_counts = {}
    for issue in issues:
        severity = issue.get('severity', 'medium')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    # Only show if there's actual distribution
    if len(severity_counts) == 1:
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
    """Create high-value feature importance bar chart."""
    if not importance_dict:
        return None
    
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    if not sorted_features:
        return None
    
    names = [feat for feat, _ in sorted_features]
    values = [imp for _, imp in sorted_features]
    
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
        xaxis_title="Importance Score",
        yaxis_title=None,
        showlegend=False,
        font=dict(size=11)
    )
    
    fig.update_yaxes(autorange="reversed")
    
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    max_features: int = 20
) -> go.Figure:
    """
    Create correlation heatmap for numeric features.
    Limits to top features by variance to avoid clutter.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return None
    
    # Select top features by variance
    if numeric_df.shape[1] > max_features:
        top_features = numeric_df.var().nlargest(max_features).index
        numeric_df = numeric_df[top_features]
    
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 9},
        hoverongaps=False
    ))
    
    size = min(400, 250 + numeric_df.shape[1] * 15)
    
    fig.update_layout(
        title="Feature Correlations",
        height=size,
        width=size,
        margin=dict(l=100, r=50, t=50, b=100),
        xaxis=dict(tickangle=-45),
        font=dict(size=10)
    )
    
    return fig


def create_reliability_gauge(
    score: float,
    status: str = "Fair"
) -> go.Figure:
    """Create professional reliability gauge."""
    color = "#c0392b"  # critical
    if score >= 85:
        color = "#27ae60"  # excellent
    elif score >= 70:
        color = "#3498db"  # good
    elif score >= 50:
        color = "#f39c12"  # fair
    elif score >= 30:
        color = "#e67e22"  # poor
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#ffebee'},
                {'range': [30, 50], 'color': '#fff3e0'},
                {'range': [50, 70], 'color': '#fffde7'},
                {'range': [70, 85], 'color': '#e8f5e9'},
                {'range': [85, 100], 'color': '#c8e6c9'}
            ]
        },
        number={'suffix': '%'}
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=20, b=0),
        font=dict(size=14)
    )
    
    return fig


def create_missing_values_chart(
    missing_pct: Dict[str, float],
    threshold: float = 0.05
) -> go.Figure:
    """
    Create chart only if there are notable missing values.
    Skip trivial amounts.
    """
    if not missing_pct:
        return None
    
    # Filter to notable missing values
    notable = {k: v for k, v in missing_pct.items() if v >= threshold}
    
    if not notable:
        return None
    
    sorted_items = sorted(notable.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if not sorted_items:
        return None
    
    columns, percentages = zip(*sorted_items)
    
    fig = go.Figure(data=[go.Bar(
        x=list(columns),
        y=list(percentages),
        marker=dict(color='#e67e22'),
        text=[f'{p:.1f}%' for p in percentages],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="Column",
        yaxis_title="Missing %",
        height=250,
        margin=dict(l=0, r=0, t=30, b=80),
        showlegend=False,
        xaxis=dict(tickangle=-45),
        font=dict(size=10)
    )
    
    return fig


def create_class_distribution_chart(
    target_series: pd.Series
) -> go.Figure:
    """Create class distribution for classification tasks."""
    if target_series is None or len(target_series) == 0:
        return None
    
    value_counts = target_series.value_counts()
    
    if len(value_counts) < 2:
        return None
    
    fig = go.Figure(data=[go.Bar(
        x=value_counts.index.astype(str),
        y=value_counts.values,
        marker=dict(color=['#3498db', '#e74c3c', '#f39c12', '#27ae60'][:len(value_counts)]),
        text=value_counts.values,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Target Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        font=dict(size=11)
    )
    
    return fig


def create_model_metrics_table(
    metrics: Dict[str, float]
) -> pd.DataFrame:
    """Convert metrics to displayable table format."""
    if not metrics:
        return pd.DataFrame()
    
    df_data = []
    for metric, value in sorted(metrics.items()):
        df_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Value': f'{value:.6f}' if isinstance(value, float) else str(value)
        })
    
    return pd.DataFrame(df_data)