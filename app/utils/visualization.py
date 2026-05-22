"""
Visualization utilities for dashboard charts and metrics rendering.
Provides production-quality Plotly visualizations for ML diagnostics.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def format_metric_card(value: float, label: str, suffix: str = "", 
                      color: str = "#3498db", icon: str = "") -> str:
    """Format a metric card in HTML."""
    return f"""
    <div style="background-color: white; border-left: 4px solid {color}; 
                padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 14px; color: #666; margin-bottom: 10px;">
            {icon} {label}
        </div>
        <div style="font-size: 32px; font-weight: bold; color: {color};">
            {value:.2f}{suffix}
        </div>
    </div>
    """


def create_severity_chart(severity_counts: Dict[str, int], 
                         height: int = 300) -> go.Figure:
    """Create a severity distribution chart."""
    severity_order = ['Critical', 'High', 'Medium', 'Low']
    labels = []
    values = []
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#27ae60']
    
    for severity, color in zip(severity_order, colors):
        key = severity.lower()
        if key in severity_counts or severity in severity_counts:
            val = severity_counts.get(key, severity_counts.get(severity, 0))
            if val > 0:
                labels.append(severity)
                values.append(val)
                colors_filtered = color
    
    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=['#e74c3c', '#f39c12', '#f1c40f', '#27ae60'][:len(labels)]),
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Issues by Severity",
        xaxis_title="Count",
        yaxis_title="Severity",
        height=height,
        showlegend=False,
        margin=dict(l=100, r=20, t=30, b=20)
    )
    
    return fig


def create_feature_importance_chart(importance_dict: Dict[str, float],
                                   feature_names: Dict[str, str] = None,
                                   top_n: int = 15,
                                   height: int = 400) -> go.Figure:
    """
    Create a feature importance bar chart.
    
    Args:
        importance_dict: Feature importance scores
        feature_names: Mapping of original to cleaned names
        top_n: Number of top features to display
        height: Chart height
    """
    if not importance_dict:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data available")
        return fig
    
    # Get top features
    top_features = dict(sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n])
    
    # Clean feature names if mapping provided
    display_names = []
    for name in top_features.keys():
        if feature_names and name in feature_names:
            display_names.append(feature_names[name])
        else:
            display_names.append(name)
    
    importances = list(top_features.values())
    
    # Create chart
    fig = go.Figure(data=[go.Bar(
        y=display_names,
        x=importances,
        orientation='h',
        marker=dict(
            color=importances,
            colorscale='Viridis',
            showscale=False,
            line=dict(color='white', width=1)
        ),
        text=[f'{imp:.4f}' for imp in importances],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"Top {len(top_features)} Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature Name",
        height=height,
        margin=dict(l=250, r=100, t=40, b=20),
        hovermode='closest',
        showlegend=False
    )
    
    fig.update_yaxes(autorange="reversed")
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, 
                              top_n: int = 20,
                              height: int = 600) -> go.Figure:
    """
    Create correlation heatmap for numeric features.
    
    Args:
        df: DataFrame with numeric features
        top_n: Number of top correlated features to show
        height: Chart height
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] == 0:
        fig = go.Figure()
        fig.add_annotation(text="No numeric features for correlation analysis")
        return fig
    
    # Compute correlation
    corr = numeric_df.corr()
    
    # Optionally limit to top features by variance
    if len(corr) > top_n:
        # Select features with highest variance
        feature_var = numeric_df.var().nlargest(top_n).index
        corr = corr.loc[feature_var, feature_var]
    
    # Create heatmap
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
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        height=height,
        width=height,
        margin=dict(l=150, r=50, t=50, b=150),
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_reliability_gauge(score: float, 
                            max_value: float = 100) -> go.Figure:
    """Create a gauge chart for reliability score."""
    
    # Determine color based on score
    if score >= 85:
        color = '#27ae60'  # Green
    elif score >= 70:
        color = '#2ecc71'  # Light green
    elif score >= 50:
        color = '#f39c12'  # Orange
    elif score >= 30:
        color = '#e67e22'  # Dark orange
    else:
        color = '#e74c3c'  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Pipeline Health Score"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#ffcccc"},
                {'range': [30, 50], 'color': "#ffe6cc"},
                {'range': [50, 70], 'color': "#ffffcc"},
                {'range': [70, 85], 'color': "#e6f2ff"},
                {'range': [85, 100], 'color': "#ccf5dd"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig


def create_class_distribution_chart(target_values: pd.Series,
                                   height: int = 300) -> go.Figure:
    """Create a class distribution chart for classification tasks."""
    
    value_counts = target_values.value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=value_counts.index.astype(str),
        y=value_counts.values,
        marker=dict(
            color=value_counts.values,
            colorscale='Viridis'
        ),
        text=value_counts.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Target Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


def create_missing_values_chart(missing_pct: Dict[str, float],
                               height: int = 300) -> go.Figure:
    """Create a missing values distribution chart."""
    
    if not missing_pct:
        fig = go.Figure()
        fig.add_annotation(text="No missing values detected")
        return fig
    
    # Sort by percentage
    sorted_cols = sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)[:10]
    
    columns = [col for col, _ in sorted_cols]
    percentages = [pct for _, pct in sorted_cols]
    
    fig = go.Figure(data=[go.Bar(
        x=columns,
        y=percentages,
        marker=dict(
            color=percentages,
            colorscale='Reds'
        ),
        text=[f'{pct:.1f}%' for pct in percentages],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Top Missing Values by Column",
        xaxis_title="Column",
        yaxis_title="Missing %",
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_outlier_distribution_chart(outliers_dict: Dict[str, int],
                                     height: int = 300) -> go.Figure:
    """Create an outlier distribution chart."""
    
    if not outliers_dict:
        fig = go.Figure()
        fig.add_annotation(text="No outliers detected")
        return fig
    
    # Sort and limit
    sorted_outliers = sorted(outliers_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    columns = [col for col, _ in sorted_outliers]
    counts = [count for _, count in sorted_outliers]
    
    fig = go.Figure(data=[go.Bar(
        x=columns,
        y=counts,
        marker=dict(color='#e74c3c'),
        text=counts,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Univariate Outliers by Column",
        xaxis_title="Column",
        yaxis_title="Outlier Count",
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_model_metrics_comparison(metrics_dict: Dict[str, float],
                                   metric_names: List[str] = None,
                                   height: int = 300) -> go.Figure:
    """Create a comparison chart for model metrics."""
    
    if metric_names:
        display_metrics = {k: v for k, v in metrics_dict.items() if k in metric_names}
    else:
        display_metrics = metrics_dict
    
    if not display_metrics:
        fig = go.Figure()
        fig.add_annotation(text="No metrics available")
        return fig
    
    fig = go.Figure(data=[go.Bar(
        x=list(display_metrics.keys()),
        y=list(display_metrics.values()),
        marker=dict(color='#3498db'),
        text=[f'{v:.4f}' for v in display_metrics.values()],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=50),
        xaxis=dict(tickangle=45)
    )
    
    return fig


def create_timeline_chart(steps: List[Dict[str, Any]],
                         height: int = 300) -> go.Figure:
    """Create a pipeline execution timeline chart."""
    
    if not steps:
        fig = go.Figure()
        fig.add_annotation(text="No execution steps recorded")
        return fig
    
    # Extract data
    step_names = [step['name'] for step in steps]
    durations = [step['duration_ms'] for step in steps]
    
    fig = go.Figure(data=[go.Bar(
        x=durations,
        y=step_names,
        orientation='h',
        marker=dict(color='#3498db'),
        text=[f'{d}ms' for d in durations],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Pipeline Execution Timeline",
        xaxis_title="Duration (ms)",
        yaxis_title="Step",
        height=height,
        showlegend=False,
        margin=dict(l=150, r=20, t=30, b=20)
    )
    
    fig.update_yaxes(autorange="reversed")
    
    return fig