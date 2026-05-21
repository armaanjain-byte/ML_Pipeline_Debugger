import json
import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ML Reliability Platform",
    layout="wide"
)

# =====================================================
# LOAD REPORT
# =====================================================

with open("reports/report.json") as f:
    report = json.load(f)

# =====================================================
# RELIABILITY SCORE
# =====================================================

def compute_reliability_score(report):

    score = 100

    issues = report.get(
        "issues",
        []
    )

    for issue in issues:

        severity = issue.get(
            "severity",
            "low"
        ).lower()

        issue_type = issue.get(
            "type",
            ""
        )

        if severity == "critical":
            score -= 20

        elif severity == "high":
            score -= 10

        elif severity == "medium":
            score -= 4

        elif severity == "low":
            score -= 1

        if issue_type == "target_leakage":
            score -= 15

        elif issue_type == "class_imbalance":
            score -= 5

        elif issue_type == "multivariate_outliers":
            score -= 5

    return max(score, 0)

# =====================================================
# LOAD DATA
# =====================================================

dataset = report.get(
    "dataset",
    {}
)

metrics = report.get(
    "metrics",
    {}
)

issues = report.get(
    "issues",
    []
)

recommendations = report.get(
    "recommendations",
    []
)

feature_importance = report.get(
    "feature_importance",
    {}
)

reliability_score = compute_reliability_score(
    report
)

# =====================================================
# TITLE
# =====================================================

st.title(
    "ML Reliability & Diagnostics Platform"
)

st.caption(
    "Observability Layer for Machine Learning Pipelines"
)

# =====================================================
# OVERVIEW
# =====================================================

st.header("Pipeline Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Rows",
    dataset.get(
        "rows",
        0
    )
)

col2.metric(
    "Columns",
    dataset.get(
        "columns",
        0
    )
)

col3.metric(
    "Issues",
    len(issues)
)

accuracy = metrics.get(
    "accuracy",
    0
)

if isinstance(accuracy, float):

    accuracy = round(
        accuracy,
        4
    )

col4.metric(
    "Accuracy",
    accuracy
)

col5.metric(
    "Reliability",
    f"{reliability_score}/100"
)

# =====================================================
# RISK STATUS
# =====================================================

if reliability_score >= 85:

    st.success(
        "LOW RISK PIPELINE"
    )

elif reliability_score >= 60:

    st.warning(
        "MODERATE RISK PIPELINE"
    )

else:

    st.error(
        "HIGH RISK PIPELINE"
    )

# =====================================================
# DATASET PROFILE
# =====================================================

st.header("Dataset Profile")

profile1, profile2, profile3 = st.columns(3)

profile1.metric(
    "Numeric Features",
    dataset.get(
        "numeric_features",
        0
    )
)

profile2.metric(
    "Categorical Features",
    dataset.get(
        "categorical_features",
        0
    )
)

profile3.metric(
    "Target Column",
    dataset.get(
        "target",
        "unknown"
    )
)

# =====================================================
# DETECTED FINDINGS
# =====================================================

st.header("Detected Findings")

issues_df = pd.DataFrame(issues)

st.dataframe(
    issues_df,
    use_container_width=True
)

# =====================================================
# SEVERITY DISTRIBUTION
# =====================================================

st.header(
    "Severity Distribution"
)

severity_counts = (
    issues_df["severity"]
    .value_counts()
    .reset_index()
)

severity_counts.columns = [
    "Severity",
    "Count"
]

fig = px.bar(
    severity_counts,
    x="Severity",
    y="Count",
    color="Severity"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# ISSUE CATEGORIES
# =====================================================

st.header("Issue Categories")

category_counts = (
    issues_df["category"]
    .value_counts()
    .reset_index()
)

category_counts.columns = [
    "Category",
    "Count"
]

fig = px.pie(
    category_counts,
    names="Category",
    values="Count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# PIPELINE EXECUTION FLOW
# =====================================================

st.header("Pipeline Execution Flow")

pipeline_steps = [

    "1. Data Loading",

    "2. Train/Test Split",

    "3. Diagnostic Analysis",

    "4. Recommendation Generation",

    "5. Model Training",

    "6. Prediction",

    "7. Cross Validation",

    "8. Evaluation",

    "9. Feature Importance"
]

for step in pipeline_steps:

    st.success(step)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

if feature_importance:

    st.header("Feature Importance")

    fi_df = pd.DataFrame({

        "Feature":
            list(feature_importance.keys()),

        "Importance":
            list(feature_importance.values())
    })

    fi_df = fi_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(

        fi_df.head(10),

        x="Importance",

        y="Feature",

        orientation="h",

        text="Importance",

        title="Top Predictive Features"
    )

    fig.update_layout(

        yaxis=dict(
            categoryorder="total ascending"
        ),

        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# RECOMMENDATIONS
# =====================================================

st.header("Recommendations")

title_map = {

    "dataset_shape":
        "Dataset Structure",

    "duplicate_rows":
        "Duplicate Records",

    "high_cardinality":
        "High Cardinality Features",

    "class_balance":
        "Class Imbalance",

    "multivariate_outliers":
        "Multivariate Outliers",

    "high_correlation":
        "Feature Correlation",

    "constant_feature":
        "Constant Features"
}

for i, rec in enumerate(recommendations):

    raw_type = rec.get(
        "type",
        "Recommendation"
    )

    title = title_map.get(
        raw_type,
        raw_type.replace("_", " ").title()
    )

    with st.expander(title):

        st.write(
            f"Severity: {rec.get('severity')}"
        )

        st.write(
            f"Column: {rec.get('column')}"
        )

        st.write(
            f"Description: {rec.get('description')}"
        )

        st.write(
            f"Action: {rec.get('action')}"
        )

        rationale = rec.get(
            "rationale",
            ""
        )

        if rationale:

            st.write(
                f"Rationale: {rationale}"
            )

        recs = rec.get(
            "recommendations",
            []
        )

        if recs:

            st.write(
                "Recommendations:"
            )

            for item in recs:

                st.markdown(
                    f"- {item}"
                )

# =====================================================
# MODEL METRICS
# =====================================================

st.header("Model Metrics")

metrics_df = pd.DataFrame(
    metrics.items(),
    columns=["Metric", "Value"]
)

st.dataframe(
    metrics_df,
    use_container_width=True
)

# =====================================================
# RAW REPORT
# =====================================================

with st.expander("Raw Report JSON"):

    st.json(report)