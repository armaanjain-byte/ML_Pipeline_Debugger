# ML Pipeline Debugger

A developer-focused ML pipeline diagnostics and orchestration framework designed to detect silent failures in machine learning workflows before model training begins.

The system analyzes datasets, validates preprocessing behavior, detects statistical anomalies, and generates actionable engineering recommendations for improving pipeline reliability.

---
LIVE DEMO ->https://auz6hkv5aerealwygrhgrr.streamlit.app/

<img width="1914" height="1070" alt="Screenshot 2026-05-23 145845" src="https://github.com/user-attachments/assets/94c9a283-235b-4e60-a5a1-40c7d8ffcd43" />

# Why This Project Exists

Most machine learning systems do not fail because of model architecture.
They fail because of:

* target leakage
* train/test contamination
* unstable preprocessing
* schema inconsistencies
* hidden outliers
* distribution drift
* poor feature engineering decisions

These problems frequently pass silently through traditional ML workflows and produce models that appear accurate during validation but fail in production.

ML Pipeline Debugger acts as a validation and observability layer between raw data ingestion and model training.

---

# Core Features

## Automated Diagnostics Engine

The framework automatically detects:

* target leakage
* high-cardinality feature risks
* multicollinearity
* class imbalance
* schema inconsistencies
* multivariate outliers
* distribution anomalies
* preprocessing issues

---

## Recommendation Engine

Instead of only reporting errors, the system generates remediation suggestions with severity-aware recommendations.

Example:

```json
{
  "type": "multivariate_outliers",
  "severity": "high",
  "action": "investigate_anomalies",
  "recommendations": [
    "Investigate anomalous records",
    "Use robust scaling methods",
    "Consider RobustScaler for preprocessing"
  ]
}
```

---

## Robust ML Pipeline Construction

Uses native scikit-learn abstractions including:

* Pipeline
* ColumnTransformer
* cross-validation
* train/test isolation

This prevents:

* scaling leakage
* encoding contamination
* improper preprocessing application

---

## Structured Logging and Execution Tracing

Each pipeline stage is logged with:

* execution state
* diagnostic outcomes
* model evaluation metrics
* warning generation
* preprocessing behavior

---

# System Architecture

```text
                    ┌────────────────────┐
                    │   Raw Dataset      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    Data Loader     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Diagnostic Engine  │
                    └─────────┬──────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
 ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
 │ Leakage Check  │ │ Outlier Engine │ │ Schema Checks  │
 └────────────────┘ └────────────────┘ └────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Recommendation     │
                    │ Engine             │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Model Pipeline     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Structured Reports │
                    └────────────────────┘
```

---

# Project Structure

```text
ML_Pipeline_Debugger/
│
├── app/
│   ├── core/              # Configurations and exception handling
│   ├── debugger/          # Diagnostics and recommendation engines
│   ├── pipeline/          # Data loading, preprocessing, model execution
│   └── utils/             # Logging and helper utilities
│
├── data/                  # Example datasets
├── tests/                 # Unit and integration tests
├── main.py                # CLI entry point
├── requirements.txt
└── README.md
```

---

# Real Pipeline Execution Example

## Dataset

The framework was tested on the Telco Customer Churn dataset:

* 7,043 customer records
* mixed categorical and numerical features
* binary classification target

---

## Run

```bash
python main.py \
    --file data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --target Churn \
    --task classification
```

---

## Execution Trace

```text
============================================================
ML PIPELINE DEBUGGER INITIALIZED
============================================================

Dataset: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
Target: Churn
Task: Classification

------------------------------------------------------------

[Step 1/8] Loading data...
✓ Data loaded: 7043 rows

[Step 2/8] Running diagnostics on training set...
✓ Found 1 issue

[Step 3/8] Generating recommendations...

[Step 4/8] Training robust model pipeline...

Feature 'customerID' has high cardinality
Feature 'TotalCharges' has high cardinality

✓ Model trained on 20 features

[Step 5/8] Making predictions on holdout set...

[Step 6/8] Evaluating model performance...
✓ Metrics generated:
- accuracy
- precision
- recall
- f1-score
- cross-validation metrics

[Step 7/8] Computing feature importance...
WARNING: Feature importance method not found in Model class.

============================================================
PIPELINE SUCCESS
============================================================
```

---

# Diagnostic Recommendation Example

```json
{
  "recommendations": [
    {
      "type": "multivariate_outliers",
      "column": "dataset_wide",
      "severity": "high",
      "description": "Detected 3190 multivariate outliers (56.6%)",
      "action": "investigate_anomalies",
      "recommendations": [
        "Investigate anomalous records",
        "Use robust scaling methods",
        "Consider RobustScaler for preprocessing"
      ]
    }
  ]
}
```

---

# Technical Design

## Data Validation Layer

The framework validates datasets before model training begins.

Checks include:

* missing values
* datatype inconsistencies
* class imbalance
* variance analysis
* feature cardinality analysis

---

## Outlier Detection

Implements:

* Isolation Forest for multivariate anomaly detection
* IQR-based analysis for univariate outliers

This helps identify records capable of destabilizing model behavior.

---

## Leakage Prevention

Preprocessing operations are isolated using scikit-learn Pipelines to ensure:

* imputers fit only on training data
* scalers avoid test contamination
* encoders preserve holdout integrity

---

## Modular OOP Architecture

The project follows a modular object-oriented design:

* diagnostics are isolated from orchestration
* preprocessing is independent from training
* recommendation generation is extensible
* components can be tested independently

---

# Engineering Practices

* defensive programming
* structured logging
* modular architecture
* reusable preprocessing pipelines
* unit testing
* exception-safe execution
* configurable diagnostics

---

# Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* SciPy
* pytest

---

# Installation

```bash
git clone https://github.com/armaanjain-byte/ML_Pipeline_Debugger.git

cd ML_Pipeline_Debugger

python -m venv venv

source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

---

# Usage

## Classification

```bash
python main.py \
    --file data/dataset.csv \
    --target target_column \
    --task classification
```

## Regression

```bash
python main.py \
    --file data/dataset.csv \
    --target target_column \
    --task regression
```

## Development Mode

```bash
python main.py \
    --file data/dataset.csv \
    --target target_column \
    --task classification \
    --dev-mode
```

---

# Current Limitations

* feature importance extraction is currently model-dependent
* DAG-style tracing is limited to current sklearn-oriented workflows
* distributed execution support is not implemented
* visualization dashboards are not yet available

---

# Future Improvements

Planned extensions:

* MLflow integration
* Airflow pipeline tracing
* report visualization layer
* distributed execution support
* dataset drift monitoring
* experiment tracking integration

---

# Key Engineering Focus

This project focuses on ML systems reliability, observability, and debugging infrastructure rather than only model experimentation.

The goal is to make ML pipelines easier to validate, debug, and trust before deployment.
