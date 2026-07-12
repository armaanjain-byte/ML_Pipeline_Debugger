# ML Pipeline Debugger

A diagnostics and observability framework that detects silent failures in ML pipelines before model training begins.

**Live Demo →** [auz6hkv5aerealwygrhgrr.streamlit.app](https://auz6hkv5aerealwygrhgrr.streamlit.app/)

---

<!-- Streamlit app screenshot placeholder -->
ML Pipeline Debugger UI
<img width="1914" height="1070" alt="Screenshot 2026-05-23 145845" src="https://github.com/user-attachments/assets/336fd1ad-9ac3-43ec-98a5-91806dbb1017" />


---

## The Problem

Most ML pipelines fail silently.

The model trains. Metrics look fine. Then it fails in production — not because the architecture was wrong, but because the data going into training was wrong in ways that were never caught:

- target leakage
- train/test contamination from improperly scoped preprocessing
- schema inconsistencies masked by type coercion
- distribution anomalies that distort learned boundaries
- high-cardinality features treated as categorical with no encoding guard

These failures are hard to see, easy to miss, and expensive to debug post-deployment.

ML Pipeline Debugger is a validation and observability layer that runs before training and catches these problems structurally.

---

## What It Does

```
Raw Dataset
    ↓
Data Loader → schema audit, type validation
    ↓
Diagnostic Engine → leakage, outliers, cardinality, class balance, multicollinearity
    ↓
Recommendation Engine → severity-aware remediation suggestions
    ↓
Robust Pipeline Construction → sklearn pipelines with proper train-only fitting
    ↓
Model Evaluation → CV metrics, feature importance
    ↓
Structured Report
```

---

## Diagnostics Covered

| Check | What It Catches |
|---|---|
| Target Leakage Detection | Features that directly encode the label |
| Train/Test Contamination | Scalers/encoders fit on full dataset before splitting |
| Multivariate Outlier Detection | Isolation Forest — detects anomaly clusters invisible to univariate checks |
| Class Imbalance | Label distribution skew that inflates accuracy metrics |
| High-Cardinality Risk | Features likely to cause dimensionality explosion or overfitting |
| Multicollinearity | Correlated features that destabilize linear model coefficients |
| Schema Inconsistency | Type drift, implicit nulls, mixed representations |
| Distribution Anomaly | Skew, kurtosis, and non-normal distributions that affect preprocessing assumptions |

---

## Recommendation Format

Every diagnostic produces a structured, actionable recommendation — not just a warning:

```json
{
  "type": "multivariate_outliers",
  "column": "dataset_wide",
  "severity": "high",
  "description": "Detected 3190 multivariate outliers (56.6%)",
  "action": "investigate_anomalies",
  "recommendations": [
    "Investigate anomalous records before training",
    "Use RobustScaler instead of StandardScaler",
    "Consider outlier-robust loss functions"
  ]
}
```

Severity levels: `low`, `medium`, `high`. High-severity findings block pipeline continuation by default in strict mode.

---

## Real Execution Trace

Tested on the Telco Customer Churn dataset (7,043 records):

```
============================================================
ML PIPELINE DEBUGGER INITIALIZED
============================================================

Dataset:  data/WA_Fn-UseC_-Telco-Customer-Churn.csv
Target:   Churn
Task:     Classification

[Step 1/8] Loading data...
✓ Data loaded: 7043 rows

[Step 2/8] Running diagnostics on training set...
✓ Found 1 issue

[Step 3/8] Generating recommendations...

[Step 4/8] Training robust model pipeline...
  ⚠ Feature 'customerID' has high cardinality
  ⚠ Feature 'TotalCharges' has high cardinality
  ✓ Model trained on 20 features

[Step 5/8] Making predictions on holdout set...
[Step 6/8] Evaluating model performance...
  ✓ accuracy, precision, recall, f1-score, cross-validation metrics

[Step 7/8] Computing feature importance...
[Step 8/8] Generating report...

============================================================
PIPELINE SUCCESS
============================================================
```

---

## Pipeline Design — What "Robust" Means

Sklearn's `Pipeline` and `ColumnTransformer` are used throughout — not as convenience, but as a correctness constraint:

**Imputers fit on training folds only.** Values from the test set never inform imputation parameters.

**Scalers scoped to training data.** Mean and variance computed from training, applied to test. Standard practice that's frequently violated in notebook-first workflows.

**Encoders preserve holdout integrity.** Unseen categories are handled explicitly, not silently dropped or errored.

This isn't complexity for its own sake — it's the minimum required to produce trustworthy evaluation metrics.

---

## System Architecture

```
                    ┌────────────────────┐
                    │   Raw Dataset      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │    Data Loader     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Diagnostic Engine │
                    └──┬──────┬──────┬───┘
                       │      │      │
              ┌────────▼─┐ ┌──▼───┐ ┌▼────────────┐
              │ Leakage  │ │Outl. │ │Schema Checks│
              └────────┬─┘ └──┬───┘ └┬────────────┘
                       └──────┴──────┘
                              │
                    ┌─────────▼──────────┐
                    │ Recommendation     │
                    │ Engine             │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Model Pipeline   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Structured Report │
                    └────────────────────┘
```

---

## Usage

### Classification

```bash
python main.py \
    --file data/dataset.csv \
    --target target_column \
    --task classification
```

### Regression

```bash
python main.py \
    --file data/dataset.csv \
    --target target_column \
    --task regression
```

### Development Mode (verbose diagnostics)

```bash
python main.py \
    --file data/dataset.csv \
    --target target_column \
    --task classification \
    --dev-mode
```

---

## Installation

```bash
git clone https://github.com/armaanjain-byte/ML_Pipeline_Debugger.git
cd ML_Pipeline_Debugger

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Project Structure

```
ML_Pipeline_Debugger/
│
├── app/
│   ├── core/          # configuration, exception handling
│   ├── debugger/      # diagnostic engine, recommendation engine
│   ├── pipeline/      # data loading, preprocessing, model orchestration
│   └── utils/         # logging, helpers
│
├── data/              # example datasets
├── tests/             # unit and integration tests
├── docs/              # screenshots
├── main.py
└── requirements.txt
```

---

## Tech Stack

| Component | Library |
|---|---|
| Data | Pandas, NumPy |
| Modeling | Scikit-learn |
| Anomaly Detection | Scikit-learn (Isolation Forest) |
| Statistical Tests | SciPy |
| Testing | Pytest |
| UI | Streamlit |

---

## Current Limitations

- Feature importance extraction is model-dependent (not available for all sklearn estimators)
- DAG-style pipeline tracing is limited to sklearn-compatible workflows
- No distributed execution support
- No MLflow or Airflow integration yet

---

## Roadmap

- [ ] MLflow experiment tracking integration
- [ ] Airflow DAG tracing support
- [ ] Dataset drift monitoring (train-time vs inference-time distribution shift)
- [ ] Visual report layer (HTML export)
- [ ] Distributed execution support

---

## Author

**Armaan Jain** · [github.com/armaanjain-byte](https://github.com/armaanjain-byte)
