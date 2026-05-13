# ML Pipeline Debugger & Orchestrator

An automated MLOps diagnostic orchestrator designed to intercept, analyze, and prevent silent failures in Machine Learning pipelines prior to model training.

## Overview

Machine learning models rarely fail due to algorithmic inaccuracies; they fail due to flawed data architecture. Target leakage, extreme multicollinearity, out-of-vocabulary categorical drift, and class imbalances often pass silently through standard preprocessing pipelines, resulting in deployed models that degrade in production.

This tool acts as a middleware validation layer. It intercepts raw datasets, executes a comprehensive suite of statistical diagnostics, generates actionable engineering recommendations, and only proceeds to model compilation if the data architecture is sound.

## Core Capabilities

* **Pre-emptive Data Validation:** Detects zero-variance features, class imbalance (for classification tasks), and high-correlation matrices.
* **Target Leakage Detection:** Scans for variables exhibiting suspiciously high correlation with the dependent variable to prevent artificial accuracy inflation.
* **Zero-Leakage Architecture:** Strict isolation of imputation and scaling parameters to the training distribution using native scikit-learn Pipelines, completely eliminating test-set data leakage.
* **Multivariate Anomaly Detection:** Utilizes Isolation Forests to flag complex statistical outliers across continuous features.
* **Cross-Validated Evaluation:** Implements K-Fold Cross-Validation to assess true model stability and variance, moving beyond simple holdout metrics.
* **Automated Diagnostics:** Outputs a structured JSON payload of specific architectural actions based on failure severity.

## System Architecture

The system utilizes strict Object-Oriented principles, separating the orchestrator layer from the diagnostic engines to allow for modular testing and scaling.

```text
ML_Pipeline_Debugger/
├── app/
│   ├── core/              # Global configurations and Exception classes
│   ├── debugger/          # Diagnostic engines (DataChecks, Recommendations)
│   ├── pipeline/          # Execution layers (DataLoader, Preprocessor, Model)
│   └── utils/             # System logging framework
├── tests/                 # Integration and unit testing suite
├── data/                  # Local directory for dataset ingestion
├── main.py                # Command Line Interface (CLI)
└── requirements.txt       # Environment dependencies

```

## Installation

Clone the repository and initialize the Python environment:

```bash
git clone https://github.com/armaanjain-byte/ML_Pipeline_Debugger.git
cd ML_Pipeline_Debugger
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

## Command Line Interface (CLI) Usage

The orchestrator is accessed via the `main.py` entry point.

```bash
python main.py --file data/dataset.csv --target target_column --task regression

```

**Arguments:**

* `--file` `-f`: Path to the target dataset (CSV format).
* `--target` `-t`: The dependent variable for model prediction.
* `--task`: Type of machine learning task (`regression` or `classification`). Defaults to regression.

## Example Output

When executed on a compromised dataset, the diagnostic engine isolates specific variables and provides resolution strategies:

```json
{
    "recommendations": [
        {
            "type": "target_leakage",
            "column": "suspicious_feature",
            "severity": "critical",
            "description": "Suspected leakage: correlation with target is 0.985",
            "action": "remove_leaking_feature",
            "recommendations": [
                "Drop the feature from the dataset immediately",
                "Verify if this data point is actually available at prediction time"
            ],
            "rationale": "Features perfectly correlated with the target usually indicate data leakage from the future.",
            "urgency": "critical"
        },
        {
            "type": "multivariate_outliers",
            "column": "dataset_wide",
            "severity": "medium",
            "description": "Detected 42 multivariate outliers (3.6%)",
            "action": "investigate_anomalies",
            "recommendations": [
                "Investigate root cause of outliers",
                "Consider robust scaling (median/IQR based)"
            ],
            "rationale": "Multivariate outliers can heavily skew distance-based models and regressions.",
            "urgency": "medium"
        }
    ],
    "total_issues": 2,
    "critical_issues": 1
}

```

## Testing Methodology

This system utilizes `pytest` for pipeline validation. Ensure all dependencies are installed before running the test suite.

```bash
python -m pytest tests/ -v

```