# ML Pipeline Debugger & Orchestrator

An automated MLOps diagnostic middleware designed to intercept, analyze, and prevent silent failures in Machine Learning pipelines prior to model training.

## 📌 Project Overview

Machine learning models rarely fail due to algorithmic inaccuracies; they fail due to flawed data architecture. Issues like target leakage, extreme multicollinearity, and class imbalances often pass silently through standard preprocessing, resulting in models that look accurate in training but degrade in production.

This tool acts as a **middleware validation layer**. It intercepts raw datasets, executes a comprehensive suite of statistical diagnostics, and generates actionable engineering recommendations. It ensures model compilation only proceeds if the underlying data architecture is sound.

## ✨ Core Capabilities

* **Zero-Leakage Architecture**: Enforces strict isolation of imputation and scaling parameters to the training distribution using native scikit-learn Pipelines, eliminating test-set contamination.
* **Pre-emptive Data Validation**: Detects zero-variance features, class imbalances, and high-correlation matrices before resources are spent on training.
* **Target Leakage Detection**: Scans for variables exhibiting suspiciously high correlation with the dependent variable to prevent artificial accuracy inflation.
* **Statistical Anomaly Detection**: Utilizes **Isolation Forests** to flag complex multivariate outliers and IQR methods for univariate anomalies.
* **Automated Diagnostics**: Outputs a structured JSON payload of specific architectural actions based on failure severity (Critical, High, Medium, Low).

## 🏗 System Architecture

The system utilizes strict Object-Oriented principles, separating the orchestrator layer from the diagnostic engines to allow for modular testing and scaling.

```text
ML_Pipeline_Debugger/
├── app/
│   ├── core/              # Global configurations and custom Exception classes
│   ├── debugger/          # Diagnostic engines (DataChecks, Recommendations)
│   ├── pipeline/          # Execution layers (DataLoader, Preprocessor, Model)
│   └── utils/             # System logging framework
├── tests/                 # Integration and unit testing suite
├── data/                  # Local directory for dataset ingestion
├── main.py                # Command Line Interface (CLI)
└── requirements.txt       # Environment dependencies

```

## 🚀 Getting Started

### Installation

Clone the repository and initialize the environment:

```bash
git clone https://github.com/armaanjain-byte/ML_Pipeline_Debugger.git
cd ML_Pipeline_Debugger
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### CLI Usage

The orchestrator is accessed via the `main.py` entry point:

```bash
python main.py --file data/dataset.csv --target target_column --task classification

```

**Arguments:**

* `--file` / `-f`: Path to the target dataset (CSV).
* `--target` / `-t`: The dependent variable for prediction.
* `--task`: Type of ML task (`regression` or `classification`).
* `--dev-mode`: Optional flag to sample data for rapid iteration.

## 📊 Diagnostic Intelligence

When executed, the engine generates a structured diagnostic report:

```json
{
    "type": "target_leakage",
    "column": "suspicious_feature",
    "severity": "critical",
    "description": "Suspected leakage: correlation with target is 0.985",
    "action": "remove_leaking_feature",
    "rationale": "Features perfectly correlated with the target usually indicate data leakage from the future.",
    "urgency": "critical"
}

```

## 🛠 Engineering Standards

* **Robust Preprocessing**: Dynamically builds pipelines using `ColumnTransformer` to handle numeric scaling and categorical encoding (One-Hot) correctly.
* **Cross-Validated Evaluation**: Implements K-Fold Cross-Validation to assess true model stability beyond simple holdout metrics.
* **Defensive Programming**: Includes custom exception handling for data loading and preprocessing failures to ensure graceful system exits.

---

*Developed by Armaan Jain - Focused on building reliable, production-ready AI systems.*