import pytest
import pandas as pd
import numpy as np
from app.debugger.data_checks import DataChecks

@pytest.fixture
def clean_df():
    """Generates a perfectly clean dataset for baseline testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100, p=[0.5, 0.5])
    })

@pytest.fixture
def imbalanced_df():
    """Generates a dataset with severe class imbalance."""
    return pd.DataFrame({
        "feature_1": np.random.randn(100),
        "target": [0] * 90 + [1] * 10
    })

@pytest.fixture
def problematic_df():
    """Generates a dataset with a multitude of statistical flaws."""
    np.random.seed(42)
    df = pd.DataFrame({
        "missing_col": [1.0, np.nan, 3.0, 4.0, 5.0] * 20,
        "constant_col": [1] * 100,
        "feature_a": np.random.randn(100),
        "feature_b": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })
    
    # Inject exact duplicates
    df.loc[99] = df.loc[98]
    
    # Inject multicollinearity
    df["feature_a_correlated"] = df["feature_a"] * 2 + np.random.normal(0, 0.1, 100)
    
    # Inject target leakage
    df["leaking_feature"] = df["target"] * 10 + np.random.normal(0, 0.01, 100)
    
    # Inject multivariate outlier
    df.loc[0, ["feature_a", "feature_b"]] = [15.0, -15.0]
    
    return df

class TestMissingValues:
    def test_missing_values_detected(self, problematic_df):
        checker = DataChecks()
        missing = checker.check_missing_values(problematic_df)
        assert "missing_col" in missing
        assert missing["missing_col"] == 20.0

    def test_clean_df_no_missing(self, clean_df):
        checker = DataChecks()
        assert len(checker.check_missing_values(clean_df)) == 0

class TestConstantFeatures:
    def test_constant_feature_detected(self, problematic_df):
        checker = DataChecks()
        constants = checker.check_constant_features(problematic_df)
        assert "constant_col" in constants
        assert "feature_a" not in constants

class TestDuplicates:
    def test_duplicates_detected(self, problematic_df):
        checker = DataChecks()
        dups = checker.check_duplicates(problematic_df)
        assert dups["has_duplicates"] is True
        assert dups["total_duplicates"] >= 1

class TestClassImbalance:
    def test_imbalance_detected_classification(self, imbalanced_df):
        checker = DataChecks()
        imbalance = checker.check_class_imbalance(imbalanced_df, "target", task_type="classification")
        assert imbalance["is_imbalanced"] is True
        assert imbalance["minority_class_percentage"] == 10.0
        
    def test_regression_skipped(self, imbalanced_df):
        checker = DataChecks()
        imbalance = checker.check_class_imbalance(imbalanced_df, "target", task_type="regression")
        assert imbalance == {}

class TestCorrelationsAndLeakage:
    def test_high_correlation_detected(self, problematic_df):
        checker = DataChecks()
        correlations = checker.check_high_correlation(problematic_df, threshold=0.90)
        corr_columns = [col1 for col1, col2, val in correlations] + [col2 for col1, col2, val in correlations]
        assert "feature_a" in corr_columns
        assert "feature_a_correlated" in corr_columns

    def test_target_leakage_detected(self, problematic_df):
        checker = DataChecks()
        leakage = checker.check_target_leakage(problematic_df, "target", threshold=0.95)
        assert len(leakage) > 0
        leaking_cols = [l["column"] for l in leakage]
        assert "leaking_feature" in leaking_cols

class TestMultivariateOutliers:
    def test_multivariate_outliers_detected(self, problematic_df):
        checker = DataChecks()
        outliers = checker.check_multivariate_outliers(problematic_df)
        assert outliers["has_outliers"] is True
        assert outliers["count"] > 0

class TestExecutionPipeline:
    def test_run_all_checks_structure(self, problematic_df):
        checker = DataChecks()
        output = checker.run_all_checks(problematic_df, "target", task_type="classification")
        
        required_keys = [
            "missing_values", "constant_features", "duplicates", 
            "class_imbalance", "high_correlation", "target_leakage",
            "multivariate_outliers", "data_types", "issues"
        ]
        
        for key in required_keys:
            assert key in output
            
        assert len(output["issues"]) > 0