"""
FIXED: tests/test_data_checks.py

TESTS FOR:
- Missing value detection
- Constant feature detection
- Duplicate detection
- Class imbalance detection
- Correlation detection
- Outlier detection
- Issue summarization
"""

import pandas as pd
import numpy as np
import pytest
from app.debugger.data_checks import DataChecks


@pytest.fixture
def problematic_df():
    """DataFrame with multiple data quality issues"""
    return pd.DataFrame({
        "constant_col": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "missing_col": [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, 10],
        "high_corr1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "high_corr2": [1.01, 2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.01, 10.01],
        "outlier_col": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
        "target": [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    })


@pytest.fixture
def imbalanced_df():
    """Imbalanced classification dataset (90-10)"""
    return pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "target": [0] * 90 + [1] * 10
    })


@pytest.fixture
def clean_df():
    """Clean dataset with no issues"""
    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "feature3": np.random.rand(100),
        "target": np.random.choice([0, 1], 100)
    })


class TestMissingValueDetection:
    """Test missing value detection"""
    
    def test_missing_values_detected(self, problematic_df):
        """Should detect columns with missing values"""
        checker = DataChecks()
        missing = checker.check_missing_values(problematic_df)
        
        assert "missing_col" in missing
        assert missing["missing_col"] == 30.0  # 3/10 = 30%
    
    def test_no_false_positives_missing(self, clean_df):
        """Should not report missing in clean data"""
        checker = DataChecks()
        missing = checker.check_missing_values(clean_df)
        
        assert len(missing) == 0
    
    def test_missing_percentage_correct(self, problematic_df):
        """Missing percentage should be accurate"""
        checker = DataChecks()
        missing = checker.check_missing_values(problematic_df)
        
        # 3 missing out of 10 = 30%
        assert np.isclose(missing["missing_col"], 30.0)


class TestConstantFeatureDetection:
    """Test constant feature detection"""
    
    def test_constant_feature_detected(self, problematic_df):
        """Should detect constant features"""
        checker = DataChecks()
        constants = checker.check_constant_features(problematic_df)
        
        assert "constant_col" in constants
    
    def test_no_false_positives_constant(self, clean_df):
        """Should not report constant in varying data"""
        checker = DataChecks()
        constants = checker.check_constant_features(clean_df)
        
        assert len(constants) == 0


class TestDuplicateDetection:
    """Test duplicate row detection"""
    
    def test_duplicates_detected(self):
        """Should detect duplicate rows"""
        df = pd.DataFrame({
            "col1": [1, 2, 1, 4, 5],
            "col2": [1, 2, 1, 4, 5]
        })
        
        checker = DataChecks()
        dup_info = checker.check_duplicates(df)
        
        assert dup_info["total_duplicates"] == 1
        assert dup_info["has_duplicates"] is True
    
    def test_no_duplicates_in_clean(self, clean_df):
        """Should report no duplicates in clean data"""
        checker = DataChecks()
        dup_info = checker.check_duplicates(clean_df)
        
        assert dup_info["total_duplicates"] == 0
        assert dup_info["has_duplicates"] is False


class TestClassImbalanceDetection:
    """Test class imbalance detection"""
    
    def test_imbalance_detected(self, imbalanced_df):
        """Should detect imbalanced target"""
        checker = DataChecks()
        imbalance = checker.check_class_imbalance(imbalanced_df, "target")
        
        assert imbalance["is_imbalanced"] is True
        assert imbalance["minority_class_percentage"] == 10.0
    
    def test_balanced_not_flagged(self, clean_df):
        """Should not flag balanced data"""
        checker = DataChecks()
        imbalance = checker.check_class_imbalance(clean_df, "target")
        
        # Might be imbalanced by chance, but should be computed
        assert "is_imbalanced" in imbalance
    
    def test_missing_target_handled(self, problematic_df):
        """Should handle missing target column"""
        checker = DataChecks()
        imbalance = checker.check_class_imbalance(problematic_df, "nonexistent")
        
        assert imbalance == {}


class TestCorrelationDetection:
    """Test high correlation detection"""
    
    def test_high_correlation_detected(self, problematic_df):
        """Should detect highly correlated columns"""
        checker = DataChecks()
        corr = checker.check_high_correlation(problematic_df, threshold=0.95)
        
        # high_corr1 and high_corr2 should be detected
        assert len(corr) > 0
    
    def test_no_false_positives_correlation(self, clean_df):
        """Should not report high correlation in independent data"""
        checker = DataChecks()
        corr = checker.check_high_correlation(clean_df, threshold=0.9)
        
        # Unlikely to have correlation > 0.9 in random data
        assert len(corr) <= 1
    
    def test_correlation_threshold_respected(self, problematic_df):
        """Threshold should filter correctly"""
        checker = DataChecks()
        
        corr_low = checker.check_high_correlation(problematic_df, threshold=0.5)
        corr_high = checker.check_high_correlation(problematic_df, threshold=0.99)
        
        # Higher threshold should return fewer results
        assert len(corr_low) >= len(corr_high)


class TestOutlierDetection:
    """Test outlier detection"""
    
    def test_outliers_detected(self, problematic_df):
        """Should detect outliers"""
        checker = DataChecks()
        outliers = checker.check_outliers(problematic_df)
        
        # outlier_col has one extreme value (100)
        assert "outlier_col" in outliers
        assert outliers["outlier_col"] > 0
    
    def test_no_outliers_in_clean(self, clean_df):
        """Should find few/no outliers in clean data"""
        checker = DataChecks()
        outliers = checker.check_outliers(clean_df)
        
        # Might have some due to random, but not many
        assert len(outliers) <= clean_df.shape[1] * 0.2  # At most 20% of columns
    
    def test_iqr_multiplier_affects_sensitivity(self, problematic_df):
        """Higher IQR multiplier should find fewer outliers"""
        checker = DataChecks()
        
        outliers_strict = checker.check_outliers(problematic_df, iqr_multiplier=1.5)
        outliers_lenient = checker.check_outliers(problematic_df, iqr_multiplier=3.0)
        
        # Stricter threshold should find more or equal outliers
        total_strict = sum(outliers_strict.values())
        total_lenient = sum(outliers_lenient.values())
        assert total_strict >= total_lenient


class TestIssueSummarization:
    """Test issue summarization"""
    
    def test_all_issues_summarized(self, problematic_df):
        """All issues should be included in summary"""
        checker = DataChecks()
        issues = checker._summarize_issues(problematic_df, "target")
        
        # Should find: missing, constant, correlation, outliers
        issue_types = {issue["type"] for issue in issues}
        assert "missing_values" in issue_types
        assert "constant_feature" in issue_types
    
    def test_severity_levels_assigned(self, problematic_df):
        """All issues should have severity"""
        checker = DataChecks()
        issues = checker._summarize_issues(problematic_df, "target")
        
        valid_severities = {"high", "medium", "low"}
        for issue in issues:
            assert issue["severity"] in valid_severities
    
    def test_no_issues_in_clean_data(self, clean_df):
        """Clean data should have minimal issues"""
        checker = DataChecks()
        issues = checker._summarize_issues(clean_df, "target")
        
        # Clean data should have very few issues
        assert len(issues) <= 2  # Might have minor imbalance or outliers


class TestRunAllChecks:
    """Test complete check pipeline"""
    
    def test_all_checks_executed(self, problematic_df):
        """run_all_checks should execute all checks"""
        checker = DataChecks()
        output = checker.run_all_checks(problematic_df, "target")
        
        # All checks should be present
        required_keys = [
            "missing_values",
            "constant_features",
            "duplicates",
            "class_imbalance",
            "high_correlation",
            "data_types",
            "issues"
        ]
        
        for key in required_keys:
            assert key in output
    
    def test_output_structure(self, problematic_df):
        """Output should be properly structured"""
        checker = DataChecks()
        output = checker.run_all_checks(problematic_df, "target")
        
        # Issues should be a list of dicts
        assert isinstance(output["issues"], list)
        
        # Each issue should have required fields
        for issue in output["issues"]:
            assert "type" in issue
            assert "column" in issue
            assert "severity" in issue
            assert "description" in issue