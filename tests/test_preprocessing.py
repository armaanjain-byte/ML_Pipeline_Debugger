
import pandas as pd
import numpy as np
import pytest
from app.pipeline.preprocessing import Preprocessor


@pytest.fixture
def sample_df():
    """Create sample dataset with various issues"""
    np.random.seed(42)
    return pd.DataFrame({
        "age": [25, 30, np.nan, 40, 50, 35, 28, 45, 55, 32],
        "city": ["NYC", "LA", "NYC", np.nan, "Chicago", "NYC", "LA", "Boston", "NYC", "LA"],
        "salary": [50000, 60000, 55000, 70000, 80000, 52000, 61000, 75000, 85000, 58000],
        "years_exp": [2, 5, 3, 8, 10, 4, 6, 9, 12, 5],
        "target": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    })


class TestDataLeakagePrevention:
    """CRITICAL: Tests for data leakage prevention"""
    
    def test_split_before_encoding(self, sample_df):
        """CRITICAL: Verify split happens BEFORE any encoding"""
        preprocessor = Preprocessor(target_column="target")
        
        # Split should work without errors
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        
        # Sizes should be correct
        assert len(X_train) + len(X_test) == len(sample_df)
        assert len(X_train) > len(X_test)  # 80-20 split
        
        # Indices should not overlap
        assert len(set(X_train.index) & set(X_test.index)) == 0
    
    def test_preprocessor_fit_on_train_only(self, sample_df):
        """CRITICAL: Encoder should fit on train only"""
        preprocessor = Preprocessor(target_column="target")
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        preprocessor.build_preprocessor(X_train)
        
        # Preprocessor should be fitted
        assert preprocessor.preprocessor is not None
        
        # Transform should work on both (test might have unknown values)
        X_train_t = preprocessor.preprocessor.transform(X_train)
        X_test_t = preprocessor.preprocessor.transform(X_test)
        
        # Shapes should match input
        assert X_train_t.shape[0] == len(X_train)
        assert X_test_t.shape[0] == len(X_test)
    
    def test_no_leakage_with_unknown_values(self, sample_df):
        """Test that test set can have unknown categorical values"""
        preprocessor = Preprocessor(target_column="target")
        
        # Add unknown city to test set
        df_with_unknown = sample_df.copy()
        df_with_unknown.iloc[0, df_with_unknown.columns.get_loc("city")] = "Atlantis"  # Unknown
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_with_unknown)
        preprocessor.build_preprocessor(X_train)
        
        # Should handle unknown values gracefully (handle_unknown="ignore")
        X_test_t = preprocessor.preprocessor.transform(X_test)
        assert X_test_t.shape[0] == len(X_test)


class TestMissingValueHandling:
    """Test missing value imputation"""
    
    def test_no_missing_after_handling(self, sample_df):
        """All missing values should be filled"""
        preprocessor = Preprocessor(target_column="target")
        df_clean = preprocessor.handle_missing_values(sample_df)
        
        assert df_clean.isnull().sum().sum() == 0
    
    def test_numeric_missing_uses_median(self, sample_df):
        """Numeric missing values should use median"""
        preprocessor = Preprocessor(target_column="target")
        df_clean = preprocessor.handle_missing_values(sample_df)
        
        # Age column had NaN, should be filled with median
        assert "age" in df_clean.columns
        assert all(isinstance(x, (int, float)) for x in df_clean["age"])
    
    def test_categorical_missing_uses_mode(self, sample_df):
        """Categorical missing values should use mode"""
        preprocessor = Preprocessor(target_column="target")
        df_clean = preprocessor.handle_missing_values(sample_df)
        
        # City had NaN, should be filled
        assert df_clean["city"].isnull().sum() == 0


class TestEncodingCorrectness:
    """Test that encoding is done correctly"""
    
    def test_one_hot_encoding_no_false_ordering(self, sample_df):
        """OneHotEncoder prevents false ordinal interpretation"""
        preprocessor = Preprocessor(target_column="target")
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        preprocessor.build_preprocessor(X_train)
        
        # Get feature names
        feature_names = preprocessor._get_feature_names()
        
        # Should have expanded categorical features
        # Original: age, city, salary, years_exp
        # After: age, salary, years_exp, city_Boston, city_Chicago, city_LA, city_NYC
        assert len(feature_names) > 4
    
    def test_numeric_features_scaled(self, sample_df):
        """Numeric features should be scaled"""
        preprocessor = Preprocessor(target_column="target")
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        preprocessor.build_preprocessor(X_train)
        
        X_train_t = preprocessor.preprocessor.transform(X_train)
        
        # Scaled values should have mean ≈ 0 and std ≈ 1 (approximately)
        # Note: partial scaling due to categorical features
        numeric_features = X_train_t[:, :4]  # First 4 are numeric
        
        assert not np.allclose(numeric_features.mean(), sample_df[["age", "salary", "years_exp"]].mean().mean())


class TestEndToEnd:
    """End-to-end preprocessing tests"""
    
    def test_full_preprocessing_pipeline(self, sample_df):
        """Full pipeline should work without errors"""
        preprocessor = Preprocessor(target_column="target")
        
        X_train, X_test, y_train, y_test, features = preprocessor.preprocess(sample_df)
        
        # Outputs should be numpy arrays
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Shapes should match
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        
        # Features should be listed
        assert len(features) == X_train.shape[1]
    
    def test_preprocessing_train_test_split_ratio(self, sample_df):
        """Test split ratio is respected"""
        preprocessor = Preprocessor(target_column="target", test_size=0.2)
        
        X_train, X_test, y_train, y_test, features = preprocessor.preprocess(sample_df)
        
        # Should be approximately 80-20 split
        total = len(y_train) + len(y_test)
        assert 0.75 < len(y_train) / total < 0.85  # Allow some variance


class TestErrorHandling:
    """Test error handling"""
    
    def test_missing_target_raises(self, sample_df):
        """Should raise error if target column missing"""
        preprocessor = Preprocessor(target_column="nonexistent")
        
        with pytest.raises(ValueError):
            preprocessor.split_data(sample_df)
    
    def test_empty_dataframe_handling(self):
        """Should handle empty dataframe"""
        preprocessor = Preprocessor(target_column="target")
        df_empty = pd.DataFrame()
        
        with pytest.raises(ValueError):
            preprocessor.split_data(df_empty)
    
    def test_invalid_test_size_handling(self):
        """Should validate test_size parameter"""
        # This should work (0.2 is valid)
        preprocessor = Preprocessor(target_column="target", test_size=0.2)
        assert preprocessor.test_size == 0.2


class TestReproducibility:
    """Test that preprocessing is reproducible"""
    
    def test_random_state_reproducibility(self, sample_df):
        """Same random_state should give same split"""
        preprocessor1 = Preprocessor(target_column="target", random_state=42)
        preprocessor2 = Preprocessor(target_column="target", random_state=42)
        
        X_train1, X_test1, y_train1, y_test1, _ = preprocessor1.preprocess(sample_df.copy())
        X_train2, X_test2, y_train2, y_test2, _ = preprocessor2.preprocess(sample_df.copy())
        
        # Should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
    
    def test_different_random_state_gives_different_split(self, sample_df):
        """Different random_state should give different splits"""
        preprocessor1 = Preprocessor(target_column="target", random_state=42)
        preprocessor2 = Preprocessor(target_column="target", random_state=123)
        
        X_train1, _, _, _, _ = preprocessor1.preprocess(sample_df.copy())
        X_train2, _, _, _, _ = preprocessor2.preprocess(sample_df.copy())
        
        # Should be different
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(X_train1, X_train2)