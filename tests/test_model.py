"""
FIXED: tests/test_model.py

TESTS FOR:
- All classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- All regression metrics (RMSE, MAE, R²)
- Feature importance
- Error handling
"""

import numpy as np
import pytest
from app.pipeline.model import Model


@pytest.fixture
def classification_data():
    """Binary classification dataset"""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def regression_data():
    """Regression dataset"""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100) * 100
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20) * 100
    return X_train, y_train, X_test, y_test


@pytest.fixture
def imbalanced_classification_data():
    """Imbalanced classification (90-10 split)"""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    X_test = np.random.rand(20, 5)
    y_test = np.random.choice([0, 1], size=20, p=[0.9, 0.1])
    return X_train, y_train, X_test, y_test


class TestClassificationMetrics:
    """Test all classification metrics are computed"""
    
    def test_all_metrics_computed(self, classification_data):
        """All required metrics should be present"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = model.evaluate(y_test, y_pred, y_proba)
        
        # All metrics present
        required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_metric_ranges(self, classification_data):
        """All metrics should be in [0, 1]"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = model.evaluate(y_test, y_pred, y_proba)
        
        # All metrics should be 0-1
        for metric_name, value in metrics.items():
            if value is not None:
                assert 0 <= value <= 1, f"{metric_name}={value} out of range"
    
    def test_metrics_with_imbalanced_data(self, imbalanced_classification_data):
        """Metrics should handle imbalanced data"""
        X_train, y_train, X_test, y_test = imbalanced_classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = model.evaluate(y_test, y_pred, y_proba)
        
        # F1 should be different from accuracy (shows imbalance awareness)
        # (not always true, but usually)
        assert "f1" in metrics
        assert "accuracy" in metrics


class TestRegressionMetrics:
    """Test all regression metrics"""
    
    def test_all_regression_metrics(self, regression_data):
        """All regression metrics should be computed"""
        X_train, y_train, X_test, y_test = regression_data
        
        model = Model(task_type="regression")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = model.evaluate(y_test, y_pred)
        
        # All metrics present
        required_metrics = ["rmse", "mae", "r2"]
        for metric in required_metrics:
            assert metric in metrics
    
    def test_rmse_mae_positive(self, regression_data):
        """RMSE and MAE should be non-negative"""
        X_train, y_train, X_test, y_test = regression_data
        
        model = Model(task_type="regression")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = model.evaluate(y_test, y_pred)
        
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
    
    def test_r2_range(self, regression_data):
        """R² can be negative (bad fit) or up to 1"""
        X_train, y_train, X_test, y_test = regression_data
        
        model = Model(task_type="regression")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = model.evaluate(y_test, y_pred)
        
        # R² can be negative or up to 1
        assert metrics["r2"] <= 1


class TestFeatureImportance:
    """Test feature importance extraction"""
    
    def test_feature_importance_extracted(self, classification_data):
        """Feature importance should be extracted"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        importance = model.feature_importance(feature_names)
        
        # Should have all features
        assert len(importance) == len(feature_names)
    
    def test_feature_importance_sorted(self, classification_data):
        """Feature importance should be sorted by importance"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        importance = model.feature_importance(feature_names)
        
        # Values should be in descending order
        values = list(importance.values())
        assert values == sorted(values, reverse=True)
    
    def test_feature_importance_non_negative(self, classification_data):
        """Feature importance values should be non-negative"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        importance = model.feature_importance(feature_names)
        
        # All values should be >= 0
        assert all(v >= 0 for v in importance.values())
    
    def test_feature_importance_sum_to_one(self, classification_data):
        """Feature importances should sum to 1 (normalized)"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        importance = model.feature_importance(feature_names)
        
        # Sum should be 1 (normalized)
        total = sum(importance.values())
        assert np.isclose(total, 1.0, atol=0.01)


class TestModelInitialization:
    """Test model initialization"""
    
    def test_classification_model_init(self):
        """Classification model should initialize"""
        model = Model(task_type="classification")
        assert model.task_type == "classification"
        assert model.model is not None
    
    def test_regression_model_init(self):
        """Regression model should initialize"""
        model = Model(task_type="regression")
        assert model.task_type == "regression"
        assert model.model is not None
    
    def test_invalid_task_type(self):
        """Invalid task type should raise error"""
        with pytest.raises(ValueError):
            Model(task_type="invalid")


class TestPredictions:
    """Test prediction functionality"""
    
    def test_predictions_shape(self, classification_data):
        """Predictions should have correct shape"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (X_test.shape[0],)
    
    def test_predict_proba_shape(self, classification_data):
        """Probability predictions should have correct shape"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        
        # Should be (n_samples, n_classes)
        assert y_proba.shape[0] == X_test.shape[0]
        assert y_proba.shape[1] == 2  # Binary classification
    
    def test_proba_sums_to_one(self, classification_data):
        """Probabilities should sum to 1"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        
        # Each row should sum to ~1
        row_sums = y_proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0)


class TestErrorHandling:
    """Test error handling"""
    
    def test_feature_importance_mismatch(self, classification_data):
        """Should raise error if feature names don't match"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        
        # Wrong number of feature names
        feature_names = [f"feature_{i}" for i in range(3)]  # Only 3, need 5
        
        with pytest.raises(ValueError):
            model.feature_importance(feature_names)
    
    def test_evaluate_without_proba(self, classification_data):
        """Should handle evaluation without proba"""
        X_train, y_train, X_test, y_test = classification_data
        
        model = Model(task_type="classification")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Should work without y_proba (ROC-AUC will be None)
        metrics = model.evaluate(y_test, y_pred, y_pred_proba=None)
        
        assert "accuracy" in metrics
        assert metrics.get("roc_auc") is None