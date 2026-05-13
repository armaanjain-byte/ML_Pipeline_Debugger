"""
FIXED: app/pipeline/model.py

CRITICAL CHANGES:
- Added F1, precision, recall (classification)
- Added R², MAE (regression)
- Added ROC-AUC (binary classification)
- Better feature importance ranking
- Proper error handling
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from typing import Dict, Any, Optional

class Model:
    """ML model wrapper with comprehensive metrics"""
    
    def __init__(self, task_type: str = "classification"):
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        self.task_type = task_type
        
        if task_type == "classification":
            self.model = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        """Get probability predictions (classification only)"""
        if self.task_type != "classification":
            return None
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        
        return None
    
    def evaluate(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model with COMPREHENSIVE metrics.
        
        Classification:
        - accuracy
        - precision (weighted)
        - recall (weighted)
        - f1 (weighted)
        - roc_auc (binary only)
        
        Regression:
        - rmse
        - mae
        - r2
        """
        if self.task_type == "classification":
            return self._evaluate_classification(y_test, y_pred, y_pred_proba)
        else:
            return self._evaluate_regression(y_test, y_pred)
    
    def _evaluate_classification(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Classification metrics"""
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        }
        
        # ROC-AUC for binary classification only
        if len(set(y_test)) == 2 and y_pred_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
            except Exception:
                metrics["roc_auc"] = None
        
        return metrics
    
    def _evaluate_regression(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Regression metrics"""
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred))
        }
    
    def feature_importance(self, feature_names: list) -> Dict[str, float]:
        """
        Get feature importance ranked by importance.
        
        Returns:
            dict: {feature_name: importance_score} (sorted descending)
        """
        if not hasattr(self.model, "feature_importances_"):
            return {}
        
        importances = self.model.feature_importances_
        
        if len(feature_names) != len(importances):
            raise ValueError(
                f"Mismatch: {len(feature_names)} feature names vs {len(importances)} importance scores"
            )
        
        # Rank by importance (descending)
        ranked = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(ranked)