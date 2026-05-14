import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Model:
    """
    Handles model initialization, training, evaluation, and cross-validation.
    """

    def __init__(self, task_type: str = "regression", random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state
        
        if self.task_type == "regression":
            # Added n_estimators and max_depth for lightning-fast training
            self.model = RandomForestRegressor(
                n_estimators=10, 
                max_depth=5, 
                random_state=self.random_state, 
                n_jobs=-1
            )
        elif self.task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=10, 
                max_depth=5, 
                random_state=self.random_state, 
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
    
    

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        return np.array([])

    def cross_validate(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Performs K-Fold cross-validation on the training set to evaluate stability.
        """
        if self.task_type == "regression":
            scoring = 'neg_mean_squared_error'
            scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
            # Convert negative MSE to RMSE
            rmse_scores = np.sqrt(-scores)
            
            return {
                "cv_mean_rmse": float(np.mean(rmse_scores)),
                "cv_std_rmse": float(np.std(rmse_scores)),
                "cv_folds": cv
            }
        else:
            scoring = 'f1_macro'
            scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
            
            return {
                "cv_mean_f1": float(np.mean(scores)),
                "cv_std_f1": float(np.std(scores)),
                "cv_folds": cv
            }

    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        metrics = {}
        if self.task_type == "regression":
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["r2"] = float(r2_score(y_test, y_pred))
        elif self.task_type == "classification":
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["precision"] = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
        return metrics

    def feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        if not hasattr(self.model, "feature_importances_"):
            return {}
            
        importances = self.model.feature_importances_
        
        # Fallback if feature names mismatch
        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
        feature_importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance descending
        sorted_importance = {
            k: float(v) for k, v in sorted(
                feature_importance_dict.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        }
        
        return sorted_importance
    # app/pipeline/model.py


def _build_pipeline(self, X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Bind the preprocessor and the model together
    self.pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', self.model)
    ])