import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, task_type: str, estimator=None, dev_mode: bool = False):
        self.task_type = task_type
        # If no estimator is provided, default to RF, but allow injection
        self.custom_estimator = estimator 
        self.dev_mode = dev_mode
        self.pipeline = None
    def _build_pipeline(self, X: pd.DataFrame):
        """Dynamically builds a preprocessing pipeline based on data types."""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        for col in categorical_features:
             unique_count = X[col].nunique()
             if unique_count > 100:
                  logger.warning(f"Feature '{col}' has high cardinality ({unique_count} unique values). Consider Target Encoding.")

# Use sparse_output=True if the data is large to save memory
        categorical_transformer = Pipeline(steps=[
             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) 
        ])
        # Route 1: Scale numbers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

    
        # Combine routes
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Define the base estimator (Keeping dev-mode settings for speed)
        if self.dev_mode:
            # Lightweight settings for 3-second testing
            model_kwargs = {"n_estimators": 10, "max_depth": 5, "random_state": self.random_state, "n_jobs": -1}
        else:
            # Production settings (What your config.py actually intended)
            model_kwargs = {"n_estimators": 100, "max_depth": None, "random_state": self.random_state, "n_jobs": -1}

        # Define the base estimator using the dynamic kwargs
        if self.task_type == "regression":
            estimator = RandomForestRegressor(**model_kwargs)
        elif self.task_type == "classification":
            estimator = RandomForestClassifier(**model_kwargs)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # The final pipeline object
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Builds the pipeline and trains the model."""
        self.feature_names_in_ = X_train.columns.tolist()
        self._build_pipeline(X_train)
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X_test)

    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        if self.task_type == "classification":
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        else:
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """Runs Cross-Validation using the entire robust pipeline."""
        if self.pipeline is None:
            self._build_pipeline(X_train)
            
        scoring = 'f1_weighted' if self.task_type == 'classification' else 'neg_root_mean_squared_error'
        scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        if self.task_type == 'regression':
            scores = -scores # Scikit-learn returns negative MSE/RMSE

        return {
            f"cv_mean_{scoring.replace('neg_', '')}": float(scores.mean()),
            f"cv_std_{scoring.replace('neg_', '')}": float(scores.std()),
            "cv_folds": cv
        }
    # app/pipeline/model.py

def feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Extracts feature importance with correct mapping for encoded features."""
        if self.pipeline is None:
            return {}
    
        try:
            # 1. Get the trained estimator
            estimator = self.pipeline.named_steps['estimator']
            importances = estimator.feature_importances_
            
            # 2. Get the transformed feature names from the preprocessor
            preprocessor = self.pipeline.named_steps['preprocessor']
            # This correctly handles the expansion from OneHotEncoder
            transformed_names = preprocessor.get_feature_names_out()
            
            # 3. Map and sort
            feat_imp = dict(zip(transformed_names, importances))
            return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {str(e)}")
            return {}

Model.random_state = 42
