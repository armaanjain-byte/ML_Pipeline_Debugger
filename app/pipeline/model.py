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
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger(__name__)

class Model:
    random_state = 42  # Class-level random state
    
    def __init__(self, task_type: str, estimator=None, dev_mode: bool = False):
        self.task_type = task_type
        # If no estimator is provided, default to RF, but allow injection
        self.custom_estimator = estimator 
        self.dev_mode = dev_mode
        self.pipeline = None
        self.feature_names_in_ = None
        self._X_train_cache = None
        self._y_train_cache = None

    def _build_pipeline(self, X: pd.DataFrame):
        """Dynamically builds a preprocessing pipeline based on data types."""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        for col in categorical_features:
            unique_count = X[col].nunique()
            if unique_count > 100:
                logger.warning(f"Feature '{col}' has high cardinality ({unique_count} unique values). Consider Target Encoding.")

        # Ensure sparse_output=False so get_feature_names_out and permutation logic are easier to manage
        categorical_transformer = Pipeline(steps=[
             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
        ])
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
    
        # Combine routes
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Apply base estimator
        if self.custom_estimator is not None:
            estimator = self.custom_estimator
        else:
            if self.dev_mode:
                model_kwargs = {"n_estimators": 10, "max_depth": 5, "random_state": self.random_state, "n_jobs": -1}
            else:
                model_kwargs = {"n_estimators": 100, "max_depth": None, "random_state": self.random_state, "n_jobs": -1}

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
        self._X_train_cache = X_train.copy()
        self._y_train_cache = y_train.copy()
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
            scores = -scores

        return {
            f"cv_mean_{scoring.replace('neg_', '')}": float(scores.mean()),
            f"cv_std_{scoring.replace('neg_', '')}": float(scores.std()),
            "cv_folds": cv
        }

    def feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """Extracts feature importance using a robust fallback hierarchy (Tree -> Linear -> Permutation)."""
        if self.pipeline is None:
            return {}
    
        try:
            estimator = self.pipeline.named_steps['estimator']
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            try:
                transformed_names = preprocessor.get_feature_names_out()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(getattr(estimator, 'n_features_in_', 0))]
            
            importances = None

            # Priority 1: Tree-based feature importances
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
            
            # Priority 2: Linear model coefficients
            elif hasattr(estimator, 'coef_'):
                importances = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)
            
            # Priority 3: Permutation Importance Fallback
            else:
                if self._X_train_cache is not None and self._y_train_cache is not None:
                    # Transform X once for the estimator to avoid full pipeline permutation overhead
                    X_transformed = preprocessor.transform(self._X_train_cache)
                    result = permutation_importance(
                        estimator, X_transformed, self._y_train_cache, 
                        n_repeats=5, random_state=self.random_state, n_jobs=-1
                    )
                    importances = result.importances_mean

            if importances is None or len(importances) == 0:
                return {}

            # Normalize importances so they scale properly in visualization
            total_importance = np.sum(importances)
            if total_importance > 0:
                importances = importances / total_importance
            
            feat_imp = dict(zip(transformed_names, importances))
            return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {str(e)}")
            return {}