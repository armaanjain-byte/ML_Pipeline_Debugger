import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List

class Preprocessor:
    """
    Handles data preprocessing with STRICT ZERO leakage and advanced observability.
    Uses RobustScaler for outlier protection and handles high cardinality safely.
    """
    
    def __init__(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Protect against imbalance if target is categorical
        is_classification = y.nunique() < 20 and y.dtype in ['object', 'category', 'int64']
        stratify_col = y if is_classification else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
        return X_train, X_test, y_train, y_test
    
    def build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        
        transformers = []
        
        if numeric_cols:
            # add_indicator=True preserves information about missingness which is often predictive
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler())  # Better than Standard against anomalies
            ])
            transformers.append(("num", numeric_transformer, numeric_cols))
        
        if categorical_cols:
            # min_frequency=0.01 prevents explosion from high cardinality by grouping rare cats
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore", min_frequency=0.01))
            ])
            transformers.append(("cat", categorical_transformer, categorical_cols))
        
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        self.preprocessor.fit(X_train)
        return self.preprocessor
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Executes the full protected split and transform cycle."""
        X_train, X_test, y_train, y_test = self.split_data(df)
        self.build_preprocessor(X_train)
        
        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        self.feature_names = self._get_feature_names()
        return X_train_transformed, X_test_transformed, y_train.values, y_test.values, self.feature_names
    
    def _get_feature_names(self) -> List[str]:
        if self.preprocessor is None:
            return []
            
        try:
            # Standard in modern scikit-learn
            return list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback for older environments
            feature_names = []
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == "num":
                    feature_names.extend(cols)
                    # Account for Missing Indicator columns
                    if transformer.named_steps["imputer"].add_indicator:
                        missing_mask = transformer.named_steps["imputer"].indicator_.features_
                        feature_names.extend([f"{cols[i]}_is_missing" for i in missing_mask])
                elif name == "cat":
                    ohe = transformer.named_steps["cat"]
                    if hasattr(ohe, "get_feature_names_out"):
                        feature_names.extend(ohe.get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)
            return feature_names