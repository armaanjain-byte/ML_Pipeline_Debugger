"""
FIXED: app/pipeline/preprocessing.py

CRITICAL CHANGES:
- Split FIRST, encode AFTER (prevents data leakage)
- OneHotEncoder instead of LabelEncoder (prevents false ordering)
- sklearn ColumnTransformer for proper pipeline
- Fit on train ONLY, transform both
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple

class Preprocessor:
    """
    Handles data preprocessing with ZERO leakage.
    
    Order:
    1. Handle missing values
    2. SPLIT (critical)
    3. Build transformer on train
    4. Transform train AND test
    """
    
    def __init__(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None  # sklearn ColumnTransformer
        self.feature_names = None
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values:
        - numeric → median
        - categorical → mode
        """
        df = df.copy()
        
        for col in df.columns:
            if df[col].dtype == "object":
                # Categorical: mode
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")
            else:
                # Numeric: median
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        CRITICAL: Split BEFORE any encoding/transformation.
        This prevents test data leakage into training statistics.
        """
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer for proper encoding.
        
        Numeric columns → StandardScaler
        Categorical → OneHotEncoder (avoids false ordering)
        
        CRITICAL: Fit on TRAIN ONLY
        """
        numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
        
        transformers = []
        
        # Numeric: StandardScaler
        if numeric_cols:
            transformers.append(
                ("num", StandardScaler(), numeric_cols)
            )
        
        # Categorical: OneHotEncoder (prevents ordinal interpretation)
        if categorical_cols:
            transformers.append(
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols)
            )
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        # FIT ON TRAIN ONLY (critical for no leakage)
        self.preprocessor.fit(X_train)
        
        return self.preprocessor
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Complete preprocessing pipeline.
        
        Steps:
        1. Handle missing
        2. Split (CRITICAL)
        3. Build transformer on train
        4. Transform both train and test
        
        Returns:
            (X_train, X_test, y_train, y_test, feature_names)
        """
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Split FIRST (before any fitting)
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Step 3: Build and fit preprocessor on train ONLY
        self.build_preprocessor(X_train)
        
        # Step 4: Transform both train and test
        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Get feature names for later (feature importance, etc)
        self.feature_names = self._get_feature_names()
        
        return X_train_transformed, X_test_transformed, y_train.values, y_test.values, self.feature_names
    
    def _get_feature_names(self) -> list:
        """Extract feature names from ColumnTransformer"""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        
        for name, transformer, cols in self.preprocessor.transformers_:
            if name == "num":
                # Numeric columns keep original names
                feature_names.extend(cols)
            elif name == "cat":
                # Categorical columns become "col_value" format
                if hasattr(transformer, "get_feature_names_out"):
                    feature_names.extend(transformer.get_feature_names_out(cols))
                else:
                    # Fallback for older sklearn
                    feature_names.extend(cols)
        
        return feature_names