# FILE: app/pipeline/preprocessing.py


from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class Preprocessor:
    """
    Centralized preprocessing layer with:
    - deterministic feature ordering
    - sklearn compatibility handling
    - stable feature name extraction
    - leakage-safe train/test handling
    - robust missing-value support

    IMPORTANT:
    This class intentionally preserves the original analytical behavior
    while stabilizing compatibility and schema consistency.
    """

    def __init__(
        self,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor: ColumnTransformer | None = None
        self.feature_names: List[str] = []

        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []

    # ==========================================================
    # Public API
    # ==========================================================

    def split_data(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Performs a leakage-safe train/test split.
        """

        self._validate_target_column(dataframe)

        features = dataframe.drop(columns=[self.target_column])
        target = dataframe[self.target_column]

        stratify_target = self._get_stratify_target(target)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_target,
        )

        return X_train, X_test, y_train, y_test

    def build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """
        Builds and fits the preprocessing pipeline.

        The implementation preserves:
        - original scaling behavior
        - original imputation behavior
        - original categorical encoding behavior

        while improving:
        - sklearn compatibility
        - deterministic naming
        - feature ordering stability
        """

        self.numeric_columns = sorted(
            X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        )

        self.categorical_columns = sorted(
            X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        )

        transformers = []

        if self.numeric_columns:
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                        ),
                    ),
                    (
                        "scaler",
                        RobustScaler(),
                    ),
                ]
            )

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns,
                )
            )

        if self.categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        self._build_one_hot_encoder(),
                    ),
                ]
            )

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns,
                )
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0,
        )

        self.preprocessor.fit(X_train)

        self.feature_names = self._extract_feature_names()

        return self.preprocessor

    def preprocess(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Full preprocessing execution pipeline.
        """

        X_train, X_test, y_train, y_test = self.split_data(dataframe)

        self.build_preprocessor(X_train)

        if self.preprocessor is None:
            raise RuntimeError("Preprocessor was not initialized correctly.")

        X_train_transformed = self.preprocessor.transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        X_train_transformed = self._ensure_dense_array(X_train_transformed)
        X_test_transformed = self._ensure_dense_array(X_test_transformed)

        return (
            X_train_transformed,
            X_test_transformed,
            y_train.to_numpy(),
            y_test.to_numpy(),
            self.feature_names,
        )

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    def _validate_target_column(self, dataframe: pd.DataFrame) -> None:
        if self.target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available columns: {list(dataframe.columns)}"
            )

    def _get_stratify_target(self, target: pd.Series):
        """
        Preserves original classification stratification logic
        while making dtype handling more robust.
        """

        is_classification = (
            target.nunique(dropna=False) < 20
            and (
                pd.api.types.is_object_dtype(target)
                or pd.api.types.is_categorical_dtype(target)
                or pd.api.types.is_integer_dtype(target)
                or pd.api.types.is_bool_dtype(target)
            )
        )

        return target if is_classification else None

    def _build_one_hot_encoder(self) -> OneHotEncoder:
        """
        sklearn compatibility shim.

        Handles:
        - sklearn >= 1.2 using sparse_output
        - sklearn < 1.2 using sparse
        """

        encoder_kwargs = {
            "handle_unknown": "ignore",
            "min_frequency": 0.01,
        }

        try:
            return OneHotEncoder(
                sparse_output=False,
                **encoder_kwargs,
            )
        except TypeError:
            return OneHotEncoder(
                sparse=False,
                **encoder_kwargs,
            )

    def _extract_feature_names(self) -> List[str]:
        """
        Deterministic feature-name extraction.

        Critical goals:
        - preserve transformed ordering
        - stabilize naming across sklearn versions
        - avoid fallback corruption
        - prevent feature importance misalignment
        """

        if self.preprocessor is None:
            return []

        extracted_feature_names: List[str] = []

        for transformer_name, transformer, original_columns in self.preprocessor.transformers_:
            if transformer_name == "remainder":
                continue

            if transformer_name == "numeric":
                numeric_feature_names = self._extract_numeric_feature_names(
                    transformer=transformer,
                    original_columns=list(original_columns),
                )

                extracted_feature_names.extend(numeric_feature_names)

            elif transformer_name == "categorical":
                categorical_feature_names = self._extract_categorical_feature_names(
                    transformer=transformer,
                    original_columns=list(original_columns),
                )

                extracted_feature_names.extend(categorical_feature_names)

        return extracted_feature_names

    def _extract_numeric_feature_names(
        self,
        transformer: Pipeline,
        original_columns: List[str],
    ) -> List[str]:
        """
        Preserves deterministic numeric feature ordering.
        """

        numeric_feature_names = list(original_columns)

        imputer = transformer.named_steps.get("imputer")

        if imputer is not None and getattr(imputer, "add_indicator", False):
            indicator = getattr(imputer, "indicator_", None)

            if indicator is not None:
                missing_feature_indices = getattr(indicator, "features_", [])

                for feature_index in missing_feature_indices:
                    if feature_index < len(original_columns):
                        original_name = original_columns[feature_index]
                        numeric_feature_names.append(
                            f"{original_name}__missing_indicator"
                        )

        return numeric_feature_names

    def _extract_categorical_feature_names(
        self,
        transformer: Pipeline,
        original_columns: List[str],
    ) -> List[str]:
        """
        Stable categorical feature extraction.
        """

        encoder = transformer.named_steps.get("encoder")

        if encoder is None:
            return original_columns

        try:
            encoded_feature_names = encoder.get_feature_names_out(original_columns)
            return [str(feature_name) for feature_name in encoded_feature_names]

        except Exception:
            fallback_feature_names: List[str] = []

            categories = getattr(encoder, "categories_", None)

            if categories is None:
                return original_columns

            for column_name, category_values in zip(original_columns, categories):
                for category_value in category_values:
                    fallback_feature_names.append(
                        f"{column_name}__{category_value}"
                    )

            return fallback_feature_names

    def _ensure_dense_array(self, transformed_matrix):
        """
        Ensures deterministic ndarray output across sklearn versions.
        """

        if hasattr(transformed_matrix, "toarray"):
            return transformed_matrix.toarray()

        return np.asarray(transformed_matrix)

