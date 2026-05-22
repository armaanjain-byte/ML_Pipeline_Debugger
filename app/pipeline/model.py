from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
)

logger = logging.getLogger(__name__)


class Model:
    """
    Stable model orchestration layer.
    """

    RANDOM_STATE = 42

    def __init__(
        self,
        task_type: str,
        estimator=None,
        dev_mode: bool = False,
    ):

        self.task_type = (
            task_type.lower().strip()
        )

        self.custom_estimator = estimator
        self.dev_mode = dev_mode

        self.pipeline: Optional[
            Pipeline
        ] = None

        if self.task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.RANDOM_STATE,
                n_jobs=-1,
            )

        elif self.task_type == "regression":
            self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.RANDOM_STATE,
            n_jobs=-1,
        )

        else:
            self.model = None

        self.feature_names_in_: List[
            str
        ] = []

        self.transformed_feature_names_: List[
            str
        ] = []

        self.numeric_features_: List[
            str
        ] = []

        self.categorical_features_: List[
            str
        ] = []

        self._X_train_cache: Optional[
            pd.DataFrame
        ] = None

        self._y_train_cache: Optional[
            pd.Series
        ] = None

        self._validate_task_type()

    # ==========================================================
    # Public API
    # ==========================================================

    def train(
        self,
        X_train,
        y_train,
    ) -> None:
        """
        Builds and trains pipeline.
        """

        if isinstance(
            X_train,
            np.ndarray,
        ):
            X_train = pd.DataFrame(
                X_train
            )

        if isinstance(
            y_train,
            np.ndarray,
        ):
            y_train = pd.Series(
                y_train
            )

        self._validate_training_data(
            X_train,
            y_train,
        )

        self.feature_names_in_ = list(
            X_train.columns
        )

        self._X_train_cache = (
            X_train.copy()
        )

        self._y_train_cache = (
            y_train.copy()
        )

        self._build_pipeline(
            X_train
        )

        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline construction failed."
            )

        self.pipeline.fit(
            X_train,
            y_train,
        )

        self.model = self.pipeline

        self.transformed_feature_names_ = (
            self._extract_transformed_feature_names()
        )

    def predict(
        self,
        X_test,
    ) -> np.ndarray:
        """
        Stable prediction wrapper.
        """

        if isinstance(
            X_test,
            np.ndarray,
        ):
            X_test = pd.DataFrame(
                X_test
            )

        if self.pipeline is None:
            raise RuntimeError(
                "Model pipeline has not been trained."
            )

        return self.pipeline.predict(
            X_test
        )

    def predict_proba(
        self,
        X_test,
    ):
        """
        Stable probability prediction wrapper.
        """

        if isinstance(
            X_test,
            np.ndarray,
        ):
            X_test = pd.DataFrame(
                X_test
            )

        if self.pipeline is None:
            raise RuntimeError(
                "Model pipeline has not been trained."
            )

        estimator = (
            self.pipeline.named_steps[
                "estimator"
            ]
        )

        if not hasattr(
            estimator,
            "predict_proba",
        ):
            raise AttributeError(
                "Underlying estimator does not support predict_proba."
            )

        return self.pipeline.predict_proba(
            X_test
        )

    def evaluate(
    self,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba=None,
    ) -> Dict[str, float]:
        """
        Stable metric evaluation.
        """

        if (
            self.task_type
            == "classification"
        ):
            return self._evaluate_classification(
                y_test,
                y_pred,
                y_pred_proba,
            )

        return self._evaluate_regression(
            y_test,
            y_pred,
        )

    def cross_validate(
        self,
        X_train,
        y_train,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """
        Robust cross-validation wrapper.
        """

        if isinstance(
            X_train,
            np.ndarray,
        ):
            X_train = pd.DataFrame(
                X_train
            )

        if isinstance(
            y_train,
            np.ndarray,
        ):
            y_train = pd.Series(
                y_train
            )

        if self.pipeline is None:
            self._build_pipeline(
                X_train
            )

        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline construction failed."
            )

        scoring = (
            self._get_cv_scoring_strategy()
        )

        validation_pipeline = clone(
            self.pipeline
        )

        scores = cross_val_score(
            validation_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        if scoring.startswith("neg_"):
            scores = -scores

        normalized_scoring_name = (
            scoring.replace(
                "neg_",
                "",
            )
        )

        return {
            f"cv_mean_{normalized_scoring_name}":
                float(np.mean(scores)),
            f"cv_std_{normalized_scoring_name}":
                float(np.std(scores)),
            "cv_folds": int(cv),
        }

    def feature_importance(
        self,
        feature_names,
    ) -> Dict[str, float]:
        """
        Stable feature-importance extraction.
        """

        if self.pipeline is None:
            return {}

        try:

            estimator = (
                self.pipeline.named_steps[
                    "estimator"
                ]
            )

            importances = (
                self._extract_importance_values(
                    estimator
                )
            )

            if importances is None:
                return {}

            transformed_names = list(
            feature_names
            )

            if (
                    len(transformed_names)
                    != len(importances)
            ):
                raise ValueError(
                "Feature names length does not match importance values."
            )

            normalized_importances = (
                self._normalize_importances(
                    importances
                )
            )

            feature_importance_mapping = {
                str(feature_name):
                float(importance)
                for (
                    feature_name,
                    importance,
                ) in zip(
                    transformed_names,
                    normalized_importances,
                )
            }

            sorted_mapping = dict(
                sorted(
                    feature_importance_mapping.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            return sorted_mapping

        except Exception as error:

            logger.warning(
                "Could not compute feature importance: %s",
                str(error),
            )
            if isinstance(error, ValueError):
                raise
            return {}

    # ==========================================================
    # Pipeline Construction
    # ==========================================================

    def _build_pipeline(
        self,
        X: pd.DataFrame,
    ) -> None:
        """
        Builds sklearn pipeline.
        """

        self.numeric_features_ = sorted(
            X.select_dtypes(
                include=[
                    "int64",
                    "float64",
                    "int32",
                    "float32",
                ],
            ).columns.tolist()
        )

        self.categorical_features_ = sorted(
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool",
                ],
            ).columns.tolist()
        )

        self._log_high_cardinality_features(
            X
        )

        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    RobustScaler(),
                ),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    self._build_one_hot_encoder(),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    self.numeric_features_,
                ),
                (
                    "cat",
                    categorical_transformer,
                    self.categorical_features_,
                ),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

        estimator = (
            self._resolve_estimator()
        )

        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "estimator",
                    estimator,
                ),
            ]
        )

    # ==========================================================
    # Estimator Resolution
    # ==========================================================

    def _resolve_estimator(
        self,
    ):

        if (
            self.custom_estimator
            is not None
        ):
            return self.custom_estimator

        if self.dev_mode:

            estimator_config = {
                "n_estimators": 10,
                "max_depth": 5,
                "random_state":
                    self.RANDOM_STATE,
                "n_jobs": -1,
            }

        else:

            estimator_config = {
                "n_estimators": 100,
                "max_depth": None,
                "random_state":
                    self.RANDOM_STATE,
                "n_jobs": -1,
            }

        if (
            self.task_type
            == "classification"
        ):

            return RandomForestClassifier(
                **estimator_config
            )

        return RandomForestRegressor(
            **estimator_config
        )

    # ==========================================================
    # Metrics
    # ==========================================================

    def _evaluate_classification(
    self,
    y_test,
    y_pred,
    y_pred_proba=None,
) -> Dict[str, float]:

            metrics = {
                "accuracy":
            float(
                accuracy_score(
                    y_test,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "roc_auc": 0.0,
    }

            

            if y_pred_proba is not None:

                try:

                    if len(y_pred_proba.shape) > 1:
                        positive_scores = (
                    y_pred_proba[:, 1]
                    )

                    else:
                        positive_scores = (
                        y_pred_proba
                    )

                    metrics["roc_auc"] = float(
                        roc_auc_score(
                        y_test,
                        positive_scores,
                    )
                    )

                except Exception:
                    metrics["roc_auc"] = 0.0

            return metrics

    def _evaluate_regression(
        self,
        y_test,
        y_pred,
    ) -> Dict[str, float]:

        return {
            "rmse":
                float(
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            y_pred,
                        )
                    )
                ),
            "mae":
                float(
                    mean_absolute_error(
                        y_test,
                        y_pred,
                    )
                ),
            "r2":
                float(
                    r2_score(
                        y_test,
                        y_pred,
                    )
                ),
        }

    # ==========================================================
    # Feature Importance
    # ==========================================================

    def _extract_importance_values(
        self,
        estimator,
    ):

        if hasattr(
            estimator,
            "feature_importances_",
        ):
            return np.asarray(
                estimator.feature_importances_
            )

        if hasattr(
            estimator,
            "coef_",
        ):

            coefficients = np.asarray(
                estimator.coef_
            )

            if coefficients.ndim > 1:
                coefficients = np.mean(
                    np.abs(coefficients),
                    axis=0,
                )
            else:
                coefficients = np.abs(
                    coefficients
                )

            return coefficients

        return self._compute_permutation_importance(
            estimator
        )

    def _compute_permutation_importance(
        self,
        estimator,
    ):

        if (
            self._X_train_cache is None
            or self._y_train_cache is None
        ):
            return None

        if self.pipeline is None:
            return None

        try:

            preprocessor = (
                self.pipeline.named_steps[
                    "preprocessor"
                ]
            )

            transformed_X = (
                preprocessor.transform(
                    self._X_train_cache
                )
            )

            result = permutation_importance(
                estimator,
                transformed_X,
                self._y_train_cache,
                n_repeats=5,
                random_state=self.RANDOM_STATE,
                n_jobs=-1,
            )

            return np.asarray(
                result.importances_mean
            )

        except Exception as error:

            logger.warning(
                "Permutation importance computation failed: %s",
                str(error),
            )

            return None

    def _normalize_importances(
        self,
        importances,
    ) -> np.ndarray:

        importances = np.asarray(
            importances,
            dtype=float,
        )

        total_importance = np.sum(
            importances
        )

        if total_importance <= 0:
            return importances

        return (
            importances
            / total_importance
        )

    # ==========================================================
    # Feature Name Extraction
    # ==========================================================

    def _extract_transformed_feature_names(
        self,
    ) -> List[str]:

        if self.pipeline is None:
            return []

        preprocessor = (
            self.pipeline.named_steps.get(
                "preprocessor"
            )
        )

        if preprocessor is None:
            return []

        try:

            feature_names = (
                preprocessor.get_feature_names_out()
            )

            return [
                str(name)
                for name in feature_names
            ]

        except Exception:

            extracted_names: List[
                str
            ] = []

            for (
                transformer_name,
                transformer,
                columns,
            ) in preprocessor.transformers_:

                if (
                    transformer_name
                    == "remainder"
                ):
                    continue

                column_list = list(
                    columns
                )

                if (
                    transformer_name
                    == "num"
                ):
                    extracted_names.extend(
                        column_list
                    )

                elif (
                    transformer_name
                    == "cat"
                ):

                    extracted_names.extend(
                        self._extract_categorical_feature_names(
                            transformer,
                            column_list,
                        )
                    )

            return extracted_names

    def _extract_categorical_feature_names(
        self,
        transformer,
        columns,
    ) -> List[str]:

        encoder = (
            transformer.named_steps.get(
                "onehot"
            )
        )

        if encoder is None:
            return columns

        try:

            feature_names = (
                encoder.get_feature_names_out(
                    columns
                )
            )

            return [
                str(name)
                for name in feature_names
            ]

        except Exception:

            fallback_names: List[
                str
            ] = []

            categories = getattr(
                encoder,
                "categories_",
                None,
            )

            if categories is None:
                return columns

            for (
                column_name,
                category_values,
            ) in zip(
                columns,
                categories,
            ):

                for category_value in category_values:

                    fallback_names.append(
                        f"{column_name}__{category_value}"
                    )

            return fallback_names

    # ==========================================================
    # Compatibility Helpers
    # ==========================================================

    def _build_one_hot_encoder(
        self,
    ) -> OneHotEncoder:

        encoder_kwargs = {
            "handle_unknown": "ignore",
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

    # ==========================================================
    # Validation & Logging
    # ==========================================================

    def _validate_task_type(
        self,
    ) -> None:

        valid_task_types = {
            "classification",
            "regression",
        }

        if (
            self.task_type
            not in valid_task_types
        ):
            raise ValueError(
                f"Unsupported task type '{self.task_type}'. "
                f"Expected one of {sorted(valid_task_types)}"
            )

    def _validate_training_data(
        self,
        X_train,
        y_train,
    ) -> None:

        if len(X_train) == 0:
            raise ValueError(
                "Training feature set is empty."
            )

        if (
            len(X_train)
            != len(y_train)
        ):
            raise ValueError(
                "Feature and target lengths do not match."
            )

    def _get_cv_scoring_strategy(
        self,
    ) -> str:

        if (
            self.task_type
            == "classification"
        ):
            return "f1_weighted"

        return (
            "neg_root_mean_squared_error"
        )

    def _log_high_cardinality_features(
        self,
        dataframe,
    ) -> None:

        for (
            column_name
        ) in self.categorical_features_:

            unique_count = dataframe[
                column_name
            ].nunique(dropna=False)

            if unique_count > 100:

                logger.warning(
                    "Feature '%s' has high cardinality (%s unique values). "
                    "Consider alternative encoding strategies.",
                    column_name,
                    unique_count,
                )


class ModelTrainer(Model):

    def train(
        self,
        X_train,
        y_train,
    ):

        super().train(
            X_train,
            y_train,
        )

        if isinstance(
            X_train,
            np.ndarray,
        ):
            X_train = pd.DataFrame(
                X_train
            )

        if isinstance(
            y_train,
            np.ndarray,
        ):
            y_train = pd.Series(
                y_train
            )

        predictions = self.predict(
            X_train
        )

        metrics = self.evaluate(
            y_train,
            predictions,
        )

        return {
            "model": self,
            "metrics": {
                "train": metrics,
                "holdout": metrics,
                "cv": {},
            },
            "feature_importance": (
                self.model.feature_importance(
                list(X_train.columns)
                )
            or {}
            ),
            "observability_flags": [],
        }