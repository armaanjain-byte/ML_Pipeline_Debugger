import pandas as pd
import numpy as np

from typing import Dict, Any, List

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.model_selection import (
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

import logging

logger = logging.getLogger(__name__)


class Model:

    def __init__(
        self,
        task_type: str,
        estimator=None,
        dev_mode: bool = False,
        random_state: int = 42
    ):

        self.task_type = task_type

        self.custom_estimator = estimator

        self.dev_mode = dev_mode

        self.random_state = random_state

        self.pipeline = None

        self.feature_names_in_ = []

    # =====================================================
    # BUILD PIPELINE
    # =====================================================

    def _build_pipeline(
        self,
        X: pd.DataFrame
    ):

        """
        Dynamically builds preprocessing
        and model pipeline.
        """

        numeric_features = (
            X.select_dtypes(
                include=[
                    "int64",
                    "float64"
                ]
            )
            .columns
            .tolist()
        )

        categorical_features = (
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool"
                ]
            )
            .columns
            .tolist()
        )

        # =================================================
        # HIGH CARDINALITY WARNINGS
        # =================================================

        for col in categorical_features:

            unique_count = (
                X[col].nunique()
            )

            if unique_count > 100:

                logger.warning(
                    f"Feature '{col}' has "
                    f"high cardinality "
                    f"({unique_count} unique values). "
                    f"Consider Target Encoding."
                )

        # =================================================
        # TRANSFORMERS
        # =================================================

        categorical_transformer = Pipeline(
            steps=[

                (
                    "onehot",

                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True
                    )
                )
            ]
        )

        numeric_transformer = Pipeline(
            steps=[

                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # =================================================
        # PREPROCESSOR
        # =================================================

        preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numeric_transformer,
                    numeric_features
                ),

                (
                    "cat",
                    categorical_transformer,
                    categorical_features
                )
            ]
        )

        # =================================================
        # MODEL CONFIG
        # =================================================

        if self.dev_mode:

            model_kwargs = {

                "n_estimators":
                    10,

                "max_depth":
                    5,

                "random_state":
                    self.random_state,

                "n_jobs":
                    -1
            }

        else:

            model_kwargs = {

                "n_estimators":
                    100,

                "max_depth":
                    None,

                "random_state":
                    self.random_state,

                "n_jobs":
                    -1
            }

        # =================================================
        # ESTIMATOR
        # =================================================

        if self.custom_estimator is not None:

            estimator = self.custom_estimator

        else:

            if self.task_type == "regression":

                estimator = (
                    RandomForestRegressor(
                        **model_kwargs
                    )
                )

            elif self.task_type == "classification":

                estimator = (
                    RandomForestClassifier(
                        **model_kwargs
                    )
                )

            else:

                raise ValueError(
                    f"Unknown task type: "
                    f"{self.task_type}"
                )

        # =================================================
        # FINAL PIPELINE
        # =================================================

        self.pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "estimator",
                    estimator
                )
            ]
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):

        self.feature_names_in_ = (
            X_train.columns.tolist()
        )

        self._build_pipeline(X_train)

        self.pipeline.fit(
            X_train,
            y_train
        )

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(
        self,
        X_test: pd.DataFrame
    ) -> np.ndarray:

        return self.pipeline.predict(
            X_test
        )

    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:

        if self.task_type == "classification":

            return {

                "accuracy":
                    float(
                        accuracy_score(
                            y_test,
                            y_pred
                        )
                    ),

                "precision":
                    float(
                        precision_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0
                        )
                    ),

                "recall":
                    float(
                        recall_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0
                        )
                    ),

                "f1":
                    float(
                        f1_score(
                            y_test,
                            y_pred,
                            average="weighted",
                            zero_division=0
                        )
                    )
            }

        else:

            return {

                "rmse":
                    float(
                        np.sqrt(
                            mean_squared_error(
                                y_test,
                                y_pred
                            )
                        )
                    ),

                "mae":
                    float(
                        mean_absolute_error(
                            y_test,
                            y_pred
                        )
                    ),

                "r2":
                    float(
                        r2_score(
                            y_test,
                            y_pred
                        )
                    )
            }

    # =====================================================
    # CROSS VALIDATION
    # =====================================================

    def cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5
    ) -> Dict[str, Any]:

        if self.pipeline is None:

            self._build_pipeline(
                X_train
            )

        scoring = (

            "f1_weighted"

            if self.task_type == "classification"

            else "neg_root_mean_squared_error"
        )

        scores = cross_val_score(

            self.pipeline,

            X_train,

            y_train,

            cv=cv,

            scoring=scoring,

            n_jobs=-1
        )

        if self.task_type == "regression":

            scores = -scores

        return {

            f"cv_mean_{scoring.replace('neg_', '')}":
                float(scores.mean()),

            f"cv_std_{scoring.replace('neg_', '')}":
                float(scores.std()),

            "cv_folds":
                cv
        }

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    def feature_importance(
        self,
        feature_names: List[str]
    ) -> Dict[str, float]:

        """
        Extract feature importances
        from trained estimator.
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
                estimator.feature_importances_
            )

            preprocessor = (
                self.pipeline.named_steps[
                    "preprocessor"
                ]
            )

            transformed_names = (
                preprocessor.get_feature_names_out()
            )

            feat_imp = dict(

                zip(
                    transformed_names,
                    importances
                )
            )

            return dict(

                sorted(

                    feat_imp.items(),

                    key=lambda item: item[1],

                    reverse=True
                )
            )

        except Exception as e:

            logger.warning(
                f"Could not compute "
                f"feature importance: {str(e)}"
            )

            return {}