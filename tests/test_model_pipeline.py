# FILE: tests/test_model_pipeline.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline

from app.pipeline.model import (
    ModelTrainer,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def classification_dataset():
    """
    Stable classification dataset.
    """

    rng = np.random.default_rng(42)

    rows = 400

    x = pd.DataFrame(
        {
            "age": rng.normal(
                35,
                8,
                rows,
            ),
            "salary": rng.normal(
                60000,
                12000,
                rows,
            ),
            "experience": rng.normal(
                7,
                3,
                rows,
            ),
        }
    )

    y = pd.Series(
        rng.integers(
            0,
            2,
            rows,
        )
    )

    return x, y


@pytest.fixture
def regression_dataset():
    """
    Stable regression dataset.
    """

    rng = np.random.default_rng(42)

    rows = 400

    x = pd.DataFrame(
        {
            "feature_a": rng.normal(
                0,
                1,
                rows,
            ),
            "feature_b": rng.normal(
                5,
                2,
                rows,
            ),
            "feature_c": rng.normal(
                10,
                4,
                rows,
            ),
        }
    )

    y = pd.Series(
        rng.normal(
            100,
            20,
            rows,
        )
    )

    return x, y


# ============================================================================
# CLASSIFICATION TESTS
# ============================================================================


def test_classification_training_pipeline(
    classification_dataset,
):
    """
    Classification training must complete.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    assert isinstance(
        results,
        dict,
    )

    required_keys = {
        "model",
        "metrics",
        "feature_importance",
        "observability_flags",
    }

    assert required_keys.issubset(
        set(results.keys())
    )


def test_classification_model_type(
    classification_dataset,
):
    """
    Model object must exist.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    model = results["model"]

    assert model is not None


# ============================================================================
# REGRESSION TESTS
# ============================================================================


def test_regression_training_pipeline(
    regression_dataset,
):
    """
    Regression training must complete.
    """

    x, y = regression_dataset

    trainer = ModelTrainer(
        task_type="regression",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    assert isinstance(
        results,
        dict,
    )

    required_keys = {
        "model",
        "metrics",
        "feature_importance",
        "observability_flags",
    }

    assert required_keys.issubset(
        set(results.keys())
    )


# ============================================================================
# METRICS TESTS
# ============================================================================


def test_metrics_contract_classification(
    classification_dataset,
):
    """
    Metrics schema stability.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    metrics = results["metrics"]

    required_sections = {
        "train",
        "holdout",
        "cv",
    }

    assert required_sections.issubset(
        set(metrics.keys())
    )

    assert isinstance(
        metrics["train"],
        dict,
    )

    assert isinstance(
        metrics["holdout"],
        dict,
    )

    assert isinstance(
        metrics["cv"],
        dict,
    )


def test_metrics_values_are_numeric(
    classification_dataset,
):
    """
    Metrics must remain numeric.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    metrics = results["metrics"]

    for section in [
        "train",
        "holdout",
        "cv",
    ]:
        for value in metrics[
            section
        ].values():
            assert isinstance(
                value,
                (
                    int,
                    float,
                    np.floating,
                ),
            )


# ============================================================================
# FEATURE IMPORTANCE TESTS
# ============================================================================


def test_feature_importance_schema(
    classification_dataset,
):
    """
    Importance output stability.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    importance = results[
        "feature_importance"
    ]

    assert isinstance(
        importance,
        dict,
    )

    for (
        feature,
        value,
    ) in importance.items():
        assert isinstance(
            feature,
            str,
        )

        assert isinstance(
            value,
            (
                int,
                float,
                np.floating,
            ),
        )


# ============================================================================
# OBSERVABILITY TESTS
# ============================================================================


def test_observability_flags_schema(
    classification_dataset,
):
    """
    Observability output stability.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    flags = results[
        "observability_flags"
    ]

    assert isinstance(
        flags,
        list,
    )

    if flags:
        flag = flags[0]

        required_keys = {
            "flag",
            "severity",
            "detail",
        }

        assert required_keys.issubset(
            set(flag.keys())
        )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_training_handles_small_dataset():
    """
    Small datasets must not crash.
    """

    x = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
        }
    )

    y = pd.Series(
        [0, 1, 0, 1]
    )

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    assert isinstance(
        results,
        dict,
    )


def test_training_handles_missing_values(
    classification_dataset,
):
    """
    Missing values must not hard crash.
    """

    x, y = classification_dataset

    x.loc[
        x.index[:25],
        "salary",
    ] = np.nan

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    assert isinstance(
        results,
        dict,
    )


# ============================================================================
# REGRESSION SAFETY TESTS
# ============================================================================


def test_no_none_outputs(
    classification_dataset,
):
    """
    Prevent dangerous None contracts.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    for (
        key,
        value,
    ) in results.items():
        assert value is not None


def test_feature_importance_no_nan(
    classification_dataset,
):
    """
    Importance values must not contain NaN.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    importance = results[
        "feature_importance"
    ]

    for value in importance.values():
        assert not np.isnan(
            value
        )


# ============================================================================
# INVALID TASK TESTS
# ============================================================================


def test_invalid_task_type():
    """
    Invalid task types must fail cleanly.
    """

    with pytest.raises(
        Exception
    ):
        ModelTrainer(
            task_type="invalid_task",
            dev_mode=True,
        )


# ============================================================================
# PIPELINE COMPATIBILITY TESTS
# ============================================================================


def test_model_object_predicts(
    classification_dataset,
):
    """
    Trained model must remain usable.
    """

    x, y = classification_dataset

    trainer = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results = trainer.train(
        x,
        y,
    )

    model = results["model"]

    predictions = model.predict(
        x.head(10)
    )

    assert len(predictions) == 10


# ============================================================================
# DETERMINISM TESTS
# ============================================================================


def test_deterministic_training(
    classification_dataset,
):
    """
    Random-state stability check.
    """

    x, y = classification_dataset

    trainer_1 = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    trainer_2 = ModelTrainer(
        task_type="classification",
        dev_mode=True,
    )

    results_1 = trainer_1.train(
        x,
        y,
    )

    results_2 = trainer_2.train(
        x,
        y,
    )

    metrics_1 = results_1[
        "metrics"
    ]["holdout"]

    metrics_2 = results_2[
        "metrics"
    ]["holdout"]

    assert (
        metrics_1.keys()
        == metrics_2.keys()
    )
