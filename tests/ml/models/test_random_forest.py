"""
Tests for Random Forest regression model.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.models.random_forest import RandomForestModel

@pytest.fixture
def features() -> pd.DataFrame:
    """Return sample training features."""


    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.fixture
def target() -> pd.Series:
    """Return sample training target."""


    return pd.Series(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        name="target",
    )


@pytest.fixture
def model() -> RandomForestModel:
    """Return a configured Random Forest model."""


    return RandomForestModel(
        n_estimators=50,
        random_state=42,
        n_jobs=1,
    )


def test_model_name() -> None:
    """Model name must be random_forest."""


    model = RandomForestModel(
        n_estimators=10,
        random_state=42,
        n_jobs=1,
    )

    assert model.name == "random_forest"


def test_default_configuration() -> None:
    """Default configuration must be valid."""


    model = RandomForestModel(
        n_jobs=1,
    )

    assert model.n_estimators == 200
    assert model.max_depth is None
    assert model.min_samples_split == 2
    assert model.min_samples_leaf == 1
    assert model.random_state == 42
    assert model.n_jobs == 1


def test_model_is_not_fitted_on_initialization() -> None:
    """Model must not be fitted after initialization."""

    model = RandomForestModel(
        n_estimators=10,
        random_state=42,
        n_jobs=1,
    )

    with pytest.raises(Exception):
        model.predict(
            pd.DataFrame(
                {
                    "feature_1": [1],
                    "feature_2": [10],
                }
            )
        )

def test_get_params() -> None:
    """Model parameters must be exposed."""

    model = RandomForestModel(
        n_estimators=50,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=123,
        n_jobs=1,
    )

    params = model.get_params()

    assert params == {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "random_state": 123,
        "n_jobs": 1,
    }


def test_fit_returns_model(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Fit must return the same model instance."""


    result = model.fit(
        features,
        target,
    )

    assert result is model
    assert model._is_fitted is True


def test_fit_creates_fitted_estimator(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Fit must create a fitted sklearn estimator."""


    model.fit(
        features,
        target,
    )

    assert hasattr(
        model.model,
        "estimators_",
    )

    assert len(
        model.model.estimators_
    ) == 50


def test_predict_returns_series(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Predict must return a pandas Series."""


    model.fit(
        features,
        target,
    )

    predictions = model.predict(
        features,
    )

    assert isinstance(
        predictions,
        pd.Series,
    )


def test_predict_preserves_index(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Predictions must preserve the feature index."""


    features = features.copy()

    features.index = [
        10,
        20,
        30,
        40,
        50,
        60,
    ]

    model.fit(
        features,
        target,
    )

    predictions = model.predict(
        features,
    )

    pd.testing.assert_index_equal(
        predictions.index,
        features.index,
    )


def test_predict_returns_expected_number_of_values(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Prediction count must match the number of input rows."""


    model.fit(
        features,
        target,
    )

    predictions = model.predict(
        features,
    )

    assert len(predictions) == len(features)


def test_predictions_are_numeric(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Predictions must be numeric."""


    model.fit(
        features,
        target,
    )

    predictions = model.predict(
        features,
    )

    assert pd.api.types.is_numeric_dtype(
        predictions
    )


def test_predict_before_fit_raises_error(
    model: RandomForestModel,
    features: pd.DataFrame,
    ) -> None:
    """Prediction before fitting must raise an error."""


    with pytest.raises(Exception):
        model.predict(
            features,
        )


@pytest.mark.parametrize(
    "parameter,value",
    [
    ("n_estimators", 0),
    ("n_estimators", -1),
    ("min_samples_split", 0),
    ("min_samples_split", 1),
    ("min_samples_leaf", 0),
    ("max_depth", 0),
    ("max_depth", -1),
    ],
)
def test_invalid_configuration_raises_error(
    parameter: str,
    value: int,
    ) -> None:
    """Invalid Random Forest configuration must raise an error."""


    kwargs = {
        parameter: value,
    }

    with pytest.raises(DataValidationError):
        RandomForestModel(
            **kwargs,
        )


def test_zero_n_jobs_raises_error() -> None:
    """n_jobs equal to zero must raise an error."""


    with pytest.raises(DataValidationError):
        RandomForestModel(
            n_jobs=0,
        )


def test_invalid_training_features_raises_error(
    model: RandomForestModel,
    target: pd.Series,
    ) -> None:
    """Invalid training features must raise an error."""


    with pytest.raises(Exception):
        model.fit(
            pd.DataFrame(),
            target,
        )


def test_invalid_training_target_raises_error(
    model: RandomForestModel,
    features: pd.DataFrame,
    ) -> None:
    """Invalid training target must raise an error."""


    with pytest.raises(Exception):
        model.fit(
            features,
            pd.Series(dtype=float),
        )


def test_invalid_prediction_features_raises_error(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Invalid prediction features must raise an error."""


    model.fit(
        features,
        target,
    )

    with pytest.raises(Exception):
        model.predict(
            pd.DataFrame(),
        )


def test_random_state_makes_predictions_reproducible(
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Same random state must produce identical predictions."""


    model_1 = RandomForestModel(
        n_estimators=50,
        random_state=42,
        n_jobs=1,
    )

    model_2 = RandomForestModel(
        n_estimators=50,
        random_state=42,
        n_jobs=1,
    )

    model_1.fit(
        features,
        target,
    )

    model_2.fit(
        features,
        target,
    )

    predictions_1 = model_1.predict(
        features,
    )

    predictions_2 = model_2.predict(
        features,
    )

    pd.testing.assert_series_equal(
        predictions_1,
        predictions_2,
    )


def test_model_does_not_modify_training_features(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Fit must not modify the input features."""


    original = features.copy(
        deep=True
    )

    model.fit(
        features,
        target,
    )

    pd.testing.assert_frame_equal(
        features,
        original,
    )


def test_model_does_not_modify_prediction_features(
    model: RandomForestModel,
    features: pd.DataFrame,
    target: pd.Series,
    ) -> None:
    """Predict must not modify the input features."""


    model.fit(
        features,
        target,
    )

    original = features.copy(
        deep=True
    )

    model.predict(
        features,
    )

    pd.testing.assert_frame_equal(
        features,
        original,
    )
