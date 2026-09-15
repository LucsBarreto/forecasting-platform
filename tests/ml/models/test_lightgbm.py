"""
Tests for LightGBM model.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.models.lightgbm import LightGBMModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return deterministic training data."""

    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    y = pd.Series(
        [10.0, 20.0, 30.0, 40.0, 50.0],
        name="target",
    )

    return X, y


@pytest.fixture
def model() -> LightGBMModel:
    """Return a deterministic LightGBM model."""

    return LightGBMModel(
        params={
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": -1,
        }
    )


def test_model_name(
    model: LightGBMModel,
) -> None:
    """Test model name."""

    assert model.name == "lightgbm"


def test_model_is_not_fitted_on_initialization(
    model: LightGBMModel,
) -> None:
    """Model must not be fitted on initialization."""

    assert model._is_fitted is False


def test_fit_returns_model(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fit must return the model instance."""

    X, y = training_data

    result = model.fit(
        X,
        y,
    )

    assert result is model


def test_fit_creates_fitted_estimator(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fit must update the fitted state."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    assert model._is_fitted is True


def test_predict_returns_series(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predict must return a pandas Series."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    predictions = model.predict(X)

    assert isinstance(
        predictions,
        pd.Series,
    )


def test_predict_preserves_index(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predictions must preserve the input index."""

    X, y = training_data

    X = X.copy()
    X.index = [10, 20, 30, 40, 50]

    model.fit(
        X,
        y,
    )

    predictions = model.predict(X)

    pd.testing.assert_index_equal(
        predictions.index,
        X.index,
    )


def test_predict_returns_expected_number_of_values(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predict must return one value per input row."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_predictions_are_numeric(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predictions must contain numeric values."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    predictions = model.predict(X)

    assert pd.api.types.is_numeric_dtype(
        predictions
    )


def test_prediction_before_fit_raises_error(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Prediction before fitting must raise an error."""

    X, _ = training_data

    with pytest.raises(DataValidationError):
        model.predict(X)


def test_invalid_training_features_raise_error(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Invalid training features must raise an error."""

    _, y = training_data

    X = pd.DataFrame()

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_invalid_training_target_raises_error(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Invalid training target must raise an error."""

    X, _ = training_data

    y = pd.Series(
        dtype=float,
    )

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_mismatched_training_lengths_raise_error(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Features and target must have the same length."""

    X, y = training_data

    y = y.iloc[:-1]

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_invalid_prediction_features_raise_error(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Invalid prediction features must raise an error."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    invalid_X = pd.DataFrame()

    with pytest.raises(DataValidationError):
        model.predict(
            invalid_X,
        )


def test_model_does_not_modify_training_features(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Training must not modify the input dataframe."""

    X, y = training_data

    original = X.copy(
        deep=True,
    )

    model.fit(
        X,
        y,
    )

    pd.testing.assert_frame_equal(
        X,
        original,
    )


def test_model_does_not_modify_prediction_features(
    model: LightGBMModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Prediction must not modify the input dataframe."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    original = X.copy(
        deep=True,
    )

    model.predict(X)

    pd.testing.assert_frame_equal(
        X,
        original,
    )


def test_random_state_makes_predictions_reproducible(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Same random state must produce reproducible predictions."""

    X, y = training_data

    params = {
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }

    model_1 = LightGBMModel(
        params=params.copy(),
    )

    model_2 = LightGBMModel(
        params=params.copy(),
    )

    model_1.fit(X, y)
    model_2.fit(X, y)

    predictions_1 = model_1.predict(X)
    predictions_2 = model_2.predict(X)

    pd.testing.assert_series_equal(
        predictions_1,
        predictions_2,
    )


def test_get_params_returns_model_parameters(
    model: LightGBMModel,
) -> None:
    """get_params must expose estimator parameters."""

    params = model.get_params()

    assert isinstance(
        params,
        dict,
    )

    assert params["random_state"] == 42