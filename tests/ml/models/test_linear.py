"""
Tests for linear regression model.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.models.linear import LinearRegressionModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return deterministic training data."""

    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [2.0, 4.0, 6.0, 8.0],
        }
    )

    y = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
        name="target",
    )

    return X, y


@pytest.fixture
def fitted_model(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> LinearRegressionModel:
    """Return a fitted linear regression model."""

    X, y = training_data

    model = LinearRegressionModel()

    model.fit(
        X,
        y,
    )

    return model


def test_model_name() -> None:
    """Model must expose the expected name."""

    model = LinearRegressionModel()

    assert model.name == "linear_regression"


def test_model_is_not_fitted_on_creation() -> None:
    """Model must start in an unfitted state."""

    model = LinearRegressionModel()

    assert model.is_fitted is False


def test_fit_returns_model(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fit must return the model instance."""

    X, y = training_data

    model = LinearRegressionModel()

    result = model.fit(
        X,
        y,
    )

    assert result is model


def test_fit_sets_fitted_state(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fit must mark the model as fitted."""

    X, y = training_data

    model = LinearRegressionModel()

    model.fit(
        X,
        y,
    )

    assert model.is_fitted is True


def test_predict_returns_series(
    fitted_model: LinearRegressionModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predict must return a pandas Series."""

    X, _ = training_data

    predictions = fitted_model.predict(X)

    assert isinstance(
        predictions,
        pd.Series,
    )


def test_predict_preserves_index(
    fitted_model: LinearRegressionModel,
) -> None:
    """Predictions must preserve the input index."""

    X = pd.DataFrame(
        {
            "feature_1": [5.0, 6.0],
            "feature_2": [10.0, 12.0],
        },
        index=[10, 20],
    )

    predictions = fitted_model.predict(X)

    assert predictions.index.tolist() == [
        10,
        20,
    ]


def test_predict_has_expected_name(
    fitted_model: LinearRegressionModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predictions must use the standard prediction column name."""

    X, _ = training_data

    predictions = fitted_model.predict(X)

    assert predictions.name == "prediction"


def test_predictions_are_numeric(
    fitted_model: LinearRegressionModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predictions must contain numeric values."""

    X, _ = training_data

    predictions = fitted_model.predict(X)

    assert pd.api.types.is_numeric_dtype(
        predictions
    )


def test_linear_relationship_is_learned(
    fitted_model: LinearRegressionModel,
) -> None:
    """Model must learn the deterministic linear relationship."""

    X = pd.DataFrame(
        {
            "feature_1": [5.0, 6.0],
            "feature_2": [10.0, 12.0],
        }
    )

    predictions = fitted_model.predict(X)

    assert predictions.tolist() == pytest.approx(
        [50.0, 60.0]
    )


def test_fit_rejects_empty_features() -> None:
    """Empty training features must raise an error."""

    X = pd.DataFrame()
    y = pd.Series(
        [1.0, 2.0],
    )

    model = LinearRegressionModel()

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_fit_rejects_empty_target() -> None:
    """Empty training target must raise an error."""

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        }
    )

    y = pd.Series(
        dtype=float,
    )

    model = LinearRegressionModel()

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_fit_rejects_different_lengths() -> None:
    """Features and target must have the same length."""

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    y = pd.Series(
        [1.0, 2.0],
    )

    model = LinearRegressionModel()

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_fit_rejects_missing_features() -> None:
    """Missing feature values must raise an error."""

    X = pd.DataFrame(
        {
            "feature": [1.0, None, 3.0],
        }
    )

    y = pd.Series(
        [1.0, 2.0, 3.0],
    )

    model = LinearRegressionModel()

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_fit_rejects_missing_target() -> None:
    """Missing target values must raise an error."""

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    y = pd.Series(
        [1.0, None, 3.0],
    )

    model = LinearRegressionModel()

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_predict_before_fit_raises_error() -> None:
    """Prediction before fitting must raise an error."""

    model = LinearRegressionModel()

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        }
    )

    with pytest.raises(DataValidationError):
        model.predict(X)


def test_predict_rejects_empty_features(
    fitted_model: LinearRegressionModel,
) -> None:
    """Empty prediction features must raise an error."""

    X = pd.DataFrame()

    with pytest.raises(DataValidationError):
        fitted_model.predict(X)


def test_predict_rejects_missing_features(
    fitted_model: LinearRegressionModel,
) -> None:
    """Missing prediction values must raise an error."""

    X = pd.DataFrame(
        {
            "feature": [1.0, None],
        }
    )

    with pytest.raises(DataValidationError):
        fitted_model.predict(X)


def test_model_preserves_training_features(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fitting must not modify the input feature dataframe."""

    X, y = training_data

    original = X.copy(
        deep=True,
    )

    model = LinearRegressionModel()

    model.fit(
        X,
        y,
    )

    pd.testing.assert_frame_equal(
        X,
        original,
    )


def test_model_preserves_training_target(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fitting must not modify the input target."""

    X, y = training_data

    original = y.copy(
        deep=True,
    )

    model = LinearRegressionModel()

    model.fit(
        X,
        y,
    )

    pd.testing.assert_series_equal(
        y,
        original,
    )