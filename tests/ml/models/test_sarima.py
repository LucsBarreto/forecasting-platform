"""
Tests for SARIMA model.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.models.sarima import SarimaModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return sample training data."""

    index = pd.date_range(
        "2024-01-01",
        periods=36,
        freq="D",
    )

    X = pd.DataFrame(
        {
            "feature": np.arange(36, dtype=float),
        },
        index=index,
    )

    y = pd.Series(
        np.arange(10, 46, dtype=float),
        index=index,
        name="VOLUME",
    )

    return X, y


@pytest.fixture
def model() -> SarimaModel:
    """Return a SARIMA model."""

    return SarimaModel(
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
    )


def test_model_name(
    model: SarimaModel,
) -> None:
    """Test model name."""

    assert model.name == "sarima"


def test_model_is_not_fitted_on_initialization(
    model: SarimaModel,
) -> None:
    """Test that model starts unfitted."""

    assert model._is_fitted is False
    assert model._model is None


def test_fit_returns_model(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that fit returns the model instance."""

    X, y = training_data

    result = model.fit(
        X,
        y,
    )

    assert result is model


def test_fit_creates_fitted_model(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that fit creates the underlying estimator."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    assert model._is_fitted is True
    assert model._model is not None


def test_predict_returns_series(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test prediction return type."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    prediction_features = X.iloc[-5:].copy()

    result = model.predict(
        prediction_features,
    )

    assert isinstance(
        result,
        pd.Series,
    )


def test_predict_preserves_index(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that prediction index matches input index."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    prediction_features = X.iloc[-5:].copy()

    result = model.predict(
        prediction_features,
    )

    pd.testing.assert_index_equal(
        result.index,
        prediction_features.index,
    )


def test_predict_returns_expected_number_of_values(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test prediction length."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    prediction_features = X.iloc[-7:].copy()

    result = model.predict(
        prediction_features,
    )

    assert len(result) == 7


def test_predictions_are_numeric(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that predictions are numeric."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    result = model.predict(
        X.iloc[-5:],
    )

    assert pd.api.types.is_numeric_dtype(
        result
    )

    assert result.notna().all()


def test_predict_before_fit_raises_error(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that prediction before fitting raises an error."""

    X, _ = training_data

    with pytest.raises(DataValidationError):
        model.predict(
            X.iloc[-5:],
        )


def test_invalid_training_features_raise_error(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test invalid training features."""

    _, y = training_data

    with pytest.raises(DataValidationError):
        model.fit(
            pd.Series(
                [1, 2, 3]
            ),
            y.iloc[:3],
        )


def test_invalid_training_target_raises_error(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test invalid training target."""

    X, _ = training_data

    invalid_target = pd.Series(
        ["a", "b", "c"],
    )

    with pytest.raises(DataValidationError):
        model.fit(
            X.iloc[:3],
            invalid_target,
        )


def test_training_data_length_mismatch_raises_error(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test mismatched training data length."""

    X, y = training_data

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y.iloc[:-1],
        )


def test_empty_training_features_raise_error(
    model: SarimaModel,
) -> None:
    """Test empty training features."""

    X = pd.DataFrame()
    y = pd.Series(dtype=float)

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_missing_target_values_raise_error(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test missing target values."""

    X, y = training_data

    y = y.copy()
    y.iloc[5] = np.nan

    with pytest.raises(DataValidationError):
        model.fit(
            X,
            y,
        )


def test_empty_prediction_features_raise_error(
    model: SarimaModel,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test empty prediction features."""

    X, y = training_data

    model.fit(
        X,
        y,
    )

    with pytest.raises(DataValidationError):
        model.predict(
            pd.DataFrame(),
        )


def test_get_params(
    model: SarimaModel,
) -> None:
    """Test model parameter inspection."""

    params = model.get_params()

    assert params["order"] == (1, 0, 0)
    assert params["seasonal_order"] == (
        0,
        0,
        0,
        0,
    )
    assert params["trend"] is None


def test_validate_order_accepts_valid_order() -> None:
    """Test valid SARIMA order."""

    SarimaModel._validate_order(
        (1, 1, 1),
    )


def test_validate_order_rejects_invalid_length() -> None:
    """Test invalid SARIMA order length."""

    with pytest.raises(DataValidationError):
        SarimaModel._validate_order(
            (1, 1),
        )


def test_validate_order_rejects_negative_values() -> None:
    """Test negative SARIMA order values."""

    with pytest.raises(DataValidationError):
        SarimaModel._validate_order(
            (1, -1, 1),
        )


def test_validate_seasonal_order_accepts_valid_order() -> None:
    """Test valid seasonal order."""

    SarimaModel._validate_seasonal_order(
        (1, 1, 1, 12),
    )


def test_validate_seasonal_order_rejects_invalid_length() -> None:
    """Test invalid seasonal order length."""

    with pytest.raises(DataValidationError):
        SarimaModel._validate_seasonal_order(
            (1, 1, 1),
        )


def test_validate_seasonal_order_rejects_negative_values() -> None:
    """Test negative seasonal order values."""

    with pytest.raises(DataValidationError):
        SarimaModel._validate_seasonal_order(
            (1, -1, 1, 12),
        )


def test_validate_seasonal_order_rejects_period_one() -> None:
    """Test invalid seasonal period."""

    with pytest.raises(DataValidationError):
        SarimaModel._validate_seasonal_order(
            (1, 1, 1, 1),
        )