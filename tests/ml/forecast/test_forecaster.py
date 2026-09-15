"""
Tests for Forecaster.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.forecast import Forecaster
from src.ml.models.base_model import BaseModel


class MockModel(BaseModel):
    """Simple model used for testing."""

    @property
    def name(self) -> str:
        """Return model name."""

        return "mock"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        """Return the model without training."""

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Return deterministic predictions."""

        return pd.Series(
            [10.0] * len(X),
            index=X.index,
        )


class NumpyPredictionModel(BaseModel):
    """Model that returns predictions as a NumPy array."""

    @property
    def name(self) -> str:
        """Return model name."""

        return "numpy_prediction"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        """Return the model."""

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ):
        """Return predictions as a list."""

        return [20.0] * len(X)


class InvalidLengthModel(BaseModel):
    """Model that returns an invalid number of predictions."""

    @property
    def name(self) -> str:
        """Return model name."""

        return "invalid_length"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        """Return the model."""

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Return an invalid number of predictions."""

        return pd.Series(
            [1.0],
        )


@pytest.fixture
def features() -> pd.DataFrame:
    """Return sample prediction features."""

    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
        },
        index=[10, 20, 30],
    )


def test_forecaster_stores_model() -> None:
    """Test that the model is stored."""

    model = MockModel()

    forecaster = Forecaster(
        model=model,
    )

    assert forecaster.model is model


def test_predict_returns_series(
    features: pd.DataFrame,
) -> None:
    """Test that predictions are returned as a Series."""

    forecaster = Forecaster(
        model=MockModel(),
    )

    result = forecaster.predict(
        features,
    )

    assert isinstance(
        result,
        pd.Series,
    )


def test_predict_returns_expected_number_of_values(
    features: pd.DataFrame,
) -> None:
    """Test that the number of predictions matches the input."""

    forecaster = Forecaster(
        model=MockModel(),
    )

    result = forecaster.predict(
        features,
    )

    assert len(result) == len(features)


def test_predict_preserves_input_index(
    features: pd.DataFrame,
) -> None:
    """Test that prediction index matches feature index."""

    forecaster = Forecaster(
        model=MockModel(),
    )

    result = forecaster.predict(
        features,
    )

    pd.testing.assert_index_equal(
        result.index,
        features.index,
    )


def test_predict_returns_expected_values(
    features: pd.DataFrame,
) -> None:
    """Test prediction values."""

    forecaster = Forecaster(
        model=MockModel(),
    )

    result = forecaster.predict(
        features,
    )

    expected = pd.Series(
        [10.0, 10.0, 10.0],
        index=features.index,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_predict_does_not_modify_features(
    features: pd.DataFrame,
) -> None:
    """Test that input features are not modified."""

    original = features.copy(
        deep=True,
    )

    forecaster = Forecaster(
        model=MockModel(),
    )

    forecaster.predict(
        features,
    )

    pd.testing.assert_frame_equal(
        features,
        original,
    )


def test_predict_accepts_non_series_model_output(
    features: pd.DataFrame,
) -> None:
    """Test that array-like model predictions are converted to Series."""

    forecaster = Forecaster(
        model=NumpyPredictionModel(),
    )

    result = forecaster.predict(
        features,
    )

    assert isinstance(
        result,
        pd.Series,
    )

    assert result.tolist() == [
        20.0,
        20.0,
        20.0,
    ]

    pd.testing.assert_index_equal(
        result.index,
        features.index,
    )


def test_invalid_prediction_length_raises_error(
    features: pd.DataFrame,
) -> None:
    """Test that invalid prediction length is rejected."""

    forecaster = Forecaster(
        model=InvalidLengthModel(),
    )

    with pytest.raises(
        DataValidationError,
        match="Number of predictions does not match",
    ):
        forecaster.predict(
            features,
        )


def test_empty_features_raise_error() -> None:
    """Test that empty prediction features are rejected."""

    features = pd.DataFrame(
        columns=[
            "feature_a",
            "feature_b",
        ],
    )

    forecaster = Forecaster(
        model=MockModel(),
    )

    with pytest.raises(
        DataValidationError,
        match="Prediction features cannot be empty",
    ):
        forecaster.predict(
            features,
        )


def test_invalid_features_type_raises_error() -> None:
    """Test that invalid feature types are rejected."""

    forecaster = Forecaster(
        model=MockModel(),
    )

    with pytest.raises(
        TypeError,
        match="features must be a pandas DataFrame",
    ):
        forecaster.predict(
            [1, 2, 3],  # type: ignore[arg-type]
        )


def test_none_model_raises_error() -> None:
    """Test that a missing model is rejected."""

    forecaster = Forecaster(
        model=None,  # type: ignore[arg-type]
    )

    features = pd.DataFrame(
        {
            "feature_a": [1.0],
        },
    )

    with pytest.raises(
        DataValidationError,
        match="A model is required",
    ):
        forecaster.predict(
            features,
        )