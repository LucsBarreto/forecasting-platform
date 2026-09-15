"""
Tests for Prophet forecasting model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.models.prophet import ProphetModel


class FakeProphet:
    """Fake Prophet implementation for unit tests."""

    def __init__(self, **params):
        self.params = params
        self.training_data = None

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> FakeProphet:
        self.training_data = dataframe.copy()
        return self

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ds": dataframe["ds"],
                "yhat": range(
                    1,
                    len(dataframe) + 1,
                ),
            }
        )


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Return sample Prophet features."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ]
            ),
            "feature_1": [
                10,
                20,
                30,
                40,
                50,
            ],
        },
        index=[10, 20, 30, 40, 50],
    )


@pytest.fixture
def sample_target() -> pd.Series:
    """Return sample target values."""

    return pd.Series(
        [100, 120, 140, 160, 180],
        index=[10, 20, 30, 40, 50],
        name="VOLUME",
    )


@pytest.fixture
def prophet_model() -> ProphetModel:
    """Return a Prophet model."""

    return ProphetModel(
        date_column="DATA",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )


def test_model_name(
    prophet_model: ProphetModel,
) -> None:
    """Test model name."""

    assert prophet_model.name == "prophet"


def test_model_is_not_fitted_on_initialization(
    prophet_model: ProphetModel,
) -> None:
    """Test initial fitted state."""

    assert prophet_model._is_fitted is False
    assert prophet_model.model is None


def test_get_params(
    prophet_model: ProphetModel,
) -> None:
    """Test model parameters."""

    params = prophet_model.get_params()

    assert params["date_column"] == "DATA"
    assert params["yearly_seasonality"] is False
    assert params["weekly_seasonality"] is False
    assert params["daily_seasonality"] is False


def test_fit_returns_model(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that fit returns the model instance."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    result = prophet_model.fit(
        sample_features,
        sample_target,
    )

    assert result is prophet_model


def test_fit_creates_fitted_estimator(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that fit creates a fitted estimator."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    assert prophet_model._is_fitted is True
    assert prophet_model.model is not None


def test_fit_passes_correct_columns_to_prophet(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Prophet receives ds and y columns."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    training_data = prophet_model.model.training_data

    assert list(training_data.columns) == [
        "ds",
        "y",
    ]

    assert training_data["y"].tolist() == [
        100,
        120,
        140,
        160,
        180,
    ]


def test_predict_returns_series(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test prediction output type."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    predictions = prophet_model.predict(
        sample_features,
    )

    assert isinstance(
        predictions,
        pd.Series,
    )


def test_predict_preserves_index(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test prediction index preservation."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    predictions = prophet_model.predict(
        sample_features,
    )

    pd.testing.assert_index_equal(
        predictions.index,
        sample_features.index,
    )


def test_predict_returns_expected_number_of_values(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test prediction length."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    predictions = prophet_model.predict(
        sample_features,
    )

    assert len(predictions) == len(
        sample_features
    )


def test_predictions_are_numeric(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test prediction values are numeric."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    predictions = prophet_model.predict(
        sample_features,
    )

    assert pd.api.types.is_numeric_dtype(
        predictions
    )


def test_predict_before_fit_raises_error(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
) -> None:
    """Test prediction before fitting."""

    with pytest.raises(DataValidationError):
        prophet_model.predict(
            sample_features,
        )


def test_missing_date_column_raises_error(
    prophet_model: ProphetModel,
    sample_target: pd.Series,
) -> None:
    """Test missing date column."""

    features = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
        }
    )

    target = pd.Series(
        [10, 20, 30],
    )

    with pytest.raises(DataValidationError):
        prophet_model.fit(
            features,
            target,
        )


def test_mismatched_training_lengths_raise_error(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
) -> None:
    """Test mismatched X and y lengths."""

    target = pd.Series(
        [100, 200],
    )

    with pytest.raises(DataValidationError):
        prophet_model.fit(
            sample_features,
            target,
        )


def test_empty_training_data_raises_error(
    prophet_model: ProphetModel,
) -> None:
    """Test empty training data."""

    features = pd.DataFrame(
        {
            "DATA": pd.Series(
                dtype="datetime64[ns]"
            ),
        }
    )

    target = pd.Series(
        dtype=float,
    )

    with pytest.raises(DataValidationError):
        prophet_model.fit(
            features,
            target,
        )


def test_missing_prediction_date_column_raises_error(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test missing prediction date column."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    invalid_features = sample_features.drop(
        columns=["DATA"],
    )

    with pytest.raises(DataValidationError):
        prophet_model.predict(
            invalid_features,
        )


def test_model_does_not_modify_training_features(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test training features are not modified."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    original = sample_features.copy(
        deep=True
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    pd.testing.assert_frame_equal(
        sample_features,
        original,
    )


def test_model_does_not_modify_prediction_features(
    prophet_model: ProphetModel,
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test prediction features are not modified."""

    monkeypatch.setattr(
        "prophet.Prophet",
        FakeProphet,
    )

    prophet_model.fit(
        sample_features,
        sample_target,
    )

    original = sample_features.copy(
        deep=True
    )

    prophet_model.predict(
        sample_features,
    )

    pd.testing.assert_frame_equal(
        sample_features,
        original,
    )