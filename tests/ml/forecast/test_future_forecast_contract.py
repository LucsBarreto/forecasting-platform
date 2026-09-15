import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.forecast.future_forecast import (
    FutureFeatureAvailabilityContract,
    FutureFeatureAvailabilityResult,
    FutureFeatureFrameBuilder,
    FutureForecastContract,
    FutureForecastResult,
)
from src.ml.models.base_model import BaseModel


class DummyModel(BaseModel):
    @property
    def name(self) -> str:
        return "dummy"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DummyModel":
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series([1.0, 2.0], index=X.index)


def test_future_forecast_builder_and_contract_are_isolated() -> None:
    """Should build a future feature frame from a known history frame and predict only from it."""
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
            "target": [10.0, 11.0, 12.0],
        }
    )

    builder = FutureFeatureFrameBuilder(time_column="date")
    future_frame = builder.build(
        history_frame=history,
        horizon=2,
        feature_columns=["feature_a", "feature_b"],
    )

    assert list(future_frame.columns) == ["date", "feature_a", "feature_b"]
    assert len(future_frame) == 2

    contract = FutureForecastContract(model=DummyModel())
    result = contract.run(future_frame=future_frame)

    assert isinstance(result, FutureForecastResult)
    assert len(result.predictions) == len(future_frame)
    assert list(result.predictions.index) == list(future_frame.index)


def test_future_feature_frame_builder_respects_future_horizon_and_excludes_target_leakage() -> None:
    """Future feature construction should expose only permitted columns and a future horizon window."""
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "target": [10.0, 11.0, 12.0],
            "lag_1": [9.0, 10.0, 11.0],
            "lag_2": [8.0, 9.0, 10.0],
            "rolling_2": [9.5, 10.5, 11.5],
        }
    )

    builder = FutureFeatureFrameBuilder(time_column="date")
    future_frame = builder.build(
        history_frame=history,
        horizon=3,
        feature_columns=["lag_1", "lag_2", "rolling_2"],
    )

    assert len(future_frame) == 3
    assert "target" not in future_frame.columns
    assert list(future_frame.columns) == ["date", "lag_1", "lag_2", "rolling_2"]


def test_future_feature_frame_builder_builds_recursive_lag_and_rolling_without_leakage() -> None:
    """The future feature builder must consume historical target values for step 1 and then consume previously produced predictions for steps > 1."""
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "target": [10.0, 11.0, 12.0, 13.0],
        }
    )

    builder = FutureFeatureFrameBuilder(time_column="date")
    future_frame = builder.build_recursive(
        history_frame=history,
        horizon=3,
        target_column="target",
        predictions=[14.0, 15.0, 16.0],
        lag_values=[1, 2],
        rolling_windows=[2],
    )

    assert len(future_frame) == 3
    assert "target" not in future_frame.columns
    assert "lag_1" in future_frame.columns
    assert "lag_2" in future_frame.columns
    assert "rolling_2" in future_frame.columns
    assert future_frame.loc[0, "lag_1"] == 13.0
    assert future_frame.loc[1, "lag_1"] == 14.0
    assert future_frame.loc[2, "lag_1"] == 15.0


def test_future_feature_availability_contract_rejects_uncomputable_selected_features() -> None:
    """The future feature availability contract should explicitly reject features that the future frame cannot produce."""
    contract = FutureFeatureAvailabilityContract()

    with pytest.raises(DataValidationError, match="business_feature"):
        contract.run(
            selected_features=[
                "lag_1",
                "lag_2",
                "rolling_7",
                "dia_semana",
                "mes",
                "business_feature",
            ],
            available_features=[
                "lag_1",
                "lag_2",
                "rolling_7",
                "dia_semana",
                "mes",
            ],
        )


def test_future_feature_availability_contract_accepts_all_future_features() -> None:
    """The future feature availability contract should serialize an availability result when all selected features are reproducible."""
    contract = FutureFeatureAvailabilityContract()

    result = contract.run(
        selected_features=[
            "lag_1",
            "lag_2",
            "rolling_7",
            "dia_semana",
            "mes",
        ],
        available_features=[
            "lag_1",
            "lag_2",
            "rolling_7",
            "dia_semana",
            "mes",
        ],
    )

    assert isinstance(result, FutureFeatureAvailabilityResult)
    assert result.available is True
    assert result.missing_features == []
