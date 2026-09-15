"""
Tests for feature engineering pipeline.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.feature_engineering.business import BusinessFeatureEngineer
from src.feature_engineering.lag import LagFeatureEngineer
from src.feature_engineering.rolling import RollingFeatureEngineer
from src.feature_engineering.temporal import TemporalFeatureEngineer
from src.feature_engineering.trend import TrendFeatureEngineer
from src.pipelines.feature_engineering_pipeline import (
    FeatureEngineeringPipeline,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C001",
                "C001",
            ],
            "COD ITEM": [
                "P001",
                "P001",
                "P001",
            ],
            "VOLUME": [
                10.0,
                20.0,
                30.0,
            ],
            "VALOR": [
                100.0,
                200.0,
                300.0,
            ],
        }
    )


@pytest.fixture
def temporal() -> MagicMock:
    """Return a mocked temporal feature engineer."""

    return MagicMock(
        spec=TemporalFeatureEngineer
    )


@pytest.fixture
def lag() -> MagicMock:
    """Return a mocked lag feature engineer."""

    return MagicMock(
        spec=LagFeatureEngineer
    )


@pytest.fixture
def rolling() -> MagicMock:
    """Return a mocked rolling feature engineer."""

    return MagicMock(
        spec=RollingFeatureEngineer
    )


@pytest.fixture
def trend() -> MagicMock:
    """Return a mocked trend feature engineer."""

    return MagicMock(
        spec=TrendFeatureEngineer
    )


@pytest.fixture
def business() -> MagicMock:
    """Return a mocked business feature engineer."""

    return MagicMock(
        spec=BusinessFeatureEngineer
    )


@pytest.fixture
def pipeline(
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> FeatureEngineeringPipeline:
    """Return a feature engineering pipeline."""

    return FeatureEngineeringPipeline(
        temporal=temporal,
        lag=lag,
        rolling=rolling,
        trend=trend,
        business=business,
    )


def test_process_executes_engineers_in_order(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Feature engineers must execute in the configured order."""

    temporal_result = MagicMock(
        name="temporal_result"
    )
    lag_result = MagicMock(
        name="lag_result"
    )
    rolling_result = MagicMock(
        name="rolling_result"
    )
    trend_result = MagicMock(
        name="trend_result"
    )
    business_result = MagicMock(
        name="business_result"
    )

    temporal.process.return_value = temporal_result
    lag.process.return_value = lag_result
    rolling.process.return_value = rolling_result
    trend.process.return_value = trend_result
    business.process.return_value = business_result

    result = pipeline.process(dataframe)

    assert result is business_result

    temporal_input = temporal.process.call_args.args[0]
    lag_input = lag.process.call_args.args[0]
    rolling_input = rolling.process.call_args.args[0]
    trend_input = trend.process.call_args.args[0]
    business_input = business.process.call_args.args[0]

    assert temporal_input is not dataframe
    assert lag_input is temporal_result
    assert rolling_input is lag_result
    assert trend_input is rolling_result
    assert business_input is trend_result

    temporal.process.assert_called_once()
    lag.process.assert_called_once()
    rolling.process.assert_called_once()
    trend.process.assert_called_once()
    business.process.assert_called_once()


def test_process_returns_final_dataframe(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Pipeline must return the output from the final stage."""

    temporal_result = dataframe.copy()
    lag_result = dataframe.copy()
    rolling_result = dataframe.copy()
    trend_result = dataframe.copy()

    expected = dataframe.copy()

    temporal.process.return_value = temporal_result
    lag.process.return_value = lag_result
    rolling.process.return_value = rolling_result
    trend.process.return_value = trend_result
    business.process.return_value = expected

    result = pipeline.process(dataframe)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_process_does_not_modify_input_dataframe(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Pipeline must not modify the input dataframe."""

    original = dataframe.copy(deep=True)

    temporal.process.side_effect = lambda data: data
    lag.process.side_effect = lambda data: data
    rolling.process.side_effect = lambda data: data
    trend.process.side_effect = lambda data: data
    business.process.side_effect = lambda data: data

    pipeline.process(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )

    temporal_input = temporal.process.call_args.args[0]

    assert temporal_input is not dataframe


def test_process_passes_output_between_engineers(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Each engineer must receive the previous engineer output."""

    stage_1 = MagicMock(name="temporal_dataframe")
    stage_2 = MagicMock(name="lag_dataframe")
    stage_3 = MagicMock(name="rolling_dataframe")
    stage_4 = MagicMock(name="trend_dataframe")
    stage_5 = MagicMock(name="business_dataframe")

    temporal.process.return_value = stage_1
    lag.process.return_value = stage_2
    rolling.process.return_value = stage_3
    trend.process.return_value = stage_4
    business.process.return_value = stage_5

    result = pipeline.process(dataframe)

    assert result is stage_5

    assert (
        temporal.process.call_args.args[0]
        is not dataframe
    )

    assert (
        lag.process.call_args.args[0]
        is stage_1
    )

    assert (
        rolling.process.call_args.args[0]
        is stage_2
    )

    assert (
        trend.process.call_args.args[0]
        is stage_3
    )

    assert (
        business.process.call_args.args[0]
        is stage_4
    )


def test_process_stops_when_temporal_fails(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Pipeline must stop when temporal processing fails."""

    temporal.process.side_effect = ValueError(
        "temporal processing failed"
    )

    with pytest.raises(
        ValueError,
        match="temporal processing failed",
    ):
        pipeline.process(dataframe)

    lag.process.assert_not_called()
    rolling.process.assert_not_called()
    trend.process.assert_not_called()
    business.process.assert_not_called()


def test_process_stops_when_lag_fails(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Pipeline must stop when lag processing fails."""

    temporal_result = dataframe.copy()

    temporal.process.return_value = temporal_result

    lag.process.side_effect = ValueError(
        "lag processing failed"
    )

    with pytest.raises(
        ValueError,
        match="lag processing failed",
    ):
        pipeline.process(dataframe)

    rolling.process.assert_not_called()
    trend.process.assert_not_called()
    business.process.assert_not_called()


def test_process_stops_when_rolling_fails(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Pipeline must stop when rolling processing fails."""

    temporal.process.return_value = dataframe.copy()
    lag.process.return_value = dataframe.copy()

    rolling.process.side_effect = ValueError(
        "rolling processing failed"
    )

    with pytest.raises(
        ValueError,
        match="rolling processing failed",
    ):
        pipeline.process(dataframe)

    trend.process.assert_not_called()
    business.process.assert_not_called()


def test_process_stops_when_trend_fails(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Pipeline must stop when trend processing fails."""

    temporal.process.return_value = dataframe.copy()
    lag.process.return_value = dataframe.copy()
    rolling.process.return_value = dataframe.copy()

    trend.process.side_effect = ValueError(
        "trend processing failed"
    )

    with pytest.raises(
        ValueError,
        match="trend processing failed",
    ):
        pipeline.process(dataframe)

    business.process.assert_not_called()


def test_process_propagates_business_exception(
    pipeline: FeatureEngineeringPipeline,
    dataframe: pd.DataFrame,
    temporal: MagicMock,
    lag: MagicMock,
    rolling: MagicMock,
    trend: MagicMock,
    business: MagicMock,
) -> None:
    """Business exceptions must propagate."""

    temporal.process.return_value = dataframe.copy()
    lag.process.return_value = dataframe.copy()
    rolling.process.return_value = dataframe.copy()
    trend.process.return_value = dataframe.copy()

    business.process.side_effect = ValueError(
        "business processing failed"
    )

    with pytest.raises(
        ValueError,
        match="business processing failed",
    ):
        pipeline.process(dataframe)