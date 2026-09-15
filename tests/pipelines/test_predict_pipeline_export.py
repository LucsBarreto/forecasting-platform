"""
Tests for PredictPipeline forecast export.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipelines.predict_pipeline import (
    PredictPipeline,
    PredictionPipelineResult,
)


@pytest.fixture
def features() -> pd.DataFrame:
    """Return sample prediction features."""

    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "feature_2": [10, 20, 30],
        }
    )


@pytest.fixture
def predictions() -> pd.DataFrame:
    """Return sample forecast predictions."""

    return pd.DataFrame(
        {
            "DATA": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
            "prediction": [
                100.0,
                110.0,
                120.0,
            ],
        }
    )


@pytest.fixture
def predictor() -> MagicMock:
    """Return a mocked predictor."""

    return MagicMock()


@pytest.fixture
def forecast_exporter() -> MagicMock:
    """Return a mocked forecast exporter."""

    exporter = MagicMock()

    exporter.export_csv.return_value = Path(
        "forecasts/forecast.csv"
    )

    exporter.export_parquet.return_value = Path(
        "forecasts/forecast.parquet"
    )

    return exporter


@pytest.fixture
def pipeline(
    predictor: MagicMock,
    forecast_exporter: MagicMock,
) -> PredictPipeline:
    """Return a configured prediction pipeline."""

    return PredictPipeline(
        predictor=predictor,
        forecast_exporter=forecast_exporter,
    )


def test_pipeline_name(
    pipeline: PredictPipeline,
) -> None:
    """Test pipeline name."""

    assert pipeline.name == "pipeline de predição"


def test_run_generates_predictions(
    pipeline: PredictPipeline,
    predictor: MagicMock,
    forecast_exporter: MagicMock,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Test prediction generation."""

    model = MagicMock()

    predictor.predict.return_value = (
        predictions
    )

    result = pipeline.run(
        features=features,
        model=model,
    )

    assert isinstance(
        result,
        PredictionPipelineResult,
    )

    pd.testing.assert_frame_equal(
        result.predictions,
        predictions,
    )

    predictor.predict.assert_called_once_with(
        model,
        features,
    )

    forecast_exporter.export_csv.assert_called_once_with(
        predictions,
    )

    forecast_exporter.export_parquet.assert_called_once_with(
        predictions,
    )


def test_run_exports_csv_and_parquet(
    pipeline: PredictPipeline,
    predictor: MagicMock,
    forecast_exporter: MagicMock,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Test both forecast export formats."""

    predictor.predict.return_value = (
        predictions
    )

    result = pipeline.run(
        features=features,
        model=MagicMock(),
    )

    assert result.exported_files == {
        "csv": Path(
            "forecasts/forecast.csv"
        ),
        "parquet": Path(
            "forecasts/forecast.parquet"
        ),
    }


def test_run_can_disable_export(
    pipeline: PredictPipeline,
    predictor: MagicMock,
    forecast_exporter: MagicMock,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Test prediction without exporting."""

    predictor.predict.return_value = (
        predictions
    )

    result = pipeline.run(
        features=features,
        model=MagicMock(),
        export=False,
    )

    assert isinstance(
        result,
        PredictionPipelineResult,
    )

    assert result.exported_files == {}

    forecast_exporter.export_csv.assert_not_called()
    forecast_exporter.export_parquet.assert_not_called()


def test_run_without_exporter(
    predictor: MagicMock,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Test prediction without configured exporter."""

    predictor.predict.return_value = (
        predictions
    )

    pipeline = PredictPipeline(
        predictor=predictor,
    )

    result = pipeline.run(
        features=features,
        model=MagicMock(),
    )

    assert isinstance(
        result,
        PredictionPipelineResult,
    )

    pd.testing.assert_frame_equal(
        result.predictions,
        predictions,
    )

    assert result.exported_files == {}


def test_run_requires_features(
    pipeline: PredictPipeline,
) -> None:
    """Test missing prediction features."""

    with pytest.raises(
        ValueError,
        match="features cannot be None",
    ):
        pipeline.run(
            model=MagicMock(),
        )


def test_run_requires_model(
    pipeline: PredictPipeline,
    features: pd.DataFrame,
) -> None:
    """Test missing model."""

    with pytest.raises(
        ValueError,
        match="model must be provided",
    ):
        pipeline.run(
            features=features,
        )


def test_run_rejects_invalid_features(
    pipeline: PredictPipeline,
) -> None:
    """Test invalid feature dataframe."""

    with pytest.raises(
        TypeError,
        match="features must be a pandas DataFrame",
    ):
        pipeline.run(
            features=[1, 2, 3],
            model=MagicMock(),
        )


def test_run_rejects_empty_features(
    pipeline: PredictPipeline,
) -> None:
    """Test empty feature dataframe."""

    with pytest.raises(
        ValueError,
        match="features cannot be empty",
    ):
        pipeline.run(
            features=pd.DataFrame(),
            model=MagicMock(),
        )


def test_run_rejects_invalid_predictions(
    pipeline: PredictPipeline,
    predictor: MagicMock,
    features: pd.DataFrame,
) -> None:
    """Test invalid predictor output."""

    predictor.predict.return_value = [
        100,
        200,
        300,
    ]

    with pytest.raises(
        TypeError,
        match="predictions must be a pandas DataFrame",
    ):
        pipeline.run(
            features=features,
            model=MagicMock(),
        )


def test_run_propagates_predictor_error(
    pipeline: PredictPipeline,
    predictor: MagicMock,
    features: pd.DataFrame,
) -> None:
    """Test predictor errors are propagated."""

    predictor.predict.side_effect = RuntimeError(
        "prediction error"
    )

    with pytest.raises(
        RuntimeError,
        match="prediction error",
    ):
        pipeline.run(
            features=features,
            model=MagicMock(),
        )


def test_run_propagates_export_error(
    pipeline: PredictPipeline,
    predictor: MagicMock,
    forecast_exporter: MagicMock,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Test forecast exporter errors are propagated."""

    predictor.predict.return_value = (
        predictions
    )

    forecast_exporter.export_csv.side_effect = (
        RuntimeError("export error")
    )

    with pytest.raises(
        RuntimeError,
        match="export error",
    ):
        pipeline.run(
            features=features,
            model=MagicMock(),
        )


def test_legacy_mode_returns_dataframe(
    features: pd.DataFrame,
) -> None:
    """Test backwards-compatible prediction-only behavior."""

    pipeline = PredictPipeline()

    result = pipeline.run(
        features=features,
    )

    pd.testing.assert_frame_equal(
        result,
        features,
    )