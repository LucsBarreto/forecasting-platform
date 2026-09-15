"""
Tests for PredictPipeline.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipelines.predict_pipeline import (
    PredictPipeline,
    PredictionPipelineResult,
)


@pytest.fixture
def forecaster() -> MagicMock:
    """Return a mocked forecaster."""

    return MagicMock()


@pytest.fixture
def pipeline(
    forecaster: MagicMock,
) -> PredictPipeline:
    """Return a configured prediction pipeline."""

    return PredictPipeline(
        forecaster=forecaster,
    )


@pytest.fixture
def features() -> pd.DataFrame:
    """Return sample prediction features."""

    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def models() -> dict[str, MagicMock]:
    """Return sample trained models."""

    return {
        "linear_regression": MagicMock(),
        "random_forest": MagicMock(),
    }


def test_pipeline_name(
    pipeline: PredictPipeline,
) -> None:
    """Test pipeline name."""

    assert pipeline.name == "pipeline de predição"


def test_run_predicts_all_models(
    pipeline: PredictPipeline,
    forecaster: MagicMock,
    models: dict[str, MagicMock],
    features: pd.DataFrame,
) -> None:
    """Test that all models generate predictions."""

    linear_predictions = pd.Series(
        [100.0, 110.0, 120.0, 130.0],
    )

    random_forest_predictions = pd.Series(
        [101.0, 111.0, 121.0, 131.0],
    )

    forecaster.predict.side_effect = [
        linear_predictions,
        random_forest_predictions,
    ]

    result = pipeline.run(
        models,
        features,
    )

    assert isinstance(
        result,
        PredictionPipelineResult,
    )

    pd.testing.assert_series_equal(
        result.predictions["linear_regression"],
        linear_predictions,
    )

    pd.testing.assert_series_equal(
        result.predictions["random_forest"],
        random_forest_predictions,
    )

    assert forecaster.predict.call_count == 2

    first_call = forecaster.predict.call_args_list[0]
    second_call = forecaster.predict.call_args_list[1]

    assert first_call.args[0] is models["linear_regression"]
    assert first_call.args[1] is features

    assert second_call.args[0] is models["random_forest"]
    assert second_call.args[1] is features


def test_run_preserves_forecaster_predictions(
    pipeline: PredictPipeline,
    forecaster: MagicMock,
    features: pd.DataFrame,
) -> None:
    """Test that forecaster predictions are preserved."""

    model = MagicMock()

    expected_predictions = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
        name="prediction",
    )

    forecaster.predict.return_value = (
        expected_predictions
    )

    result = pipeline.run(
        {"baseline": model},
        features,
    )

    pd.testing.assert_series_equal(
        result.predictions["baseline"],
        expected_predictions,
    )


def test_run_does_not_modify_features(
    pipeline: PredictPipeline,
    forecaster: MagicMock,
    features: pd.DataFrame,
) -> None:
    """Test that prediction features are not modified."""

    original = features.copy()

    forecaster.predict.return_value = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
    )

    pipeline.run(
        {"baseline": MagicMock()},
        features,
    )

    pd.testing.assert_frame_equal(
        features,
        original,
    )


def test_run_empty_models_raises_error(
    pipeline: PredictPipeline,
    features: pd.DataFrame,
) -> None:
    """Test empty model validation."""

    with pytest.raises(
        ValueError,
        match="models cannot be empty",
    ):
        pipeline.run(
            {},
            features,
        )


def test_run_invalid_models_raises_error(
    pipeline: PredictPipeline,
    features: pd.DataFrame,
) -> None:
    """Test invalid models validation."""

    with pytest.raises(
        TypeError,
        match="models must be a dictionary",
    ):
        pipeline.run(
            [],
            features,
        )


def test_run_empty_features_raises_error(
    pipeline: PredictPipeline,
    models: dict[str, MagicMock],
) -> None:
    """Test empty features validation."""

    with pytest.raises(
        ValueError,
        match="features cannot be empty",
    ):
        pipeline.run(
            models,
            pd.DataFrame(),
        )


def test_run_invalid_features_raises_error(
    pipeline: PredictPipeline,
    models: dict[str, MagicMock],
) -> None:
    """Test invalid features validation."""

    with pytest.raises(
        TypeError,
        match="features must be a pandas DataFrame",
    ):
        pipeline.run(
            models,
            [[1, 2], [3, 4]],
        )


def test_run_propagates_forecaster_error(
    pipeline: PredictPipeline,
    forecaster: MagicMock,
    features: pd.DataFrame,
) -> None:
    """Test forecaster errors are propagated."""

    forecaster.predict.side_effect = RuntimeError(
        "prediction error",
    )

    with pytest.raises(
        RuntimeError,
        match="prediction error",
    ):
        pipeline.run(
            {"baseline": MagicMock()},
            features,
        )