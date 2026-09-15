"""
Tests for TrainingPipeline model export.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipelines.train_pipeline import (
    TrainingPipeline,
    TrainingPipelineResult,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.date_range(
                "2026-01-01",
                periods=10,
                freq="D",
            ),
            "feature_1": range(10),
            "feature_2": range(10, 20),
            "target": range(100, 110),
        }
    )


@pytest.fixture
def model_exporter() -> MagicMock:
    """Return a mocked model exporter."""

    exporter = MagicMock()

    exporter.save.side_effect = (
        lambda model, filename: (
            Path("models") / filename
        )
    )

    return exporter


@pytest.fixture
def pipeline(
    model_exporter: MagicMock,
) -> TrainingPipeline:
    """Return a configured training pipeline."""

    return TrainingPipeline(
        output_manager=MagicMock(),
        preprocessing=MagicMock(),
        feature_engineering=MagicMock(),
        feature_selector=MagicMock(),
        temporal_splitter=MagicMock(),
        training_manager=MagicMock(),
        model_exporter=model_exporter,
    )


def test_export_models_exports_all_models(
    pipeline: TrainingPipeline,
    model_exporter: MagicMock,
) -> None:
    """Test that all trained models are exported."""

    model_a = MagicMock()
    model_b = MagicMock()

    training_result = MagicMock()

    training_result.models = {
        "linear_regression": model_a,
        "random_forest": model_b,
    }

    result = pipeline._export_models(
        training_result,
    )

    assert result == {
        "linear_regression": Path(
            "models/linear_regression.joblib"
        ),
        "random_forest": Path(
            "models/random_forest.joblib"
        ),
    }

    assert model_exporter.save.call_count == 2

    model_exporter.save.assert_any_call(
        model_a,
        "linear_regression.joblib",
    )

    model_exporter.save.assert_any_call(
        model_b,
        "random_forest.joblib",
    )


def test_export_models_returns_empty_when_exporter_is_not_configured(
    dataframe: pd.DataFrame,
) -> None:
    """Test optional exporter behavior."""

    pipeline = TrainingPipeline(
        output_manager=MagicMock(),
        preprocessing=MagicMock(),
        feature_engineering=MagicMock(),
        feature_selector=MagicMock(),
        temporal_splitter=MagicMock(),
        training_manager=MagicMock(),
    )

    training_result = MagicMock()

    training_result.models = {
        "baseline": MagicMock(),
    }

    result = pipeline._export_models(
        training_result,
    )

    assert result == {}


def test_run_exports_trained_models(
    pipeline: TrainingPipeline,
    model_exporter: MagicMock,
    dataframe: pd.DataFrame,
) -> None:
    """Test model export during pipeline execution."""

    source = Path("data.csv")

    pipeline._load_data = MagicMock(
        return_value=dataframe,
    )

    pipeline.preprocessing.process.return_value = (
        dataframe
    )

    pipeline.feature_engineering.process.return_value = (
        dataframe
    )

    selected_features = dataframe[
        [
            "feature_1",
            "feature_2",
        ]
    ].copy()

    pipeline.feature_selector.select.return_value = (
        selected_features
    )

    split = MagicMock()

    split.train = dataframe.iloc[:7].copy()
    split.validation = dataframe.iloc[7:8].copy()
    split.test = dataframe.iloc[8:].copy()

    pipeline.temporal_splitter.split.return_value = (
        split
    )

    model = MagicMock()

    training_result = MagicMock()

    training_result.models = {
        "random_forest": model,
    }

    pipeline.training_manager.train.return_value = (
        training_result
    )

    model_exporter.save.return_value = Path(
        "models/random_forest.joblib"
    )

    result = pipeline.run(
        source,
        target_column="target",
    )

    assert isinstance(
        result,
        TrainingPipelineResult,
    )

    assert result.models == {
        "random_forest": model,
    }

    assert result.exported_models == {
        "random_forest": Path(
            "models/random_forest.joblib"
        ),
    }

    model_exporter.save.assert_called_once_with(
        model,
        "random_forest.joblib",
    )


def test_run_does_not_export_when_training_is_not_configured(
    dataframe: pd.DataFrame,
    model_exporter: MagicMock,
) -> None:
    """Test that preprocessing-only execution does not export."""

    preprocessing = MagicMock()

    preprocessing.process.return_value = (
        dataframe
    )

    pipeline = TrainingPipeline(
        output_manager=MagicMock(),
        preprocessing=preprocessing,
        model_exporter=model_exporter,
    )

    pipeline._load_data = MagicMock(
        return_value=dataframe,
    )

    result = pipeline.run(
        Path("data.csv"),
    )

    pd.testing.assert_frame_equal(
        result,
        dataframe,
    )

    model_exporter.save.assert_not_called()


def test_export_models_propagates_exporter_error(
    pipeline: TrainingPipeline,
    model_exporter: MagicMock,
) -> None:
    """Test exporter errors are propagated."""

    model_exporter.save.side_effect = RuntimeError(
        "export error"
    )

    training_result = MagicMock()

    training_result.models = {
        "random_forest": MagicMock(),
    }

    with pytest.raises(
        RuntimeError,
        match="export error",
    ):
        pipeline._export_models(
            training_result,
        )