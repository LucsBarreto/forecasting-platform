"""
Tests for TrainingPipeline.
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
def pipeline() -> TrainingPipeline:
    """Return a configured training pipeline."""

    return TrainingPipeline(
        output_manager=MagicMock(),
        preprocessing=MagicMock(),
        feature_engineering=MagicMock(),
        feature_selector=MagicMock(),
        temporal_splitter=MagicMock(),
        training_manager=MagicMock(),
    )


def test_pipeline_name(
    pipeline: TrainingPipeline,
) -> None:
    """Test pipeline name."""

    assert pipeline.name == "treinando pipeline"


def test_prepare_training_data(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test training data preparation."""

    selected_features = dataframe[
        [
            "feature_1",
            "feature_2",
        ]
    ].copy()

    pipeline.feature_selector.select.return_value = (
        selected_features
    )

    X, y = pipeline._prepare_training_data(
        dataframe,
        "target",
    )

    pipeline.feature_selector.select.assert_called_once()

    pd.testing.assert_frame_equal(
        X,
        selected_features,
    )

    pd.testing.assert_series_equal(
        y,
        dataframe["target"],
    )


def test_prepare_training_data_missing_target_raises_error(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test missing target validation."""

    with pytest.raises(
        ValueError,
        match="Target column",
    ):
        pipeline._prepare_training_data(
            dataframe,
            "missing",
        )


def test_prepare_training_data_invalid_target_raises_error(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test invalid target validation."""

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        pipeline._prepare_training_data(
            dataframe,
            "",
        )


def test_prepare_split_data(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test split data preparation."""

    feature_columns = pd.Index(
        [
            "feature_1",
            "feature_2",
        ]
    )

    X, y = pipeline._prepare_split_data(
        dataframe,
        "target",
        feature_columns,
    )

    pd.testing.assert_frame_equal(
        X,
        dataframe[
            [
                "feature_1",
                "feature_2",
            ]
        ],
    )

    pd.testing.assert_series_equal(
        y,
        dataframe["target"],
    )


def test_run_executes_training_flow(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test complete training orchestration."""

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

    training_result = MagicMock()

    training_result.models = {
        "random_forest": MagicMock(),
    }

    pipeline.training_manager.train.return_value = (
        training_result
    )

    result = pipeline.run(
        source,
        target_column="target",
    )

    pipeline.preprocessing.process.assert_called_once_with(
        dataframe,
    )

    pipeline.feature_engineering.process.assert_called_once_with(
        dataframe,
    )

    pipeline.temporal_splitter.split.assert_called_once_with(
        dataframe,
    )

    pipeline.training_manager.train.assert_called_once()

    assert isinstance(
        result,
        TrainingPipelineResult,
    )

    assert result.models == training_result.models

    pd.testing.assert_frame_equal(
        result.training_data,
        dataframe,
    )

    pd.testing.assert_frame_equal(
        result.validation_data,
        split.validation,
    )

    pd.testing.assert_frame_equal(
        result.test_data,
        split.test,
    )

def test_run_selects_features_from_temporal_train_slice_only(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Feature selection must run after the temporal split on train data only."""

    source = Path("data.csv")

    pipeline._load_data = MagicMock(
        return_value=dataframe,
    )
    pipeline.preprocessing.process.return_value = dataframe
    pipeline.feature_engineering.process.return_value = dataframe

    split = MagicMock()
    split.train = dataframe.iloc[:7].copy()
    split.validation = dataframe.iloc[7:8].copy()
    split.test = dataframe.iloc[8:].copy()

    pipeline.temporal_splitter.split.return_value = split

    selected_train = split.train[["feature_1", "feature_2"]].copy()
    pipeline.feature_selector.select.return_value = selected_train

    training_result = MagicMock()
    training_result.models = {"random_forest": MagicMock()}
    pipeline.training_manager.train.return_value = training_result

    pipeline.run(source, target_column="target")

    expected_train_features = split.train.drop(columns=["target"])
    feature_call = pipeline.feature_selector.select.call_args.args[0]
    pd.testing.assert_frame_equal(
        feature_call,
        expected_train_features,
    )


def test_run_returns_validation_and_test_data(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test validation and test datasets are returned."""

    pipeline._load_data = MagicMock(
        return_value=dataframe,
    )

    pipeline.preprocessing.process.return_value = (
        dataframe
    )

    pipeline.feature_engineering.process.return_value = (
        dataframe
    )

    pipeline.feature_selector.select.return_value = (
        dataframe[
            [
                "feature_1",
                "feature_2",
            ]
        ]
    )

    split = MagicMock()

    split.train = dataframe.iloc[:7].copy()
    split.validation = dataframe.iloc[7:9].copy()
    split.test = dataframe.iloc[9:].copy()

    pipeline.temporal_splitter.split.return_value = (
        split
    )

    training_result = MagicMock()
    training_result.models = {}

    pipeline.training_manager.train.return_value = (
        training_result
    )

    result = pipeline.run(
        Path("data.csv"),
        target_column="target",
    )

    assert isinstance(
        result,
        TrainingPipelineResult,
    )

    pd.testing.assert_frame_equal(
        result.validation_data,
        split.validation,
    )

    pd.testing.assert_frame_equal(
        result.test_data,
        split.test,
    )

def test_run_forwards_model_parameters(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test model parameters forwarding."""

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

    split = MagicMock()

    split.train = dataframe.iloc[:7].copy()
    split.validation = dataframe.iloc[7:8].copy()
    split.test = dataframe.iloc[8:].copy()

    pipeline.temporal_splitter.split.return_value = (
        split
    )

    selected_features = split.train[
        [
            "feature_1",
            "feature_2",
        ]
    ].copy()

    pipeline.feature_selector.select.return_value = (
        selected_features
    )

    training_result = MagicMock()
    training_result.models = {}

    pipeline.training_manager.train.return_value = (
        training_result
    )

    model_params = {
        "random_forest": {
            "n_estimators": 100,
        },
    }

    pipeline.run(
        source,
        target_column="target",
        model_params=model_params,
    )

    pipeline.training_manager.train.assert_called_once()

    call_args = (
        pipeline.training_manager.train.call_args
    )

    actual_X = call_args.args[0]
    actual_y = call_args.args[1]
    actual_params = call_args.kwargs["model_params"]

    expected_X = split.train[
        [
            "feature_1",
            "feature_2",
        ]
    ]

    expected_y = split.train[
        "target"
    ]

    pd.testing.assert_frame_equal(
        actual_X,
        expected_X,
    )

    pd.testing.assert_series_equal(
        actual_y,
        expected_y,
    )

    assert actual_params == model_params


def test_run_returns_preprocessed_dataframe_without_training_configuration(
    dataframe: pd.DataFrame,
) -> None:
    """Test backwards-compatible preprocessing-only behavior."""

    preprocessing = MagicMock()

    preprocessed = dataframe.copy()

    preprocessing.process.return_value = preprocessed

    pipeline = TrainingPipeline(
        output_manager=MagicMock(),
        preprocessing=preprocessing,
    )

    pipeline._load_data = MagicMock(
        return_value=dataframe,
    )

    result = pipeline.run(
        Path("data.csv"),
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    preprocessing.process.assert_called_once_with(
        dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        preprocessed,
    )


def test_run_propagates_errors(
    pipeline: TrainingPipeline,
) -> None:
    """Test pipeline error propagation."""

    pipeline._load_data = MagicMock(
        side_effect=ValueError(
            "loading error",
        ),
    )

    with pytest.raises(
        ValueError,
        match="loading error",
    ):
        pipeline.run(
            Path("data.csv"),
            target_column="target",
        )


def test_training_pipeline_result_contains_all_split_DATA(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test that the result exposes training, validation and test data."""

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
    split.validation = dataframe.iloc[7:9].copy()
    split.test = dataframe.iloc[9:].copy()

    pipeline.temporal_splitter.split.return_value = (
        split
    )

    training_result = MagicMock()
    training_result.models = {}

    pipeline.training_manager.train.return_value = (
        training_result
    )

    result = pipeline.run(
        Path("data.csv"),
        target_column="target",
    )

    assert isinstance(
        result,
        TrainingPipelineResult,
    )

    pd.testing.assert_frame_equal(
        result.training_data,
        dataframe,
    )

    pd.testing.assert_frame_equal(
        result.validation_data,
        split.validation,
    )

    pd.testing.assert_frame_equal(
        result.test_data,
        split.test,
    )


def test_run_requires_target_when_training_is_configured(
    pipeline: TrainingPipeline,
    dataframe: pd.DataFrame,
) -> None:
    """Test target requirement when training is configured."""

    pipeline._load_data = MagicMock(
        return_value=dataframe,
    )

    pipeline.preprocessing.process.return_value = (
        dataframe
    )

    with pytest.raises(
        ValueError,
        match="target_column must be provided",
    ):
        pipeline.run(
            Path("data.csv"),
        )

