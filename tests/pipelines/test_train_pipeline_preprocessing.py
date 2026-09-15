"""
Tests for TrainingPipeline preprocessing integration.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipelines.train_pipeline import TrainingPipeline


@pytest.fixture
def output_manager() -> MagicMock:
    """Return a mocked output manager."""

    return MagicMock()


@pytest.fixture
def preprocessing() -> MagicMock:
    """Return a mocked preprocessing pipeline."""

    return MagicMock()


@pytest.fixture
def pipeline(
    output_manager: MagicMock,
    preprocessing: MagicMock,
) -> TrainingPipeline:
    """Return a training pipeline with mocked dependencies."""

    pipeline = TrainingPipeline(
        output_manager=output_manager,
        preprocessing=preprocessing,
    )

    return pipeline


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "VOLUME": [10, 20],
            "VALOR": [100.0, 200.0],
        }
    )


def test_run_applies_preprocessing(
    pipeline: TrainingPipeline,
    preprocessing: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that training applies preprocessing."""

    source = Path("data.csv")

    pipeline._load_data = MagicMock(
        return_value=sample_dataframe,
    )

    processed_dataframe = sample_dataframe.copy()

    preprocessing.process.return_value = processed_dataframe

    result = pipeline.run(source)

    preprocessing.process.assert_called_once_with(
        sample_dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        processed_dataframe,
    )


def test_run_returns_preprocessed_dataframe(
    pipeline: TrainingPipeline,
    preprocessing: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that run returns the preprocessing result."""

    source = Path("data.csv")

    pipeline._load_data = MagicMock(
        return_value=sample_dataframe,
    )

    processed_dataframe = pd.DataFrame(
        {
            "DATA": sample_dataframe["DATA"],
            "VOLUME": [100, 200],
            "VALOR": [1000.0, 2000.0],
        }
    )

    preprocessing.process.return_value = processed_dataframe

    result = pipeline.run(source)

    pd.testing.assert_frame_equal(
        result,
        processed_dataframe,
    )


def test_run_passes_loaded_dataframe_to_preprocessing(
    pipeline: TrainingPipeline,
    preprocessing: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the loaded dataframe is passed to preprocessing."""

    source = Path("data.csv")

    pipeline._load_data = MagicMock(
        return_value=sample_dataframe,
    )

    preprocessing.process.return_value = sample_dataframe

    pipeline.run(source)

    preprocessing.process.assert_called_once()

    argument = preprocessing.process.call_args.args[0]

    pd.testing.assert_frame_equal(
        argument,
        sample_dataframe,
    )


def test_run_does_not_call_preprocessing_when_loading_fails(
    pipeline: TrainingPipeline,
    preprocessing: MagicMock,
) -> None:
    """Test that preprocessing is not called when loading fails."""

    source = Path("data.csv")

    pipeline._load_data = MagicMock(
        side_effect=ValueError("loading error"),
    )

    with pytest.raises(ValueError, match="loading error"):
        pipeline.run(source)

    preprocessing.process.assert_not_called()


def test_run_propagates_preprocessing_exception(
    pipeline: TrainingPipeline,
    preprocessing: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that preprocessing exceptions are propagated."""

    source = Path("data.csv")

    pipeline._load_data = MagicMock(
        return_value=sample_dataframe,
    )

    preprocessing.process.side_effect = ValueError(
        "preprocessing error",
    )

    with pytest.raises(
        ValueError,
        match="preprocessing error",
    ):
        pipeline.run(source)


def test_run_preserves_loaded_dataframe_until_preprocessing(
    pipeline: TrainingPipeline,
    preprocessing: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the loaded dataframe reaches preprocessing unchanged."""

    source = Path("data.csv")

    original = sample_dataframe.copy(deep=True)

    pipeline._load_data = MagicMock(
        return_value=sample_dataframe,
    )

    preprocessing.process.return_value = sample_dataframe

    pipeline.run(source)

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )