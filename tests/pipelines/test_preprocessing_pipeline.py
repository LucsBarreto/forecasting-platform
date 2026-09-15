"""
Tests for preprocessing pipeline.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.aggregation.aggregator import Aggregator
from src.preprocessing.cleaning import CleaningProcessor
from src.preprocessing.missing import MissingValueProcessor
from src.preprocessing.typing import TypeConverter
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C002",
            ],
            "COD ITEM": [
                "P001",
                "P002",
            ],
            "VOLUME": [
                10.0,
                20.0,
            ],
        }
    )


@pytest.fixture
def cleaning() -> MagicMock:
    """Return a mocked cleaning processor."""

    return MagicMock(spec=CleaningProcessor)


@pytest.fixture
def missing() -> MagicMock:
    """Return a mocked missing value processor."""

    return MagicMock(spec=MissingValueProcessor)


@pytest.fixture
def typing() -> MagicMock:
    """Return a mocked type converter."""

    return MagicMock(spec=TypeConverter)


@pytest.fixture
def aggregation() -> MagicMock:
    """Return a mocked aggregator."""

    return MagicMock(spec=Aggregator)


@pytest.fixture
def pipeline(
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> PreprocessingPipeline:
    """Return a preprocessing pipeline."""

    return PreprocessingPipeline(
        cleaning=cleaning,
        missing=missing,
        typing=typing,
        aggregation=aggregation,
    )


def test_process_executes_processors_in_order(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Processors must execute in the configured order."""

    cleaned = dataframe.copy()
    missing_processed = dataframe.copy()
    typed = dataframe.copy()
    aggregated = dataframe.copy()

    cleaning.process.return_value = cleaned
    missing.process.return_value = missing_processed
    typing.process.return_value = typed
    aggregation.aggregate.return_value = aggregated

    result = pipeline.process(dataframe)

    assert result is aggregated

    cleaning_input = cleaning.process.call_args.args[0]
    missing_input = missing.process.call_args.args[0]
    typing_input = typing.process.call_args.args[0]
    aggregation_input = aggregation.aggregate.call_args.args[0]

    assert cleaning_input is not dataframe
    assert missing_input is cleaned
    assert typing_input is missing_processed
    assert aggregation_input is typed

    cleaning.process.assert_called_once()
    missing.process.assert_called_once()
    typing.process.assert_called_once()
    aggregation.aggregate.assert_called_once()


def test_process_returns_final_dataframe(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Pipeline must return the output from the final stage."""

    expected = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                ["2025-01-01"]
            ),
            "VOLUME": [10.0],
        }
    )

    cleaning.process.return_value = dataframe.copy()
    missing.process.return_value = dataframe.copy()
    typing.process.return_value = dataframe.copy()
    aggregation.aggregate.return_value = expected

    result = pipeline.process(dataframe)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_process_does_not_modify_input_dataframe(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Pipeline must not modify the input dataframe."""

    original = dataframe.copy(deep=True)

    cleaning.process.side_effect = lambda data: data
    missing.process.side_effect = lambda data: data
    typing.process.side_effect = lambda data: data
    aggregation.aggregate.side_effect = lambda data: data

    pipeline.process(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )

    cleaning_input = cleaning.process.call_args.args[0]

    assert cleaning_input is not dataframe


def test_process_passes_output_between_stages(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Each stage must receive the previous stage output."""

    stage_1 = MagicMock(name="cleaned_dataframe")
    stage_2 = MagicMock(name="missing_dataframe")
    stage_3 = MagicMock(name="typed_dataframe")
    stage_4 = MagicMock(name="aggregated_dataframe")

    cleaning.process.return_value = stage_1
    missing.process.return_value = stage_2
    typing.process.return_value = stage_3
    aggregation.aggregate.return_value = stage_4

    result = pipeline.process(dataframe)

    assert result is stage_4

    cleaning_input = cleaning.process.call_args.args[0]
    missing_input = missing.process.call_args.args[0]
    typing_input = typing.process.call_args.args[0]
    aggregation_input = aggregation.aggregate.call_args.args[0]

    assert cleaning_input is not dataframe
    assert missing_input is stage_1
    assert typing_input is stage_2
    assert aggregation_input is stage_3


def test_process_propagates_cleaning_exception(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Cleaning exceptions must propagate."""

    error = ValueError("cleaning failed")

    cleaning.process.side_effect = error

    with pytest.raises(
        ValueError,
        match="cleaning failed",
    ):
        pipeline.process(dataframe)

    missing.process.assert_not_called()
    typing.process.assert_not_called()
    aggregation.aggregate.assert_not_called()


def test_process_stops_when_missing_stage_fails(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Pipeline must stop when missing-value processing fails."""

    cleaned = dataframe.copy()

    cleaning.process.return_value = cleaned

    missing.process.side_effect = ValueError(
        "missing processing failed"
    )

    with pytest.raises(
        ValueError,
        match="missing processing failed",
    ):
        pipeline.process(dataframe)

    typing.process.assert_not_called()
    aggregation.aggregate.assert_not_called()


def test_process_stops_when_type_conversion_fails(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Pipeline must stop when type conversion fails."""

    cleaned = dataframe.copy()
    missing_processed = dataframe.copy()

    cleaning.process.return_value = cleaned
    missing.process.return_value = missing_processed

    typing.process.side_effect = ValueError(
        "type conversion failed"
    )

    with pytest.raises(
        ValueError,
        match="type conversion failed",
    ):
        pipeline.process(dataframe)

    aggregation.aggregate.assert_not_called()


def test_process_propagates_aggregation_exception(
    pipeline: PreprocessingPipeline,
    dataframe: pd.DataFrame,
    cleaning: MagicMock,
    missing: MagicMock,
    typing: MagicMock,
    aggregation: MagicMock,
) -> None:
    """Aggregation exceptions must propagate."""

    cleaned = dataframe.copy()
    missing_processed = dataframe.copy()
    typed = dataframe.copy()

    cleaning.process.return_value = cleaned
    missing.process.return_value = missing_processed
    typing.process.return_value = typed

    aggregation.aggregate.side_effect = ValueError(
        "aggregation failed"
    )

    with pytest.raises(
        ValueError,
        match="aggregation failed",
    ):
        pipeline.process(dataframe)