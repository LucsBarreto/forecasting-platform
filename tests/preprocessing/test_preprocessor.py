"""
Tests for PreprocessingPipeline.
"""

from unittest.mock import MagicMock

import pandas as pd

from src.preprocessing.preprocessor import PreprocessingPipeline


def test_process_executes_processors_in_order() -> None:
    dataframe = pd.DataFrame(
        {
            "VALOR": [100],
        }
    )

    cleaning = MagicMock()
    missing = MagicMock()
    typing = MagicMock()
    datetime = MagicMock()

    cleaned = pd.DataFrame(
        {
            "VALOR": [90],
        }
    )

    missing_result = pd.DataFrame(
        {
            "VALOR": [80],
        }
    )

    typed = pd.DataFrame(
        {
            "VALOR": [80.0],
        }
    )

    final = pd.DataFrame(
        {
            "VALOR": [80.0],
        }
    )

    cleaning.process.return_value = cleaned
    missing.process.return_value = missing_result
    typing.process.return_value = typed
    datetime.process.return_value = final

    pipeline = PreprocessingPipeline(
        cleaning=cleaning,
        missing=missing,
        typing=typing,
        datetime=datetime,
    )

    result = pipeline.process(dataframe)

    # Verifica chamadas sem comparar DataFrames diretamente
    assert cleaning.process.call_count == 1
    assert missing.process.call_count == 1
    assert typing.process.call_count == 1
    assert datetime.process.call_count == 1

    cleaning_input = cleaning.process.call_args.args[0]
    missing_input = missing.process.call_args.args[0]
    typing_input = typing.process.call_args.args[0]
    datetime_input = datetime.process.call_args.args[0]

    pd.testing.assert_frame_equal(
        cleaning_input,
        dataframe,
    )

    pd.testing.assert_frame_equal(
        missing_input,
        cleaned,
    )

    pd.testing.assert_frame_equal(
        typing_input,
        missing_result,
    )

    pd.testing.assert_frame_equal(
        datetime_input,
        typed,
    )

    pd.testing.assert_frame_equal(
        result,
        final,
    )


def test_process_returns_final_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "VALOR": [100],
        }
    )

    cleaning = MagicMock()
    missing = MagicMock()
    typing = MagicMock()
    datetime = MagicMock()

    cleaning.process.return_value = dataframe
    missing.process.return_value = dataframe
    typing.process.return_value = dataframe
    datetime.process.return_value = dataframe

    pipeline = PreprocessingPipeline(
        cleaning=cleaning,
        missing=missing,
        typing=typing,
        datetime=datetime,
    )

    result = pipeline.process(dataframe)

    pd.testing.assert_frame_equal(
        result,
        dataframe,
    )