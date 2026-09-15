"""
Tests for CleaningProcessor.
"""

import pandas as pd

from src.preprocessing.cleaning import CleaningProcessor


def test_process_removes_empty_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "nome": ["A", None, "B"],
            "VALOR": [10, None, 20],
        }
    )

    result = CleaningProcessor().process(dataframe)

    assert len(result) == 2


def test_process_removes_duplicate_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "nome": ["A", "A", "B"],
            "VALOR": [10, 10, 20],
        }
    )

    result = CleaningProcessor().process(dataframe)

    assert len(result) == 2


def test_process_normalizes_column_names() -> None:
    dataframe = pd.DataFrame(
        {
            " nome   cliente ": ["A"],
            " valor  venda ": [100],
        }
    )

    result = CleaningProcessor().process(dataframe)

    assert list(result.columns) == [
        "nome cliente",
        "valor venda",
    ]


def test_process_normalizes_string_values() -> None:
    dataframe = pd.DataFrame(
        {
            "nome": [
                "  João   Silva  ",
                " Maria ",
            ],
        }
    )

    result = CleaningProcessor().process(dataframe)

    assert result["nome"].tolist() == [
        "João Silva",
        "Maria",
    ]


def test_process_does_not_modify_original_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "nome": ["  João  "],
        }
    )

    original = dataframe.copy()

    CleaningProcessor().process(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )