"""
Tests for TypeConverter.
"""

import pandas as pd
import pytest

from src.config import settings
from src.preprocessing.typing import TypeConverter


def test_convert_string() -> None:
    series = pd.Series([1, 2, 3])

    result = TypeConverter._convert_column(
        series,
        "string",
        "codigo",
    )

    assert str(result.dtype) == "string"


def test_convert_float() -> None:
    series = pd.Series(["10.5", "20.5"])

    result = TypeConverter._convert_column(
        series,
        "float64",
        "VALOR",
    )

    assert result.dtype == "float64"


def test_convert_integer() -> None:
    series = pd.Series(["10", "20"])

    result = TypeConverter._convert_column(
        series,
        "int64",
        "VOLUME",
    )

    assert result.dtype == "int64"


def test_convert_datetime() -> None:
    series = pd.Series(
        [
            "2026-01-01",
            "2026-01-02",
        ]
    )

    result = TypeConverter._convert_column(
        series,
        "datetime64[ns]",
        "DATA",
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result
    )


def test_invalid_conversion_raises_validation_error() -> None:
    series = pd.Series(["abc", "def"])

    with pytest.raises(Exception):
        TypeConverter._convert_column(
            series,
            "float64",
            "VALOR",
        )


def test_process_ignores_missing_configured_columns(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings.data.schema_config,
        "dtypes",
        {
            "inexistente": "string",
        },
    )

    dataframe = pd.DataFrame(
        {
            "nome": ["A"],
        }
    )

    result = TypeConverter().process(dataframe)

    pd.testing.assert_frame_equal(
        result,
        dataframe,
    )