"""
Tests for MissingValueProcessor.
"""

import pandas as pd

from src.preprocessing.missing import MissingValueProcessor


def test_numeric_mean_strategy() -> None:
    dataframe = pd.DataFrame(
        {
            "VALOR": [10.0, None, 30.0],
        }
    )

    result = MissingValueProcessor()._process_numeric(
        dataframe,
        _config("mean"),
    )

    assert result["VALOR"].tolist() == [10.0, 20.0, 30.0]


def test_numeric_median_strategy() -> None:
    dataframe = pd.DataFrame(
        {
            "VALOR": [10.0, None, 30.0],
        }
    )

    result = MissingValueProcessor()._process_numeric(
        dataframe,
        _config("median"),
    )

    assert result["VALOR"].tolist() == [10.0, 20.0, 30.0]


def test_numeric_zero_strategy() -> None:
    dataframe = pd.DataFrame(
        {
            "VALOR": [10.0, None, 30.0],
        }
    )

    result = MissingValueProcessor()._process_numeric(
        dataframe,
        _config("zero"),
    )

    assert result["VALOR"].tolist() == [10.0, 0.0, 30.0]


def test_categorical_mode_strategy() -> None:
    dataframe = pd.DataFrame(
        {
            "categoria": ["A", None, "A"],
        }
    )

    result = MissingValueProcessor()._process_categorical(
        dataframe,
        _config_categorical("mode"),
    )

    assert result["categoria"].tolist() == [
        "A",
        "A",
        "A",
    ]


def test_datetime_drop_strategy() -> None:
    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                ["2026-01-01", None, "2026-01-03"]
            ),
        }
    )

    result = MissingValueProcessor()._process_datetime(
        dataframe,
        _config_datetime("drop"),
    )

    assert len(result) == 2


def _config(
    strategy: str,
):
    from src.config.pipeline import MissingSettings

    return MissingSettings(
        numeric_strategy=strategy,
        categorical_strategy="mode",
        datetime_strategy="keep",
    )


def _config_categorical(
    strategy: str,
):
    from src.config.pipeline import MissingSettings

    return MissingSettings(
        numeric_strategy="median",
        categorical_strategy=strategy,
        datetime_strategy="keep",
    )


def _config_datetime(
    strategy: str,
):
    from src.config.pipeline import MissingSettings

    return MissingSettings(
        numeric_strategy="median",
        categorical_strategy="mode",
        datetime_strategy=strategy,
    )