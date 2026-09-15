"""
Tests for DatetimeProcessor.
"""

import pandas as pd

from src.config.pipeline import DatetimeSettings
from src.preprocessing.datetime import DatetimeProcessor


def test_create_datetime_features() -> None:
    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                ["2026-01-31", "2026-02-01"]
            ),
        }
    )

    config = DatetimeSettings()

    result = DatetimeProcessor._create_features(
        data=dataframe,
        column="DATA",
        config=config,
    )

    assert result["DATA_year"].tolist() == [
        2026,
        2026,
    ]

    assert result["DATA_month"].tolist() == [
        1,
        2,
    ]

    assert result["DATA_quarter"].tolist() == [
        1,
        1,
    ]

    assert result["DATA_semester"].tolist() == [
        1,
        1,
    ]

    assert result["DATA_day"].tolist() == [
        31,
        1,
    ]


def test_datetime_weekend_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2026-01-31",
                    "2026-02-01",
                ]
            ),
        }
    )

    result = DatetimeProcessor._create_features(
        data=dataframe,
        column="DATA",
        config=DatetimeSettings(),
    )

    assert result["DATA_is_weekend"].tolist() == [
        True,
        True,
    ]


def test_datetime_month_start_and_end() -> None:
    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-31",
                ]
            ),
        }
    )

    result = DatetimeProcessor._create_features(
        data=dataframe,
        column="DATA",
        config=DatetimeSettings(),
    )

    assert result["DATA_is_month_start"].tolist() == [
        True,
        False,
    ]

    assert result["DATA_is_month_end"].tolist() == [
        False,
        True,
    ]


def test_process_ignores_non_datetime_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "DATA": ["2026-01-01"],
            "VALOR": [100],
        }
    )

    result = DatetimeProcessor().process(dataframe)

    assert list(result.columns) == [
        "DATA",
        "VALOR",
    ]