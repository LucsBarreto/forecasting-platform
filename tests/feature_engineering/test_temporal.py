"""
Tests for temporal feature engineering.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config.pipeline import TemporalFeatureSettings
from src.core.exceptions.validation import DataValidationError
from src.feature_engineering.temporal import (
    HolidayCalendar,
    TemporalFeatureEngineer,
)


def create_settings(
    *,
    enabled: bool = True,
    date_column: str = "DATA",
    create_business_day: bool = True,
    create_holiday: bool = True,
    create_days_to_month_end: bool = True,
    create_holiday_distance: bool = True,
    create_cyclical_features: bool = True,
) -> TemporalFeatureSettings:
    """Create temporal feature settings for tests."""

    return TemporalFeatureSettings(
        enabled=enabled,
        date_column=date_column,
        create_business_day=create_business_day,
        create_holiday=create_holiday,
        create_days_to_month_end=create_days_to_month_end,
        create_holiday_distance=create_holiday_distance,
        create_cyclical_features=create_cyclical_features,
    )


def create_dataframe() -> pd.DataFrame:
    """Create dataframe for temporal feature tests."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-31",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C001",
                "C001",
                "C001",
                "C001",
            ],
            "COD ITEM": [
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "VOLUME": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
        }
    )


def create_holiday_calendar() -> HolidayCalendar:
    """Create a deterministic holiday calendar for tests."""

    holiday_dates = {
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-15"),
        pd.Timestamp("2025-12-25"),
    }

    holiday_names = {
        pd.Timestamp("2025-01-01"): "New Year",
        pd.Timestamp("2025-01-15"): "Test Holiday",
        pd.Timestamp("2025-12-25"): "Christmas",
    }

    return HolidayCalendar(
        dates=holiday_dates,
        names=holiday_names,
    )


def test_validate_configuration_accepts_valid_configuration() -> None:
    """Valid temporal configuration must not raise an exception."""

    config = create_settings()

    # The current implementation validates the dataframe
    # directly inside process, so this test only ensures
    # the configuration can be instantiated successfully.
    assert config.enabled is True
    assert config.date_column == "DATA"


def test_missing_date_column_raises_error() -> None:
    """Missing date column must raise DataValidationError."""

    dataframe = create_dataframe().drop(
        columns=["DATA"],
    )

    engineer = TemporalFeatureEngineer()

    with pytest.raises(DataValidationError):
        engineer.process(dataframe)


def test_create_business_day() -> None:
    """Business day feature must identify weekdays excluding holidays."""

    dataframe = create_dataframe()

    holidays = {
        pd.Timestamp("2025-01-01"),
    }

    config = create_settings(
        create_business_day=True,
        create_holiday=False,
    )

    result = TemporalFeatureEngineer._create_calendar_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=holidays,
        config=config,
    )

    assert result["is_business_day"].tolist() == [
        False,
        True,
        True,
        False,
        True,
    ]


def test_create_holiday_features() -> None:
    """Holiday-related features must be correctly generated."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "VOLUME": [10.0, 20.0, 30.0],
        }
    )

    holidays = {
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-03"),
    }

    config = create_settings(
        create_business_day=False,
        create_holiday=True,
    )

    result = TemporalFeatureEngineer._create_calendar_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=holidays,
        config=config,
    )

    assert result["is_holiday"].tolist() == [
        True,
        False,
        True,
    ]

    assert result["is_holiday_eve"].tolist() == [
        False,
        True,
        False,
    ]

    assert result["is_post_holiday"].tolist() == [
        False,
        True,
        False,
    ]


def test_non_holiday_weekend_is_not_business_day() -> None:
    """Weekend dates must not be considered business days."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-04",
                    "2025-01-05",
                ]
            ),
        }
    )

    config = create_settings(
        create_business_day=True,
        create_holiday=False,
    )

    result = TemporalFeatureEngineer._create_calendar_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=set(),
        config=config,
    )

    assert result["is_business_day"].tolist() == [
        False,
        False,
    ]


def test_create_days_to_month_end() -> None:
    """Days to month end must be correctly calculated."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-15",
                    "2025-01-30",
                    "2025-01-31",
                ]
            )
        }
    )

    config = create_settings(
        create_days_to_month_end=True,
        create_holiday_distance=False,
        create_cyclical_features=False,
    )

    result = TemporalFeatureEngineer._create_distance_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=set(),
        config=config,
    )

    assert result["days_to_month_end"].tolist() == [
        30,
        16,
        1,
        0,
    ]


def test_create_days_from_month_start() -> None:
    """Days from month start must be correctly calculated."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-15",
                    "2025-01-31",
                ]
            )
        }
    )

    config = create_settings(
        create_days_to_month_end=True,
        create_holiday_distance=False,
        create_cyclical_features=False,
    )

    result = TemporalFeatureEngineer._create_distance_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=set(),
        config=config,
    )

    assert result["days_from_month_start"].tolist() == [
        0,
        1,
        14,
        30,
    ]


def test_days_until_holiday() -> None:
    """Days until next holiday must be correctly calculated."""

    holidays = [
        pd.Timestamp("2025-01-10"),
        pd.Timestamp("2025-01-20"),
        pd.Timestamp("2025-02-01"),
    ]

    assert (
        TemporalFeatureEngineer._days_until_holiday(
            pd.Timestamp("2025-01-01"),
            holidays,
        )
        == 9
    )

    assert (
        TemporalFeatureEngineer._days_until_holiday(
            pd.Timestamp("2025-01-10"),
            holidays,
        )
        == 0
    )

    assert (
        TemporalFeatureEngineer._days_until_holiday(
            pd.Timestamp("2025-01-11"),
            holidays,
        )
        == 9
    )


def test_days_since_holiday() -> None:
    """Days since previous holiday must be correctly calculated."""

    holidays = [
        pd.Timestamp("2025-01-10"),
        pd.Timestamp("2025-01-20"),
        pd.Timestamp("2025-02-01"),
    ]

    assert (
        TemporalFeatureEngineer._days_since_holiday(
            pd.Timestamp("2025-01-01"),
            holidays,
        )
        == -1
    )

    assert (
        TemporalFeatureEngineer._days_since_holiday(
            pd.Timestamp("2025-01-20"),
            holidays,
        )
        == 0
    )

    assert (
        TemporalFeatureEngineer._days_since_holiday(
            pd.Timestamp("2025-01-25"),
            holidays,
        )
        == 5
    )


def test_create_holiday_distance_features() -> None:
    """Holiday distance features must be correctly generated."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-15",
                ]
            )
        }
    )

    holidays = {
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-15"),
    }

    config = create_settings(
        create_days_to_month_end=False,
        create_holiday_distance=True,
        create_cyclical_features=False,
    )

    result = TemporalFeatureEngineer._create_distance_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=holidays,
        config=config,
    )

    assert result["days_until_holiday"].tolist() == [
        0,
        13,
        12,
        0,
    ]

    assert result["days_since_holiday"].tolist() == [
        0,
        1,
        2,
        0,
    ]


def test_cyclical_features_are_created() -> None:
    """Cyclical calendar features must be created."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-08",
                ]
            )
        }
    )

    config = create_settings(
        create_cyclical_features=True,
        create_days_to_month_end=False,
        create_holiday_distance=False,
    )

    result = TemporalFeatureEngineer._create_cyclical_features(
        data=dataframe.copy(),
        date_column="DATA",
        config=config,
    )

    expected_columns = [
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
        "week_sin",
        "week_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_cyclical_features_are_numeric() -> None:
    """Cyclical features must contain numeric values."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-15",
                    "2025-01-31",
                ]
            )
        }
    )

    config = create_settings(
        create_cyclical_features=True,
        create_days_to_month_end=False,
        create_holiday_distance=False,
    )

    result = TemporalFeatureEngineer._create_cyclical_features(
        data=dataframe.copy(),
        date_column="DATA",
        config=config,
    )

    cyclical_columns = [
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
        "week_sin",
        "week_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]

    for column in cyclical_columns:
        assert pd.api.types.is_numeric_dtype(
            result[column]
        )


def test_cyclical_features_are_bounded() -> None:
    """Sine and cosine features must be between -1 and 1."""

    dataframe = create_dataframe()

    config = create_settings(
        create_cyclical_features=True,
        create_days_to_month_end=False,
        create_holiday_distance=False,
    )

    result = TemporalFeatureEngineer._create_cyclical_features(
        data=dataframe.copy(),
        date_column="DATA",
        config=config,
    )

    cyclical_columns = [
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
        "week_sin",
        "week_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]

    for column in cyclical_columns:
        assert result[column].between(
            -1,
            1,
        ).all()


def test_disabled_cyclical_features_return_original_dataframe() -> None:
    """Disabled cyclical features must not modify the dataframe."""

    dataframe = create_dataframe()

    config = create_settings(
        create_cyclical_features=False,
    )

    result = TemporalFeatureEngineer._create_cyclical_features(
        data=dataframe.copy(),
        date_column="DATA",
        config=config,
    )

    pd.testing.assert_frame_equal(
        result,
        dataframe,
    )


def test_calendar_features_preserve_original_columns() -> None:
    """Calendar feature creation must preserve original columns."""

    dataframe = create_dataframe()

    original_columns = set(
        dataframe.columns
    )

    config = create_settings()

    result = TemporalFeatureEngineer._create_calendar_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=set(),
        config=config,
    )

    assert original_columns.issubset(
        set(result.columns)
    )


def test_distance_features_preserve_original_columns() -> None:
    """Distance feature creation must preserve original columns."""

    dataframe = create_dataframe()

    original_columns = set(
        dataframe.columns
    )

    config = create_settings()

    result = TemporalFeatureEngineer._create_distance_features(
        data=dataframe.copy(),
        date_column="DATA",
        holiday_dates=set(),
        config=config,
    )

    assert original_columns.issubset(
        set(result.columns)
    )


def test_cyclical_features_preserve_original_columns() -> None:
    """Cyclical feature creation must preserve original columns."""

    dataframe = create_dataframe()

    original_columns = set(
        dataframe.columns
    )

    config = create_settings()

    result = TemporalFeatureEngineer._create_cyclical_features(
        data=dataframe.copy(),
        date_column="DATA",
        config=config,
    )

    assert original_columns.issubset(
        set(result.columns)
    )


def test_load_holidays(tmp_path: Path) -> None:
    """Holiday calendar must be correctly loaded from JSON."""

    holiday_file = tmp_path / "holidays.json"

    holiday_file.write_text(
        """
        [
            {
                "date": "2025-01-01",
                "name": "New Year"
            },
            {
                "date": "2025-12-25",
                "name": "Christmas"
            }
        ]
        """,
        encoding="utf-8",
    )

    result = TemporalFeatureEngineer._load_holidays(
        holiday_file,
    )

    assert result.dates == {
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-12-25"),
    }

    assert result.names == {
        pd.Timestamp("2025-01-01"): "New Year",
        pd.Timestamp("2025-12-25"): "Christmas",
    }


def test_load_holidays_missing_file(
    tmp_path: Path,
) -> None:
    """Missing holiday file must raise FileNotFoundError."""

    holiday_file = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        TemporalFeatureEngineer._load_holidays(
            holiday_file,
        )