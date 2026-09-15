"""
Tests for SchemaValidator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import (
    InvalidColumnTypeError,
    MissingColumnError,
)
from src.schemas.models import ColumnSchema, DataSchema
from src.schemas.validator import SchemaValidator


@pytest.fixture
def schema() -> DataSchema:
    """
    Return a deterministic test schema.
    """

    return DataSchema(
        columns=[
            ColumnSchema(
                name="DATA",
                dtype="datetime64[ns]",
            ),
            ColumnSchema(
                name="COD CLIENTE",
                dtype="string",
            ),
            ColumnSchema(
                name="VOLUME",
                dtype="float64",
            ),
            ColumnSchema(
                name="observacao",
                dtype="string",
                required=False,
            ),
        ],
    )


@pytest.fixture
def validator(
    schema: DataSchema,
) -> SchemaValidator:
    """
    Return a SchemaValidator configured with the test schema.
    """

    return SchemaValidator(
        schema=schema,
    )


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """
    Return a dataframe compatible with the test schema.
    """

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ],
            ),
            "COD CLIENTE": pd.Series(
                [
                    "001",
                    "002",
                    "003",
                ],
                dtype="string",
            ),
            "VOLUME": pd.Series(
                [
                    10.0,
                    20.0,
                    30.0,
                ],
                dtype="float64",
            ),
            "observacao": pd.Series(
                [
                    "A",
                    "B",
                    "C",
                ],
                dtype="string",
            ),
        },
    )


def test_validator_accepts_valid_dataframe(
    validator: SchemaValidator,
    valid_dataframe: pd.DataFrame,
) -> None:
    """
    Test that a valid dataframe passes validation.
    """

    result = validator.validate(
        valid_dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        valid_dataframe,
    )


def test_validator_returns_same_dataframe(
    validator: SchemaValidator,
    valid_dataframe: pd.DataFrame,
) -> None:
    """
    Test that validation returns the input dataframe.
    """

    result = validator.validate(
        valid_dataframe,
    )

    assert result is valid_dataframe


def test_validator_accepts_optional_column_when_missing(
    validator: SchemaValidator,
) -> None:
    """
    Test that optional columns may be absent.
    """

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ],
            ),
            "COD CLIENTE": pd.Series(
                [
                    "001",
                    "002",
                ],
                dtype="string",
            ),
            "VOLUME": pd.Series(
                [
                    10.0,
                    20.0,
                ],
                dtype="float64",
            ),
        },
    )

    result = validator.validate(
        dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        dataframe,
    )


def test_validator_raises_missing_column_error(
    validator: SchemaValidator,
    valid_dataframe: pd.DataFrame,
) -> None:
    """
    Test missing required column validation.
    """

    dataframe = valid_dataframe.drop(
        columns=["VOLUME"],
    )

    with pytest.raises(
        MissingColumnError,
    ):
        validator.validate(
            dataframe,
        )


def test_validator_identifies_missing_column_name(
    validator: SchemaValidator,
    valid_dataframe: pd.DataFrame,
) -> None:
    """
    Test that the missing column is reported in the exception.
    """

    dataframe = valid_dataframe.drop(
        columns=["VOLUME"],
    )

    with pytest.raises(
        MissingColumnError,
        match="VOLUME",
    ):
        validator.validate(
            dataframe,
        )


def test_validator_raises_invalid_column_type_error(
    validator: SchemaValidator,
    valid_dataframe: pd.DataFrame,
) -> None:
    """
    Test invalid dataframe dtype validation.
    """

    dataframe = valid_dataframe.copy()

    dataframe["VOLUME"] = dataframe[
        "VOLUME"
    ].astype("int64")

    with pytest.raises(
        InvalidColumnTypeError,
    ):
        validator.validate(
            dataframe,
        )


def test_validator_identifies_invalid_column_type(
    validator: SchemaValidator,
    valid_dataframe: pd.DataFrame,
) -> None:
    """
    Test that the invalid column is reported.
    """

    dataframe = valid_dataframe.copy()

    dataframe["VOLUME"] = dataframe[
        "VOLUME"
    ].astype("int64")

    with pytest.raises(
        InvalidColumnTypeError,
        match="VOLUME",
    ):
        validator.validate(
            dataframe,
        )


def test_validator_validates_all_required_columns(
    schema: DataSchema,
) -> None:
    """
    Test validation of all required columns.
    """

    validator = SchemaValidator(
        schema=schema,
    )

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                ["2026-01-01"],
            ),
            "COD CLIENTE": pd.Series(
                ["001"],
                dtype="string",
            ),
            "VOLUME": pd.Series(
                [10.0],
                dtype="float64",
            ),
        },
    )

    result = validator.validate(
        dataframe,
    )

    assert list(result.columns) == [
        "DATA",
        "COD CLIENTE",
        "VOLUME",
    ]


@pytest.mark.parametrize(
    "column_name",
    [
        "DATA",
        "COD CLIENTE",
        "VOLUME",
    ],
)
def test_validator_detects_each_missing_required_column(
    validator: SchemaValidator,
    valid_dataframe: pd.DataFrame,
    column_name: str,
) -> None:
    """
    Test missing-column validation for every required column.
    """

    dataframe = valid_dataframe.drop(
        columns=[column_name],
    )

    with pytest.raises(
        MissingColumnError,
    ):
        validator.validate(
            dataframe,
        )