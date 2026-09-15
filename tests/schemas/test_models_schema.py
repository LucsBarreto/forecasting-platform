"""
Tests for schema models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.models import ColumnSchema, DataSchema


def test_column_schema_stores_values() -> None:
    """
    Test that ColumnSchema stores the provided values.
    """

    column = ColumnSchema(
        name="VOLUME",
        dtype="float64",
        required=True,
        nullable=False,
    )

    assert column.name == "VOLUME"
    assert column.dtype == "float64"
    assert column.required is True
    assert column.nullable is False


def test_column_schema_uses_default_values() -> None:
    """
    Test ColumnSchema default values.
    """

    column = ColumnSchema(
        name="VOLUME",
        dtype="float64",
    )

    assert column.required is True
    assert column.nullable is True


def test_column_schema_requires_name() -> None:
    """
    Test that column name is mandatory.
    """

    with pytest.raises(ValidationError):
        ColumnSchema(
            dtype="float64",
        )


def test_column_schema_requires_dtype() -> None:
    """
    Test that column dtype is mandatory.
    """

    with pytest.raises(ValidationError):
        ColumnSchema(
            name="VOLUME",
        )


def test_data_schema_stores_columns() -> None:
    """
    Test that DataSchema stores column definitions.
    """

    columns = [
        ColumnSchema(
            name="DATA",
            dtype="datetime64[ns]",
        ),
        ColumnSchema(
            name="VOLUME",
            dtype="float64",
        ),
    ]

    schema = DataSchema(
        columns=columns,
    )

    assert schema.columns == columns
    assert len(schema.columns) == 2


def test_required_columns_returns_only_required_columns() -> None:
    """
    Test required_columns property.
    """

    schema = DataSchema(
        columns=[
            ColumnSchema(
                name="DATA",
                dtype="datetime64[ns]",
                required=True,
            ),
            ColumnSchema(
                name="observacao",
                dtype="string",
                required=False,
            ),
            ColumnSchema(
                name="VOLUME",
                dtype="float64",
                required=True,
            ),
        ],
    )

    assert schema.required_columns == [
        "DATA",
        "VOLUME",
    ]


def test_required_columns_returns_empty_list_when_no_columns_are_required() -> None:
    """
    Test required_columns when all columns are optional.
    """

    schema = DataSchema(
        columns=[
            ColumnSchema(
                name="observacao",
                dtype="string",
                required=False,
            ),
        ],
    )

    assert schema.required_columns == []


def test_dtypes_returns_column_dtype_mapping() -> None:
    """
    Test dtypes property.
    """

    schema = DataSchema(
        columns=[
            ColumnSchema(
                name="DATA",
                dtype="datetime64[ns]",
            ),
            ColumnSchema(
                name="VOLUME",
                dtype="float64",
            ),
            ColumnSchema(
                name="COD CLIENTE",
                dtype="string",
            ),
        ],
    )

    assert schema.dtypes == {
        "DATA": "datetime64[ns]",
        "VOLUME": "float64",
        "COD CLIENTE": "string",
    }


def test_dtypes_returns_empty_dict_for_empty_schema() -> None:
    """
    Test dtypes property for an empty schema.
    """

    schema = DataSchema(
        columns=[],
    )

    assert schema.dtypes == {}


def test_data_schema_requires_columns() -> None:
    """
    Test that DataSchema requires the columns field.
    """

    with pytest.raises(ValidationError):
        DataSchema()