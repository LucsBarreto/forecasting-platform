"""
Tests for the sales dataset schema.
"""

from __future__ import annotations

from src.config import settings
from src.schemas.sales_schema import SALES_SCHEMA


def test_sales_schema_contains_expected_columns() -> None:
    """
    Test that the sales schema contains the expected columns.
    """

    expected_columns = settings.data.required_columns

    actual_columns = [
        column.name
        for column in SALES_SCHEMA.columns
    ]

    assert actual_columns == expected_columns


def test_sales_schema_contains_expected_dtypes() -> None:
    """
    Test that sales schema defines expected pandas dtypes.
    """

    assert SALES_SCHEMA.dtypes == settings.data.schema_config.dtypes


def test_sales_schema_required_columns() -> None:
    """
    Test that all sales schema columns are required.
    """

    assert SALES_SCHEMA.required_columns == settings.data.required_columns


def test_sales_schema_has_no_duplicate_column_names() -> None:
    """
    Test that the sales schema does not contain duplicated columns.

    This test intentionally exposes duplicated schema definitions.
    """

    column_names = [
        column.name
        for column in SALES_SCHEMA.columns
    ]

    assert len(column_names) == len(set(column_names))