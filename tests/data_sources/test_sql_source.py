"""
Tests for the SQL data source.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.exceptions import DataLoadingError
from src.data_sources.sql_source import SqlSource


@pytest.fixture
def source() -> SqlSource:
    """Return a SQL source instance."""

    return SqlSource()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "cliente": ["A", "B", "C"],
            "VOLUME": [10, 20, 30],
            "VALOR": [100.0, 200.0, 300.0],
        }
    )


def test_read_returns_dataframe(
    source: SqlSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that SQL reading returns a dataframe."""

    connection = MagicMock()
    query = "SELECT * FROM vendas"

    with patch(
        "src.data_sources.sql_source.pd.read_sql",
        return_value=sample_dataframe,
    ) as mock_read_sql:

        result = source.read(
            query,
            connection,
        )

    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )

    mock_read_sql.assert_called_once_with(
        query,
        connection,
    )


def test_read_passes_query_to_pandas(
    source: SqlSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the SQL query is passed to pandas."""

    connection = MagicMock()
    query = "SELECT cliente, volume FROM vendas"

    with patch(
        "src.data_sources.sql_source.pd.read_sql",
        return_value=sample_dataframe,
    ) as mock_read_sql:

        source.read(
            query,
            connection,
        )

    mock_read_sql.assert_called_once_with(
        query,
        connection,
    )


def test_read_preserves_dataframe(
    source: SqlSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the returned dataframe is preserved."""

    original = sample_dataframe.copy(deep=True)

    connection = MagicMock()

    with patch(
        "src.data_sources.sql_source.pd.read_sql",
        return_value=sample_dataframe,
    ):

        source.read(
            "SELECT * FROM vendas",
            connection,
        )

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )


def test_read_propagates_dataframe_content(
    source: SqlSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that SQL results are returned unchanged."""

    connection = MagicMock()

    with patch(
        "src.data_sources.sql_source.pd.read_sql",
        return_value=sample_dataframe,
    ):

        result = source.read(
            "SELECT * FROM vendas",
            connection,
        )

    assert list(result.columns) == [
        "cliente",
        "VOLUME",
        "VALOR",
    ]

    assert len(result) == 3


def test_read_sql_error_raises_DATA_loading_error(
    source: SqlSource,
) -> None:
    """Test that SQL errors are converted to DataLoadingError."""

    connection = MagicMock()

    with patch(
        "src.data_sources.sql_source.pd.read_sql",
        side_effect=ValueError("invalid SQL query"),
    ):

        with pytest.raises(
            DataLoadingError,
            match="não foi possível ler os dados do banco",
        ):
            source.read(
                "SELECT * FROM vendas",
                connection,
            )


def test_read_preserves_original_exception(
    source: SqlSource,
) -> None:
    """Test that the original SQL exception is preserved."""

    connection = MagicMock()
    original_exception = ValueError("invalid SQL query")

    with patch(
        "src.data_sources.sql_source.pd.read_sql",
        side_effect=original_exception,
    ):

        with pytest.raises(
            DataLoadingError,
        ) as exc_info:

            source.read(
                "SELECT * FROM vendas",
                connection,
            )

    assert exc_info.value.__cause__ is original_exception