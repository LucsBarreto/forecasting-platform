"""
Tests for the data source registry.
"""

import pytest

from src.data_sources.base import BaseSource
from src.data_sources.registry import DataSourceRegistry


class DummySource(BaseSource):
    """Dummy data source used for testing."""

    def read(self, source):
        """Read data from a source."""

        raise NotImplementedError


class AnotherDummySource(BaseSource):
    """Second dummy data source used for testing."""

    def read(self, source):
        """Read data from a source."""

        raise NotImplementedError


@pytest.fixture
def registry() -> DataSourceRegistry:
    """Return an empty data source registry."""

    return DataSourceRegistry()


def test_register_reader(
    registry: DataSourceRegistry,
) -> None:
    """Test registering a reader."""

    registry.register(
        ".csv",
        DummySource,
    )

    assert registry.get(".csv") is DummySource


def test_get_registered_reader(
    registry: DataSourceRegistry,
) -> None:
    """Test retrieving a registered reader."""

    registry.register(
        ".csv",
        DummySource,
    )

    result = registry.get(".csv")

    assert result is DummySource


def test_register_normalizes_extension_without_dot(
    registry: DataSourceRegistry,
) -> None:
    """Test that extensions without a dot are normalized."""

    registry.register(
        "csv",
        DummySource,
    )

    assert registry.get(".csv") is DummySource


def test_register_normalizes_uppercase_extension(
    registry: DataSourceRegistry,
) -> None:
    """Test that uppercase extensions are normalized."""

    registry.register(
        ".CSV",
        DummySource,
    )

    assert registry.get(".csv") is DummySource


def test_get_normalizes_extension(
    registry: DataSourceRegistry,
) -> None:
    """Test that get normalizes the requested extension."""

    registry.register(
        ".csv",
        DummySource,
    )

    assert registry.get("CSV") is DummySource


def test_contains_returns_true_for_registered_extension(
    registry: DataSourceRegistry,
) -> None:
    """Test contains for a registered extension."""

    registry.register(
        ".csv",
        DummySource,
    )

    assert registry.contains(".csv") is True


def test_contains_returns_false_for_unregistered_extension(
    registry: DataSourceRegistry,
) -> None:
    """Test contains for an unregistered extension."""

    assert registry.contains(".csv") is False


def test_unregister_removes_reader(
    registry: DataSourceRegistry,
) -> None:
    """Test unregistering a reader."""

    registry.register(
        ".csv",
        DummySource,
    )

    registry.unregister(".csv")

    assert registry.contains(".csv") is False


def test_unregister_nonexistent_reader_does_not_raise(
    registry: DataSourceRegistry,
) -> None:
    """Test unregistering an unknown extension."""

    registry.unregister(".csv")

    assert registry.contains(".csv") is False


def test_clear_removes_all_readers(
    registry: DataSourceRegistry,
) -> None:
    """Test clearing the registry."""

    registry.register(
        ".csv",
        DummySource,
    )

    registry.register(
        ".parquet",
        AnotherDummySource,
    )

    registry.clear()

    assert registry.contains(".csv") is False
    assert registry.contains(".parquet") is False


def test_get_unregistered_extension_raises_key_error(
    registry: DataSourceRegistry,
) -> None:
    """Test that retrieving an unknown extension raises KeyError."""

    with pytest.raises(KeyError):
        registry.get(".csv")


def test_register_replaces_existing_reader(
    registry: DataSourceRegistry,
) -> None:
    """Test that registering the same extension replaces the reader."""

    registry.register(
        ".csv",
        DummySource,
    )

    registry.register(
        ".csv",
        AnotherDummySource,
    )

    assert registry.get(".csv") is AnotherDummySource