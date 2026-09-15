"""
Tests for the base data source.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data_sources.base import BaseSource


def test_base_source_cannot_be_instantiated() -> None:
    """Test that BaseSource cannot be instantiated directly."""

    with pytest.raises(TypeError):
        BaseSource()


def test_concrete_source_can_be_instantiated() -> None:
    """Test that a valid concrete source can be instantiated."""

    class ConcreteSource(BaseSource):
        """Concrete implementation used for testing."""

        def read(self, source: Path) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "VOLUME": [10, 20],
                    "VALOR": [100.0, 200.0],
                }
            )

    source = ConcreteSource()

    assert isinstance(source, BaseSource)


def test_concrete_source_read_returns_dataframe() -> None:
    """Test that a concrete source implements read correctly."""

    class ConcreteSource(BaseSource):
        """Concrete implementation used for testing."""

        def read(self, source: Path) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "VOLUME": [10, 20],
                    "VALOR": [100.0, 200.0],
                }
            )

    source = ConcreteSource()

    result = source.read(Path("data.csv"))

    assert isinstance(result, pd.DataFrame)


def test_concrete_source_read_returns_expected_DATA() -> None:
    """Test that the concrete source returns the expected dataframe."""

    class ConcreteSource(BaseSource):
        """Concrete implementation used for testing."""

        def read(self, source: Path) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "VOLUME": [10, 20],
                    "VALOR": [100.0, 200.0],
                }
            )

    source = ConcreteSource()

    result = source.read(Path("data.csv"))

    expected = pd.DataFrame(
        {
            "VOLUME": [10, 20],
            "VALOR": [100.0, 200.0],
        }
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_source_without_read_implementation_is_abstract() -> None:
    """Test that read must be implemented by subclasses."""

    class InvalidSource(BaseSource):
        """Invalid source without read implementation."""

        pass

    with pytest.raises(TypeError):
        InvalidSource()