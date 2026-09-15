"""
Tests for DataLoader.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data_loader.loader import DataLoader


def test_load_returns_consolidated_dataframe(
    tmp_path: Path,
) -> None:
    """Test that multiple datasets are concatenated."""

    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"

    file_a.touch()
    file_b.touch()

    dataframe_a = pd.DataFrame(
        {
            "cliente": [1],
            "VOLUME": [100.0],
        }
    )

    dataframe_b = pd.DataFrame(
        {
            "cliente": [2],
            "VOLUME": [200.0],
        }
    )

    datasource_a = MagicMock()
    datasource_a.read.return_value = dataframe_a

    datasource_b = MagicMock()
    datasource_b.read.return_value = dataframe_b

    with patch(
        "src.data_loader.loader.FileDiscovery.discover",
        return_value=[file_a, file_b],
    ), patch(
        "src.data_loader.loader.DataSourceFactory.create",
        side_effect=[
            datasource_a,
            datasource_b,
        ],
    ):

        result = DataLoader(tmp_path).load()

    expected = pd.concat(
        [
            dataframe_a,
            dataframe_b,
        ],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_load_discovers_files(
    tmp_path: Path,
) -> None:
    """Test that DataLoader uses FileDiscovery."""

    files = [
        tmp_path / "a.csv",
        tmp_path / "b.csv",
    ]

    datasource = MagicMock()
    datasource.read.return_value = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    with patch(
        "src.data_loader.loader.FileDiscovery.discover",
        return_value=files,
    ) as discover, patch(
        "src.data_loader.loader.DataSourceFactory.create",
        return_value=datasource,
    ):

        DataLoader(tmp_path).load()

    discover.assert_called_once_with()


def test_load_creates_DATAsource_for_each_file(
    tmp_path: Path,
) -> None:
    """Test that a datasource is created for each file."""

    files = [
        tmp_path / "a.csv",
        tmp_path / "b.csv",
    ]

    datasource = MagicMock()
    datasource.read.return_value = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    with patch(
        "src.data_loader.loader.FileDiscovery.discover",
        return_value=files,
    ), patch(
        "src.data_loader.loader.DataSourceFactory.create",
        return_value=datasource,
    ) as factory_create:

        DataLoader(tmp_path).load()

    assert factory_create.call_count == 2

    factory_create.assert_any_call(files[0])
    factory_create.assert_any_call(files[1])


def test_load_reads_each_file(
    tmp_path: Path,
) -> None:
    """Test that every datasource reads its corresponding file."""

    files = [
        tmp_path / "a.csv",
        tmp_path / "b.csv",
    ]

    datasource = MagicMock()
    datasource.read.return_value = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    with patch(
        "src.data_loader.loader.FileDiscovery.discover",
        return_value=files,
    ), patch(
        "src.data_loader.loader.DataSourceFactory.create",
        return_value=datasource,
    ):

        DataLoader(tmp_path).load()

    datasource.read.assert_any_call(files[0])
    datasource.read.assert_any_call(files[1])