"""
Tests for OutputManager.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.utils.output_manager import OutputManager


@pytest.fixture
def output_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> OutputManager:
    """
    Return an OutputManager configured to use a temporary directory.
    """

    monkeypatch.setattr(
        "src.core.utils.output_manager.settings.data.output_path",
        str(tmp_path),
    )

    monkeypatch.setattr(
        "src.core.utils.output_manager.settings.data.runs_folder",
        "runs",
    )

    return OutputManager()


def test_run_id_has_expected_format(
    output_manager: OutputManager,
) -> None:
    """Test execution identifier format."""

    assert output_manager.run_id.startswith(
        "RUN_"
    )

    assert len(
        output_manager.run_id
    ) == len("RUN_YYYYMMDD_HHMMSS")


def test_run_path_is_inside_configured_output_directory(
    output_manager: OutputManager,
    tmp_path: Path,
) -> None:
    """Test run path construction."""

    expected_base = (
        tmp_path / "runs"
    )

    assert output_manager.run_path.parent == (
        expected_base
    )


def test_create_run_creates_run_directory(
    output_manager: OutputManager,
) -> None:
    """Test run directory creation."""

    run_path = output_manager.create_run()

    assert run_path.exists()
    assert run_path.is_dir()
    assert run_path == output_manager.run_path


@pytest.mark.parametrize(
    "directory_getter,directory_name",
    [
        (
            "get_forecasts_dir",
            "forecasts",
        ),
        (
            "get_metrics_dir",
            "metrics",
        ),
        (
            "get_models_dir",
            "models",
        ),
        (
            "get_reports_dir",
            "reports",
        ),
        (
            "get_explainability_dir",
            "explainability",
        ),
        (
            "get_logs_dir",
            "logs",
        ),
    ],
)
def test_create_run_creates_expected_directories(
    output_manager: OutputManager,
    directory_getter: str,
    directory_name: str,
) -> None:
    """Test creation of all expected output directories."""

    output_manager.create_run()

    directory = getattr(
        output_manager,
        directory_getter,
    )()

    assert directory == (
        output_manager.run_path
        / directory_name
    )

    assert directory.exists()
    assert directory.is_dir()


def test_metadata_file_is_created(
    output_manager: OutputManager,
) -> None:
    """Test metadata file creation."""

    output_manager.create_run()

    metadata_path = (
        output_manager.metadata_path()
    )

    assert metadata_path.exists()
    assert metadata_path.is_file()


def test_metadata_contains_execution_information(
    output_manager: OutputManager,
) -> None:
    """Test metadata contents."""

    output_manager.create_run()

    with output_manager.metadata_path().open(
        mode="r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["run_id"] == (
        output_manager.run_id
    )

    assert "started_at" in metadata

    assert metadata["status"] == "RUNNING"

    assert metadata["pipeline_version"] == (
        "2.0.0"
    )


def test_update_status_changes_metadata_status(
    output_manager: OutputManager,
) -> None:
    """Test execution status update."""

    output_manager.create_run()

    output_manager.update_status(
        "SUCCESS"
    )

    with output_manager.metadata_path().open(
        mode="r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["status"] == "SUCCESS"
    assert "finished_at" in metadata


def test_update_status_without_existing_metadata_creates_metadata(
    output_manager: OutputManager,
) -> None:
    """Test status update when metadata does not yet exist."""

    output_manager.run_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_manager.update_status(
        "FAILED"
    )

    metadata_path = (
        output_manager.metadata_path()
    )

    assert metadata_path.exists()

    with metadata_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["status"] == "FAILED"
    assert "finished_at" in metadata


def test_update_status_preserves_existing_metadata(
    output_manager: OutputManager,
) -> None:
    """Test that status updates preserve existing metadata."""

    output_manager.create_run()

    metadata_path = (
        output_manager.metadata_path()
    )

    with metadata_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        original_metadata = json.load(file)

    output_manager.update_status(
        "SUCCESS"
    )

    with metadata_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        updated_metadata = json.load(file)

    assert updated_metadata["run_id"] == (
        original_metadata["run_id"]
    )

    assert updated_metadata["started_at"] == (
        original_metadata["started_at"]
    )

    assert updated_metadata["pipeline_version"] == (
        original_metadata["pipeline_version"]
    )

    assert updated_metadata["status"] == "SUCCESS"
    assert "finished_at" in updated_metadata


def test_metadata_path_points_to_run_directory(
    output_manager: OutputManager,
) -> None:
    """Test metadata path."""

    assert output_manager.metadata_path() == (
        output_manager.run_path
        / "metadata.json"
    )


def test_create_run_is_idempotent(
    output_manager: OutputManager,
) -> None:
    """Test that creating the same run twice does not fail."""

    first_path = output_manager.create_run()
    second_path = output_manager.create_run()

    assert first_path == second_path
    assert first_path.exists()
    assert second_path.exists()