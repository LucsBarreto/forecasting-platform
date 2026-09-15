"""
Tests for ModelExporter.
"""

from pathlib import Path

import pytest

from src.core.exceptions.modeling import ModelingError
from src.export.model_exporter import ModelExporter


class MockModel:
    """Simple serializable model for testing."""

    def __init__(
        self,
        value: str,
    ) -> None:
        self.value = value


def test_save_creates_model_file(
    tmp_path: Path,
) -> None:
    """Test model persistence."""

    exporter = ModelExporter(
        output_directory=tmp_path,
    )

    model = MockModel(
        value="test",
    )

    result = exporter.save(
        model,
        "model.joblib",
    )

    assert result.exists()
    assert result.name == "model.joblib"


def test_save_and_load_preserves_model(
    tmp_path: Path,
) -> None:
    """Test model round-trip persistence."""

    exporter = ModelExporter(
        output_directory=tmp_path,
    )

    model = MockModel(
        value="test",
    )

    exporter.save(
        model,
        "model.joblib",
    )

    loaded = exporter.load(
        "model.joblib",
    )

    assert isinstance(
        loaded,
        MockModel,
    )

    assert loaded.value == "test"


def test_output_directory_is_created(
    tmp_path: Path,
) -> None:
    """Test automatic directory creation."""

    output = tmp_path / "nested" / "models"

    exporter = ModelExporter(
        output_directory=output,
    )

    result = exporter.save(
        MockModel("test"),
        "model.joblib",
    )

    assert output.exists()
    assert result.exists()


def test_none_model_raises_error(
    tmp_path: Path,
) -> None:
    """Test null model validation."""

    exporter = ModelExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        ModelingError,
        match="Cannot export a null model",
    ):
        exporter.save(
            None,
            "model.joblib",
        )


def test_empty_filename_raises_error(
    tmp_path: Path,
) -> None:
    """Test empty filename validation."""

    exporter = ModelExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        ModelingError,
        match="Model filename must be a non-empty string",
    ):
        exporter.save(
            MockModel("test"),
            "",
        )


def test_filename_with_directory_raises_error(
    tmp_path: Path,
) -> None:
    """Test directory traversal validation."""

    exporter = ModelExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        ModelingError,
        match="Model filename must not contain directories",
    ):
        exporter.save(
            MockModel("test"),
            "nested/model.joblib",
        )


def test_load_missing_model_raises_error(
    tmp_path: Path,
) -> None:
    """Test loading a nonexistent model."""

    exporter = ModelExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        ModelingError,
        match="Model file was not found",
    ):
        exporter.load(
            "missing.joblib",
        )