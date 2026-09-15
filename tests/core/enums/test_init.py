"""
Tests for the enums package.
"""

from src.core.enums import (
    FileType,
    FrequencyType,
    MetricType,
    ModelType,
    ValidationActionType,
)
from src.core.enums import __all__


def test_public_exports() -> None:
    assert sorted(__all__) == sorted(
        [
            "ValidationActionType",
            "ValidationCategory",
            "ModelType",
            "MetricType",
            "FrequencyType",
            "FileType",
        ]
    )


def test_validation_action_type_is_exported() -> None:
    """Test ValidationActionType export."""

    assert ValidationActionType.__name__ == "ValidationActionType"


def test_model_type_is_exported() -> None:
    """Test ModelType export."""

    assert ModelType.__name__ == "ModelType"


def test_metric_type_is_exported() -> None:
    """Test MetricType export."""

    assert MetricType.__name__ == "MetricType"


def test_frequency_type_is_exported() -> None:
    """Test FrequencyType export."""

    assert FrequencyType.__name__ == "FrequencyType"


def test_file_type_is_exported() -> None:
    """Test FileType export."""

    assert FileType.__name__ == "FileType"