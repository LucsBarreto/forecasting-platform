"""
Tests for profiler metrics.
"""

import pytest

from src.profiler.metrics import ProfileMetrics


def test_profile_metrics_stores_values() -> None:
    """Test metric values."""

    metrics = ProfileMetrics(
        elapsed_seconds=1.5,
        memory_before_mb=10.0,
        memory_after_mb=15.0,
        memory_delta_mb=5.0,
    )

    assert metrics.elapsed_seconds == 1.5
    assert metrics.memory_before_mb == 10.0
    assert metrics.memory_after_mb == 15.0
    assert metrics.memory_delta_mb == 5.0


def test_profile_metrics_to_dict() -> None:
    """Test dictionary conversion."""

    metrics = ProfileMetrics(
        elapsed_seconds=1.5,
        memory_before_mb=10.0,
        memory_after_mb=15.0,
        memory_delta_mb=5.0,
    )

    assert metrics.to_dict() == {
        "elapsed_seconds": 1.5,
        "memory_before_mb": 10.0,
        "memory_after_mb": 15.0,
        "memory_delta_mb": 5.0,
    }


def test_profile_metrics_is_immutable() -> None:
    """Test metric immutability."""

    metrics = ProfileMetrics(
        elapsed_seconds=1.0,
        memory_before_mb=10.0,
        memory_after_mb=12.0,
        memory_delta_mb=2.0,
    )

    with pytest.raises(
        AttributeError,
    ):
        metrics.elapsed_seconds = 2.0  # type: ignore[misc]