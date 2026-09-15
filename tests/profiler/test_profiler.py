"""
Tests for Profiler.
"""

import time

import pytest

from src.profiler import ProfileMetrics, Profiler


def test_profiler_start_and_stop() -> None:
    """Test basic profiling."""

    profiler = Profiler()

    profiler.start()

    time.sleep(0.001)

    metrics = profiler.stop()

    assert isinstance(
        metrics,
        ProfileMetrics,
    )

    assert metrics.elapsed_seconds >= 0
    assert metrics.memory_before_mb >= 0
    assert metrics.memory_after_mb >= 0


def test_profiler_measures_elapsed_time() -> None:
    """Test elapsed time measurement."""

    profiler = Profiler()

    profiler.start()

    time.sleep(0.01)

    metrics = profiler.stop()

    assert metrics.elapsed_seconds >= 0.005


def test_profiler_tracks_memory() -> None:
    """Test memory measurements."""

    profiler = Profiler()

    profiler.start()

    values = [
        index
        for index in range(10_000)
    ]

    metrics = profiler.stop()

    assert metrics.memory_before_mb >= 0
    assert metrics.memory_after_mb >= 0

    # Keep the allocation alive until profiling stops.
    assert len(values) == 10_000


def test_profiler_metrics_property() -> None:
    """Test latest metrics property."""

    profiler = Profiler()

    assert profiler.metrics is None

    profiler.start()
    metrics = profiler.stop()

    assert profiler.metrics is metrics


def test_profiler_cannot_stop_before_start() -> None:
    """Test invalid stop operation."""

    profiler = Profiler()

    with pytest.raises(
        RuntimeError,
        match="Profiler has not been started",
    ):
        profiler.stop()


def test_profiler_cannot_start_twice() -> None:
    """Test duplicate start."""

    profiler = Profiler()

    profiler.start()

    with pytest.raises(
        RuntimeError,
        match="Profiler is already running",
    ):
        profiler.start()

    profiler.stop()


def test_profiler_measure_context_manager() -> None:
    """Test profiling context manager."""

    profiler = Profiler()

    with profiler.measure():
        time.sleep(0.001)

    assert profiler.metrics is not None
    assert profiler.metrics.elapsed_seconds >= 0


def test_profiler_stops_when_context_raises() -> None:
    """Test that profiling stops when execution raises."""

    profiler = Profiler()

    with pytest.raises(
        ValueError,
        match="test error",
    ):
        with profiler.measure():
            raise ValueError("test error")

    assert profiler.metrics is not None