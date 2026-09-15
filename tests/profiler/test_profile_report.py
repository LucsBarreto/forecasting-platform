"""
Tests for ProfileReport.
"""

import pandas as pd

from src.profiler import ProfileMetrics, ProfileReport


def test_report_to_dict() -> None:
    """Test dictionary report."""

    metrics = ProfileMetrics(
        elapsed_seconds=1.5,
        memory_before_mb=10.0,
        memory_after_mb=15.0,
        memory_delta_mb=5.0,
    )

    report = ProfileReport(
        metrics=metrics,
    )

    assert report.to_dict() == {
        "elapsed_seconds": 1.5,
        "memory_before_mb": 10.0,
        "memory_after_mb": 15.0,
        "memory_delta_mb": 5.0,
    }


def test_report_to_dataframe() -> None:
    """Test dataframe report."""

    metrics = ProfileMetrics(
        elapsed_seconds=1.5,
        memory_before_mb=10.0,
        memory_after_mb=15.0,
        memory_delta_mb=5.0,
    )

    report = ProfileReport(
        metrics=metrics,
    )

    result = report.to_dataframe()

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert list(
        result.columns,
    ) == [
        "elapsed_seconds",
        "memory_before_mb",
        "memory_after_mb",
        "memory_delta_mb",
    ]

    assert len(result) == 1


def test_report_preserves_metric_values() -> None:
    """Test dataframe metric values."""

    metrics = ProfileMetrics(
        elapsed_seconds=2.0,
        memory_before_mb=20.0,
        memory_after_mb=30.0,
        memory_delta_mb=10.0,
    )

    report = ProfileReport(
        metrics=metrics,
    )

    result = report.to_dataframe()

    assert result.loc[
        0,
        "elapsed_seconds",
    ] == 2.0

    assert result.loc[
        0,
        "memory_before_mb",
    ] == 20.0

    assert result.loc[
        0,
        "memory_after_mb",
    ] == 30.0

    assert result.loc[
        0,
        "memory_delta_mb",
    ] == 10.0