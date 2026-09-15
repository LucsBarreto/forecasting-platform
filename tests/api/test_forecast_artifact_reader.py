"""Tests for limited forecast artifact reading."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from src.api.services.forecast_artifact_reader import ForecastArtifactReader


@pytest.fixture
def forecast_rows() -> list[dict[str, object]]:
    """Return a small chronologically ordered forecast fixture."""
    return [
        {"date": "2026-01-01", "prediction": 100.0},
        {"date": "2026-02-01", "prediction": 110.0},
        {"date": "2026-03-01", "prediction": 120.0},
        {"date": "2026-04-01", "prediction": 130.0},
        {"date": "2026-05-01", "prediction": 140.0},
    ]


def test_reader_reads_limited_csv_page_and_reports_next_page(
    tmp_path: Path,
    forecast_rows: list[dict[str, object]],
) -> None:
    """CSV reading must return only the requested page plus enough evidence for has_next."""
    artifact = tmp_path / "forecast_volume.csv"
    artifact.write_text(
        "date,prediction\n"
        + "\n".join(f"{row['date']},{row['prediction']}" for row in forecast_rows)
        + "\n",
        encoding="utf-8",
    )

    page = ForecastArtifactReader().read_page(artifact, offset=2, limit=2)

    assert page.rows == [
        {"date": "2026-03-01", "prediction": 120.0},
        {"date": "2026-04-01", "prediction": 130.0},
    ]
    assert page.has_next is True


def test_reader_reads_limited_parquet_page_and_reports_last_page(
    tmp_path: Path,
    forecast_rows: list[dict[str, object]],
) -> None:
    """Parquet reading must expose the same structured page contract as CSV."""
    artifact = tmp_path / "forecast_volume.parquet"
    table = pa.Table.from_pylist(forecast_rows)
    parquet.write_table(table, artifact)

    page = ForecastArtifactReader().read_page(artifact, offset=4, limit=2)

    assert page.rows == [
        {"date": "2026-05-01", "prediction": 140.0},
    ]
    assert page.has_next is False


def test_reader_rejects_invalid_or_unsupported_artifacts(tmp_path: Path) -> None:
    """Reader failures remain storage-level errors, not HTTP exceptions."""
    reader = ForecastArtifactReader()
    unsupported = tmp_path / "forecast.json"
    unsupported.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported forecast artifact format"):
        reader.read_page(unsupported, offset=0, limit=2)

    with pytest.raises(ValueError, match="offset"):
        reader.read_page(unsupported, offset=-1, limit=2)

    with pytest.raises(ValueError, match="limit"):
        reader.read_page(unsupported, offset=0, limit=0)


def test_reader_returns_empty_page_for_empty_csv(tmp_path: Path) -> None:
    """An empty artifact must produce a valid empty page instead of an internal failure."""
    artifact = tmp_path / "empty.csv"
    artifact.write_text("date,prediction\n", encoding="utf-8")

    page = ForecastArtifactReader().read_page(artifact, offset=0, limit=1000)

    assert page.rows == []
    assert page.has_next is False


def test_reader_rejects_malformed_csv_rows(tmp_path: Path) -> None:
    """Malformed CSV structure must fail as a reader error rather than leaking invalid JSON keys."""
    artifact = tmp_path / "malformed.csv"
    artifact.write_text(
        "date,prediction\n2026-01-01,100.0,unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed CSV"):
        ForecastArtifactReader().read_page(artifact, offset=0, limit=1000)
