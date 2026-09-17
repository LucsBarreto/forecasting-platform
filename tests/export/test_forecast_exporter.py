"""
Tests for ForecastExporter.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions.data import DataError
from src.export.forecast_exporter import ForecastExporter
from src.ml.forecast.future_forecast import FutureForecastResult
from src.schemas.metadata import MetadataSchema


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return sample forecast data."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C002",
                "C003",
            ],
            "prediction": [
                100.0,
                120.0,
                140.0,
            ],
        }
    )


def test_export_csv_creates_file(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Test CSV export."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    result = exporter.export_csv(
        dataframe,
    )

    assert result.exists()
    assert result.name == "forecast.csv"


def test_export_csv_preserves_DATA(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Test that CSV export preserves dataframe values."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    result = exporter.export_csv(
        dataframe,
    )

    loaded = pd.read_csv(
        result,
    )

    assert loaded["COD CLIENTE"].tolist() == [
        "C001",
        "C002",
        "C003",
    ]

    assert loaded["prediction"].tolist() == [
        100.0,
        120.0,
        140.0,
    ]


def test_export_parquet_creates_file(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Test Parquet export."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    result = exporter.export_parquet(
        dataframe,
    )

    assert result.exists()
    assert result.name == "forecast.parquet"


def test_export_parquet_preserves_DATA(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Test that Parquet export preserves dataframe."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    result = exporter.export_parquet(
        dataframe,
    )

    loaded = pd.read_parquet(
        result,
    )

    pd.testing.assert_frame_equal(
        loaded,
        dataframe,
    )


def test_output_directory_is_created(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Test automatic output directory creation."""

    output = tmp_path / "nested" / "results"

    exporter = ForecastExporter(
        output_directory=output,
    )

    result = exporter.export_csv(
        dataframe,
    )

    assert output.exists()
    assert result.exists()


def test_empty_dataframe_raises_error(
    tmp_path: Path,
) -> None:
    """Test that empty dataframes are rejected."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        DataError,
        match="Cannot export an empty forecast dataframe",
    ):
        exporter.export_csv(
            pd.DataFrame(),
        )


def test_invalid_dataframe_type_raises_error(
    tmp_path: Path,
) -> None:
    """Test invalid dataframe types."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        TypeError,
        match="dataframe must be a pandas DataFrame",
    ):
        exporter.export_csv(
            [1, 2, 3],  # type: ignore[arg-type]
        )


def test_empty_filename_raises_error(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Test empty filename validation."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        DataError,
        match="Output filename must be a non-empty string",
    ):
        exporter.export_csv(
            dataframe,
            filename="",
        )


def test_filename_with_directory_raises_error(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> None:
    """Test that directory traversal is rejected."""

    exporter = ForecastExporter(
        output_directory=tmp_path,
    )

    with pytest.raises(
        DataError,
        match="Output filename must not contain directories",
    ):
        exporter.export_csv(
            dataframe,
            filename="nested/forecast.csv",
        )


def test_exporter_consumes_future_forecast_result(tmp_path: Path) -> None:
    """ForecastExporter should accept a FutureForecastResult contract and persist its predictions as a forecast artifact."""
    exporter = ForecastExporter(output_directory=tmp_path)

    result = FutureForecastResult(
        predictions=pd.Series([10.0, 11.0], index=[0, 1]),
        horizon=2,
    )

    exported_path = exporter.export_future_result(
        result,
        filename="future_forecast.csv",
    )

    assert exported_path.exists()
    assert exported_path.name == "future_forecast.csv"

    loaded = pd.read_csv(exported_path)
    assert loaded["prediction"].tolist() == [10.0, 11.0]


def test_exporter_writes_audit_metadata_sidecar(tmp_path: Path) -> None:
    """ForecastExporter should persist a metadata JSON sidecar that identifies the forecast artifact provenance without placing export concerns in the forecast contract."""
    exporter = ForecastExporter(output_directory=tmp_path)

    result = FutureForecastResult(
        predictions=pd.Series([10.0, 11.0], index=[0, 1]),
        horizon=2,
    )

    metadata = {
        "run_id": "RUN_20260114_000000",
        "target": "VOLUME",
        "model": "LinearModel",
        "horizon": 2,
        "period": {
            "start": "2026-01-15",
            "end": "2026-01-16",
        },
        "version": "2.0.0",
        "timestamps": {
            "started_at": "2026-01-14T00:00:00Z",
            "finished_at": "2026-01-14T00:10:00Z",
        },
    }

    exported_path = exporter.export_future_result(
        result,
        filename="future_forecast.csv",
        metadata=metadata,
    )

    metadata_path = tmp_path / "forecast_metadata.json"
    assert exported_path.exists()
    assert metadata_path.exists()

    with metadata_path.open(mode="r", encoding="utf-8") as file:
        saved = json.load(file)

    validated = MetadataSchema.model_validate(saved)
    assert validated.run_id == "RUN_20260114_000000"
    assert saved["run_id"] == "RUN_20260114_000000"
    assert saved["target"] == "VOLUME"
    assert saved["model"] == "LinearModel"
    assert saved["horizon"] == 2
    assert saved["period"] == {
        "start": "2026-01-15",
        "end": "2026-01-16",
    }
    assert saved["version"] == "2.0.0"
    assert saved["timestamps"]["started_at"] == "2026-01-14T00:00:00Z"


def test_exporter_rejects_inconsistent_forecast_metadata(tmp_path: Path) -> None:
    """The forecast exporter must reject metadata that contradicts the FutureForecastResult payload, so the exported CSV and metadata sidecar remain an auditable pair."""
    exporter = ForecastExporter(output_directory=tmp_path)

    result = FutureForecastResult(
        predictions=pd.Series([10.0, 11.0, 12.0], index=[0, 1, 2]),
        horizon=3,
    )

    metadata = {
        "run_id": "RUN_20260114_000000",
        "target": "VOLUME",
        "model": "LinearModel",
        "horizon": 5,
        "period": {
            "start": "2026-01-15",
            "end": "2026-01-16",
        },
        "version": "2.0.0",
        "timestamps": {
            "started_at": "2026-01-14T00:00:00Z",
            "finished_at": "2026-01-14T00:10:00Z",
        },
    }

    with pytest.raises(DataError, match="Forecast metadata is inconsistent"):
        exporter.export_future_result(
            result,
            filename="future_forecast.csv",
            metadata=metadata,
        )
