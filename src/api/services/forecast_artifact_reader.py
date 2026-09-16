"""Leitura limitada e agnóstica de artefatos de forecast."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet


@dataclass(frozen=True, slots=True)
class ForecastArtifactPage:
    """Página estruturada lida de um artefato sem dependência de transporte HTTP."""

    rows: list[dict[str, Any]]
    has_next: bool


class ForecastArtifactReader:
    """Lê CSV e Parquet em páginas sem conhecer target ou FastAPI."""

    def read_page(
        self,
        artifact_path: Path,
        offset: int,
        limit: int,
    ) -> ForecastArtifactPage:
        """Lê no máximo `limit + 1` registros para determinar `has_next`."""
        if offset < 0:
            raise ValueError("offset must be greater than or equal to zero")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)

        suffix = artifact_path.suffix.lower()
        if suffix == ".csv":
            rows = self._read_csv_page(artifact_path, offset, limit)
        elif suffix == ".parquet":
            rows = self._read_parquet_page(artifact_path, offset, limit)
        else:
            raise ValueError(f"Unsupported forecast artifact format: {suffix}")

        return ForecastArtifactPage(
            rows=rows[:limit],
            has_next=len(rows) > limit,
        )

    @staticmethod
    def _read_csv_page(
        artifact_path: Path,
        offset: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with artifact_path.open(mode="r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row_number, row in enumerate(reader):
                if row_number < offset:
                    continue
                if None in row or any(key is None for key in row):
                    raise ValueError("Malformed CSV row")
                rows.append({key: _normalize_csv_value(value) for key, value in row.items()})
                if len(rows) > limit:
                    break
        return rows

    @staticmethod
    def _read_parquet_page(
        artifact_path: Path,
        offset: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        parquet_file = parquet.ParquetFile(artifact_path)
        for batch in parquet_file.iter_batches(batch_size=max(limit, 1)):
            for row in batch.to_pylist():
                if offset > 0:
                    offset -= 1
                    continue
                rows.append(row)
                if len(rows) > limit:
                    return rows
        return rows


def _normalize_csv_value(value: str | None) -> Any:
    """Converte valores escalares CSV sem impor schema de negócio."""
    if isinstance(value, list):
        raise ValueError("Malformed CSV row")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
