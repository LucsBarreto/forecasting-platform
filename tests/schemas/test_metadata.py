"""Tests for the forecast metadata contract."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.schemas.metadata import MetadataSchema


@pytest.fixture
def valid_metadata() -> dict:
    """Return a complete forecast metadata payload."""
    return {
        "run_id": "RUN_20260114_000000",
        "target": "VOLUME",
        "models": ["LinearModel", "LightGBM"],
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


def test_metadata_schema_accepts_complete_payload(valid_metadata: dict) -> None:
    """A complete metadata envelope must validate and normalize temporal fields."""
    schema = MetadataSchema.model_validate(valid_metadata)

    assert schema.run_id == "RUN_20260114_000000"
    assert schema.target == "VOLUME"
    assert schema.models == ["LinearModel", "LightGBM"]
    assert schema.horizon == 2
    assert schema.period.start.isoformat() == "2026-01-15"
    assert schema.timestamps.started_at.isoformat() == "2026-01-14T00:00:00+00:00"


@pytest.mark.parametrize(
    "field",
    ["run_id", "target", "models", "horizon", "period", "version", "timestamps"],
)
def test_metadata_schema_rejects_missing_required_field(
    valid_metadata: dict,
    field: str,
) -> None:
    """Every minimum audit field must be required."""
    payload = deepcopy(valid_metadata)
    del payload[field]

    with pytest.raises(ValidationError):
        MetadataSchema.model_validate(payload)


@pytest.mark.parametrize(
    "run_id",
    ["", "run_20260114_000000", "RUN_20260114", "RUN_20260114_000000_extra"],
)
def test_metadata_schema_rejects_invalid_run_id(valid_metadata: dict, run_id: str) -> None:
    payload = {**valid_metadata, "run_id": run_id}

    with pytest.raises(ValidationError):
        MetadataSchema.model_validate(payload)


def test_metadata_schema_rejects_invalid_target(valid_metadata: dict) -> None:
    with pytest.raises(ValidationError):
        MetadataSchema.model_validate({**valid_metadata, "target": "SALES"})


@pytest.mark.parametrize("horizon", [0, -1, "two"])
def test_metadata_schema_rejects_invalid_horizon(valid_metadata: dict, horizon: object) -> None:
    with pytest.raises(ValidationError):
        MetadataSchema.model_validate({**valid_metadata, "horizon": horizon})


def test_metadata_schema_rejects_duplicate_models(valid_metadata: dict) -> None:
    with pytest.raises(ValidationError):
        MetadataSchema.model_validate({**valid_metadata, "models": ["LightGBM", "LightGBM"]})


def test_metadata_schema_rejects_invalid_period(valid_metadata: dict) -> None:
    with pytest.raises(ValidationError):
        MetadataSchema.model_validate(
            {
                **valid_metadata,
                "period": {"start": "not-a-date", "end": "2026-01-16"},
            }
        )


@pytest.mark.parametrize("version", ["", "v2", 2])
def test_metadata_schema_rejects_invalid_version(valid_metadata: dict, version: object) -> None:
    with pytest.raises(ValidationError):
        MetadataSchema.model_validate({**valid_metadata, "version": version})


def test_metadata_schema_rejects_invalid_timestamp(valid_metadata: dict) -> None:
    with pytest.raises(ValidationError):
        MetadataSchema.model_validate(
            {
                **valid_metadata,
                "timestamps": {"started_at": "yesterday", "finished_at": "2026-01-14T00:10:00Z"},
            }
        )