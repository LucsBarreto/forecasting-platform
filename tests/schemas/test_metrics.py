"""Tests for the persisted metrics contract."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.schemas.metrics import MetricsSchema


@pytest.fixture
def valid_metrics() -> dict:
    """Return a complete multimodel metrics artifact payload."""
    return {
        "run_id": "RUN_20260911_090311",
        "target": "VOLUME",
        "metrics": {
            "baseline": 105.73,
            "linear_regression": 147.13,
            "lightgbm": 77.09,
            "catboost": 77.29,
        },
    }


def test_metrics_schema_accepts_complete_payload(valid_metrics: dict) -> None:
    schema = MetricsSchema.model_validate(valid_metrics)

    assert schema.run_id == "RUN_20260911_090311"
    assert schema.target == "VOLUME"
    assert schema.metrics["baseline"] == 105.73


@pytest.mark.parametrize(
    "field",
    ["run_id", "target", "metrics"],
)
def test_metrics_schema_rejects_missing_required_field(
    valid_metrics: dict,
    field: str,
) -> None:
    payload = deepcopy(valid_metrics)
    del payload[field]

    with pytest.raises(ValidationError):
        MetricsSchema.model_validate(payload)


@pytest.mark.parametrize(
    "run_id",
    ["", "run_20260911_090311", "RUN_20260911", "RUN_20260911_090311_extra"],
)
def test_metrics_schema_rejects_invalid_run_id(valid_metrics: dict, run_id: str) -> None:
    with pytest.raises(ValidationError):
        MetricsSchema.model_validate({**valid_metrics, "run_id": run_id})


def test_metrics_schema_accepts_only_supported_targets(valid_metrics: dict) -> None:
    for target in ["VOLUME", "VALOR"]:
        assert MetricsSchema.model_validate({**valid_metrics, "target": target}).target == target

    with pytest.raises(ValidationError):
        MetricsSchema.model_validate({**valid_metrics, "target": "SALES"})


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"": 1.0},
        {"baseline": "12.3"},
        {"baseline": float("nan")},
        {"baseline": float("inf")},
        {1: 12.3},
        {"baseline": {"mean": 12.3}},
    ],
)
def test_metrics_schema_rejects_invalid_metrics(valid_metrics: dict, metrics: dict) -> None:
    with pytest.raises(ValidationError):
        MetricsSchema.model_validate({**valid_metrics, "metrics": metrics})