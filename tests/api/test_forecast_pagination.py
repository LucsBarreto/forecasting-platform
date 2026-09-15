"""Contract tests for paginated forecast data responses."""

import pytest
from fastapi import HTTPException

from src.api.services.forecast_pagination import (
    MAX_FORECAST_LIMIT,
    paginate_forecast_rows,
)


@pytest.fixture
def forecast_rows() -> list[dict[str, object]]:
    """Return intentionally unsorted rows for deterministic ordering proof."""
    return [
        {"date": "2026-05-01", "prediction": 140.0},
        {"date": "2026-01-01", "prediction": 100.0},
        {"date": "2026-04-01", "prediction": 130.0},
        {"date": "2026-03-01", "prediction": 120.0},
        {"date": "2026-02-01", "prediction": 110.0},
    ]


def test_pagination_returns_deterministic_chronological_pages(forecast_rows) -> None:
    """Pages must be chronological and expose stable pagination metadata."""
    first = paginate_forecast_rows("RUN_1", "VOLUME", forecast_rows, offset=0, limit=2)
    second = paginate_forecast_rows("RUN_1", "VOLUME", forecast_rows, offset=2, limit=2)
    third = paginate_forecast_rows("RUN_1", "VOLUME", forecast_rows, offset=4, limit=2)

    assert [row["date"] for row in first.data] == ["2026-01-01", "2026-02-01"]
    assert first.pagination.model_dump() == {
        "offset": 0,
        "limit": 2,
        "returned": 2,
        "has_next": True,
    }
    assert [row["date"] for row in second.data] == ["2026-03-01", "2026-04-01"]
    assert second.pagination.has_next is True
    assert [row["date"] for row in third.data] == ["2026-05-01"]
    assert third.pagination.has_next is False


def test_pagination_rejects_invalid_offset_and_limits(forecast_rows) -> None:
    """Invalid pagination values must map to the future HTTP 422 contract."""
    with pytest.raises(HTTPException) as negative_offset:
        paginate_forecast_rows("RUN_1", "VOLUME", forecast_rows, offset=-1, limit=2)
    assert negative_offset.value.status_code == 422

    with pytest.raises(HTTPException) as zero_limit:
        paginate_forecast_rows("RUN_1", "VOLUME", forecast_rows, offset=0, limit=0)
    assert zero_limit.value.status_code == 422

    with pytest.raises(HTTPException) as oversized_limit:
        paginate_forecast_rows(
            "RUN_1",
            "VOLUME",
            forecast_rows,
            offset=0,
            limit=MAX_FORECAST_LIMIT + 1,
        )
    assert oversized_limit.value.status_code == 422
