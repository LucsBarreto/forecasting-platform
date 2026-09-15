"""Regression test for the HTTP forecasts discovery contract."""

import json

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings
from src.api.services.forecast_service import ForecastService


def test_forecasts_endpoint_returns_discovered_forecast_artifacts(tmp_path, monkeypatch) -> None:
    """GET /runs/{run_id}/forecasts must return a normalized JSON representation of persisted forecast artifact metadata without executing ML or forecast generation."""
    runs_root = tmp_path / settings.data.runs_folder
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root / "RUN_20260911_090311"
    run_dir.mkdir(parents=True, exist_ok=True)

    forecasts_dir = run_dir / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    forecast_one = forecasts_dir / "forecast_volume.csv"
    forecast_two = forecasts_dir / "forecast_valor.csv"
    forecast_one.write_text("prediction\n100\n", encoding="utf-8")
    forecast_two.write_text("prediction\n200\n", encoding="utf-8")

    metadata = run_dir / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "run_id": "RUN_20260911_090311",
                "status": "SUCCESS",
                "pipeline_version": "2.0.0",
                "started_at": "2026-01-01T00:00:00",
                "finished_at": "2026-01-01T00:10:00",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    app = create_app()
    client = TestClient(app)

    response = client.get("/runs/RUN_20260911_090311/forecasts")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "forecasts" in payload
    assert len(payload["forecasts"]) == 2
    assert all("filename" in item for item in payload["forecasts"])
    assert all("target" in item for item in payload["forecasts"])
    assert {item["target"] for item in payload["forecasts"]} == {"VOLUME", "VALOR"}

    missing = client.get("/runs/RUN_DOES_NOT_EXIST/forecasts")
    assert missing.status_code == 404


def test_forecast_target_endpoint_resolves_business_target(tmp_path, monkeypatch) -> None:
    """GET /runs/{run_id}/forecasts/{target} must resolve a target without accepting an arbitrary filename."""
    runs_root = tmp_path / settings.data.runs_folder
    forecasts_dir = runs_root / "RUN_20260911_090311" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    (forecasts_dir / "forecast_volume.csv").write_text(
        "prediction\n100\n",
        encoding="utf-8",
    )
    (forecasts_dir / "forecast_valor.csv").write_text(
        "prediction\n200\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    client = TestClient(create_app())

    response = client.get("/runs/RUN_20260911_090311/forecasts/VOLUME")

    assert response.status_code == 200
    assert response.json() == {
        "filename": "forecast_volume.csv",
        "target": "VOLUME",
    }

    assert client.get("/runs/RUN_20260911_090311/forecasts/UNKNOWN").status_code == 404
    assert client.get("/runs/RUN_20260911_090311/forecasts/forecast_volume.csv").status_code == 404


def test_forecast_service_composes_reader_and_data_contract(tmp_path, monkeypatch) -> None:
    """ForecastService must resolve the target, delegate bounded reading, and build the approved response envelope."""
    forecasts_dir = tmp_path / settings.data.runs_folder / "RUN_20260911_090311" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    (forecasts_dir / "forecast_volume.csv").write_text(
        "date,prediction\n"
        "2026-01-01,100.0\n"
        "2026-02-01,110.0\n"
        "2026-03-01,120.0\n"
        "2026-04-01,130.0\n"
        "2026-05-01,140.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    response = ForecastService().get_forecast_data(
        "RUN_20260911_090311",
        "VOLUME",
        offset=2,
        limit=2,
    )

    assert response.model_dump() == {
        "run_id": "RUN_20260911_090311",
        "target": "VOLUME",
        "data": [
            {"date": "2026-03-01", "prediction": 120.0},
            {"date": "2026-04-01", "prediction": 130.0},
        ],
        "pagination": {
            "offset": 2,
            "limit": 2,
            "returned": 2,
            "has_next": True,
        },
    }


def test_forecast_data_endpoint_returns_paginated_rows_for_volume_and_valor(
    tmp_path,
    monkeypatch,
) -> None:
    """The data endpoint must expose paginated rows while keeping storage hidden from HTTP."""
    forecasts_dir = tmp_path / settings.data.runs_folder / "RUN_20260911_090311" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    (forecasts_dir / "forecast_volume.csv").write_text(
        "date,prediction\n2026-01-01,100.0\n2026-02-01,110.0\n2026-03-01,120.0\n",
        encoding="utf-8",
    )
    (forecasts_dir / "forecast_valor.csv").write_text(
        "date,prediction\n2026-01-01,1000.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    client = TestClient(create_app())

    volume = client.get(
        "/runs/RUN_20260911_090311/forecasts/VOLUME/data?offset=1&limit=1"
    )
    assert volume.status_code == 200
    assert volume.json() == {
        "run_id": "RUN_20260911_090311",
        "target": "VOLUME",
        "data": [{"date": "2026-02-01", "prediction": 110.0}],
        "pagination": {
            "offset": 1,
            "limit": 1,
            "returned": 1,
            "has_next": True,
        },
    }

    valor = client.get("/runs/RUN_20260911_090311/forecasts/VALOR/data")
    assert valor.status_code == 200
    assert valor.json()["target"] == "VALOR"
    assert valor.json()["data"] == [{"date": "2026-01-01", "prediction": 1000.0}]


def test_forecast_data_endpoint_rejects_invalid_pagination_and_missing_resources(
    tmp_path,
    monkeypatch,
) -> None:
    """The data endpoint must preserve the 422/404 contract for invalid query and resources."""
    forecasts_dir = tmp_path / settings.data.runs_folder / "RUN_20260911_090311" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    (forecasts_dir / "forecast_volume.csv").write_text(
        "date,prediction\n2026-01-01,100.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))
    client = TestClient(create_app())

    assert client.get("/runs/RUN_20260911_090311/forecasts/VOLUME/data?limit=1001").status_code == 422
    assert client.get("/runs/RUN_20260911_090311/forecasts/VOLUME/data?offset=-1").status_code == 422
    assert client.get("/runs/RUN_20260911_090311/forecasts/UNKNOWN/data").status_code == 404
    assert client.get("/runs/RUN_DOES_NOT_EXIST/forecasts/VOLUME/data").status_code == 404


def test_forecast_data_endpoint_supports_parquet_exact_limit_and_beyond_last_page(
    tmp_path,
    monkeypatch,
) -> None:
    """The API must support Parquet, the inclusive limit ceiling, and an empty page beyond the end."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    forecasts_dir = tmp_path / settings.data.runs_folder / "RUN_20260911_090311" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {"date": "2026-01-01", "prediction": 100.0},
                {"date": "2026-02-01", "prediction": 110.0},
            ]
        ),
        forecasts_dir / "forecast_volume.parquet",
    )
    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))
    client = TestClient(create_app())

    exact_limit = client.get(
        "/runs/RUN_20260911_090311/forecasts/VOLUME/data?limit=1000"
    )
    assert exact_limit.status_code == 200
    assert exact_limit.json()["pagination"]["returned"] == 2
    assert exact_limit.json()["pagination"]["has_next"] is False

    beyond_last = client.get(
        "/runs/RUN_20260911_090311/forecasts/VOLUME/data?offset=1000&limit=2"
    )
    assert beyond_last.status_code == 200
    assert beyond_last.json()["data"] == []
    assert beyond_last.json()["pagination"]["has_next"] is False


def test_api_openapi_documents_paginated_forecast_data_contract() -> None:
    """The generated OpenAPI schema must expose the data route and its query constraints."""
    client = TestClient(create_app())

    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/runs/{run_id}/forecasts/{target}/data"]["get"]
    parameters = {parameter["name"]: parameter for parameter in operation["parameters"]}

    assert "ForecastDataResponse" in str(operation["responses"])
    assert parameters["offset"]["schema"]["minimum"] == 0
    assert parameters["limit"]["schema"]["minimum"] == 1
    assert parameters["limit"]["schema"]["maximum"] == 1000


def test_forecast_data_endpoint_hides_malformed_artifact_details(tmp_path, monkeypatch) -> None:
    """Malformed artifact errors must be controlled and must not expose its physical path."""
    forecasts_dir = tmp_path / settings.data.runs_folder / "RUN_20260911_090311" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    (forecasts_dir / "forecast_volume.csv").write_text(
        "date,prediction\n2026-01-01,100.0,unexpected\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    response = TestClient(create_app()).get(
        "/runs/RUN_20260911_090311/forecasts/VOLUME/data"
    )

    assert response.status_code == 422
    assert str(tmp_path) not in response.text
    assert "traceback" not in response.text.lower()


def test_forecast_data_endpoint_does_not_allow_run_path_traversal(tmp_path, monkeypatch) -> None:
    """A run identifier must not escape the configured runs directory."""
    outside_forecasts = tmp_path / "forecasts"
    outside_forecasts.mkdir(parents=True, exist_ok=True)
    (outside_forecasts / "forecast_volume.csv").write_text(
        "date,prediction\n2026-01-01,100.0\n",
        encoding="utf-8",
    )
    (tmp_path / settings.data.runs_folder).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    response = TestClient(create_app()).get(
        "/runs/../forecasts/VOLUME/data"
    )

    assert response.status_code in {404, 405}
