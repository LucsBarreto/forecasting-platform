"""Regression test for the HTTP runs discovery contract."""

import json

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings


def test_runs_endpoint_returns_json_collection_of_runs(tmp_path, monkeypatch) -> None:
    """GET /runs must expose a normalized JSON collection and never leak raw filesystem paths."""
    runs_root = tmp_path / settings.data.runs_folder
    runs_root.mkdir(parents=True, exist_ok=True)

    run_one = runs_root / "RUN_20260911_090311"
    run_two = runs_root / "RUN_20260911_090312"
    run_one.mkdir(parents=True, exist_ok=True)
    run_two.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    app = create_app()
    client = TestClient(app)

    response = client.get("/runs")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "runs" in payload
    assert isinstance(payload["runs"], list)
    assert [item["run_id"] for item in payload["runs"]] == [
        "RUN_20260911_090311",
        "RUN_20260911_090312",
    ]
    assert "RUN_20260911_090311" in response.text
    assert "C:\\" not in response.text

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"


def test_runs_detail_endpoint_returns_a_single_run_json_and_404_for_unknown_run(
    tmp_path,
    monkeypatch,
) -> None:
    """GET /runs/{run_id} must return a normalized JSON object for an existing run and a 404 for an absent one."""
    runs_root = tmp_path / settings.data.runs_folder
    runs_root.mkdir(parents=True, exist_ok=True)

    existing = runs_root / "RUN_20260911_090311"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "RUN_20260911_090311",
                "status": "SUCCESS",
                "started_at": "2026-09-21T14:12:01",
                "finished_at": "2026-09-21T14:36:51",
                "pipeline_version": "2.0.0",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    app = create_app()
    client = TestClient(app)

    response = client.get("/runs/RUN_20260911_090311")
    assert response.status_code == 200
    assert response.json() == {
        "run_id": "RUN_20260911_090311",
        "status": "SUCCESS",
        "started_at": "2026-09-21T14:12:01",
        "finished_at": "2026-09-21T14:36:51",
    }

    missing = client.get("/runs/RUN_DOES_NOT_EXIST")
    assert missing.status_code == 404
    assert "run_id" not in missing.text


def test_runs_detail_endpoint_rejects_missing_or_invalid_metadata(tmp_path, monkeypatch) -> None:
    """GET /runs/{run_id} must not synthesize detail data without valid persisted metadata."""
    runs_root = tmp_path / settings.data.runs_folder
    runs_root.mkdir(parents=True, exist_ok=True)
    missing_metadata = runs_root / "RUN_WITHOUT_METADATA"
    missing_metadata.mkdir()
    invalid_metadata = runs_root / "RUN_WITH_INVALID_METADATA"
    invalid_metadata.mkdir()
    (invalid_metadata / "metadata.json").write_text("{invalid-json}", encoding="utf-8")
    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    client = TestClient(create_app())

    assert client.get("/runs/RUN_WITHOUT_METADATA").status_code == 422
    assert client.get("/runs/RUN_WITH_INVALID_METADATA").status_code == 422
