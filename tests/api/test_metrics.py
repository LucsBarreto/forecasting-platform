"""Regression test for the HTTP metrics contract of a run."""

import json

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings


def test_runs_metrics_endpoint_returns_metrics_json_for_existing_run(tmp_path, monkeypatch) -> None:
    """GET /runs/{run_id}/metrics must return a normalized metrics JSON envelope without recalculating forecasts or ML metrics in the HTTP layer."""
    runs_root = tmp_path / settings.data.runs_folder
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root / "RUN_20260911_090311"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / "metrics.json"
    metrics_file.write_text(
        json.dumps(
            {
                "mae": 12.3,
                "rmse": 17.4,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(settings.data, "output_path", str(tmp_path))

    app = create_app()
    client = TestClient(app)

    response = client.get("/runs/RUN_20260911_090311/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "metrics": {
            "mae": 12.3,
            "rmse": 17.4,
        }
    }

    missing = client.get("/runs/RUN_DOES_NOT_EXIST/metrics")
    assert missing.status_code == 404
