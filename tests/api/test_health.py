"""Regression test for the HTTP health contract."""

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_health_endpoint_returns_service_status() -> None:
    """The first HTTP contract must be deliberately small and prove a healthy service envelope."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "forecasting-platform",
    }
