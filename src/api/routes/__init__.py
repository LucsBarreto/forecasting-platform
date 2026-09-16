"""Rotas HTTP mínimas do projeto."""

from src.api.routes.forecasts import router as forecasts_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.runs import router as runs_router

__all__ = ["forecasts_router", "metrics_router", "runs_router"]
