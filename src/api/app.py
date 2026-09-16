"""Aplicação FastAPI de exposição mínima do projeto."""

from fastapi import FastAPI

from src.api.routes.forecasts import router as forecasts_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.runs import router as runs_router


def create_app() -> FastAPI:
    """Cria a aplicação FastAPI com o mínimo contrato de saúde, o gateway de runs, o gateway de metrics e o gateway de forecasts."""
    app = FastAPI(title="forecasting-platform", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Retorna o estado de saúde do serviço HTTP."""
        return {
            "status": "ok",
            "service": "forecasting-platform",
        }

    app.include_router(runs_router)
    app.include_router(metrics_router)
    app.include_router(forecasts_router)

    return app
