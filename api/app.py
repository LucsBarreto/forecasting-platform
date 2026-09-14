from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.runs import router as runs_router
from api.routers.metrics import router as metrics_router
from api.routers.forecasts import router as forecasts_router

app = FastAPI(
    title="Forecasting Platform API",
    version="1.0.0",
    description="API REST para consumo de execuções, métricas e previsões do projeto.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_router, prefix="/api/v1")
app.include_router(metrics_router, prefix="/api/v1")
app.include_router(forecasts_router, prefix="/api/v1")


@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
