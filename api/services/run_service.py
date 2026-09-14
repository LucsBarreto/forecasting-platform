import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings


def list_runs() -> list[dict[str, Any]]:
    """lista as execuções disponíveis em outputs/runs."""
    runs_root = Path(settings.data.output_path) / settings.data.runs_folder
    if not runs_root.exists():
        return []

    runs = []
    for run_dir in sorted(runs_root.glob("RUN_*/"), reverse=True):
        metadata = load_run_metadata(run_dir)
        runs.append(metadata)

    return runs


def load_run_metadata(run_path: str | Path) -> dict[str, Any]:
    """carrega metadata.json de uma execução."""
    path = Path(run_path)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (TypeError, ValueError, OSError):
        return {}


def load_run_metrics(run_path: str | Path) -> list[dict[str, Any]]:
    """carrega as métricas da execução em um formato serializável para JSON."""
    path = Path(run_path)
    metrics_path = path / "metrics" / "metrics.json"
    if not metrics_path.exists():
        return []

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (TypeError, ValueError, OSError):
        return []

    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for model, values in payload.items():
            if isinstance(values, dict):
                for metric_name, metric_value in values.items():
                    rows.append({
                        "model": str(model),
                        "metric": str(metric_name),
                        "value": float(metric_value),
                    })
            else:
                rows.append({
                    "model": str(model),
                    "metric": "metric",
                    "value": float(values),
                })

    return rows
