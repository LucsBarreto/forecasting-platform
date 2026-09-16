"""monta o contrato global de configuração do projeto."""

from pathlib import Path

from pydantic import BaseModel

from src.core.constants import CONFIGS_DIR, PROJECT_ROOT

from .data import DataSettings
from .features import FeatureSettings
from .forecast import ForecastSettings
from .loader import ConfigLoader
from .logging import LoggingSettings
from .models import ModelSettings
from .validation import ValidationSettings
from .pipeline import PipelineSettings


class Settings(BaseModel):
    """consolida os contratos YAML de configuração."""

    data: DataSettings
    forecast: ForecastSettings
    models: ModelSettings
    features: FeatureSettings
    logging: LoggingSettings
    validation: ValidationSettings
    pipeline: PipelineSettings


_loader = ConfigLoader(CONFIGS_DIR)


def _resolve_project_relative_path(value: str) -> str:
    """normaliza caminhos relativos para o diretório raiz do projeto."""

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_settings(config_dir: Path = CONFIGS_DIR) -> Settings:
    """carrega os contratos YAML do diretório de configuração."""

    loader = ConfigLoader(config_dir)

    resolved = Settings(
        data=DataSettings(**loader.load("data.yaml")),
        forecast=ForecastSettings(**loader.load("forecast.yaml")),
        models=ModelSettings(**loader.load("models.yaml")),
        features=FeatureSettings(**loader.load("features.yaml")),
        logging=LoggingSettings(**loader.load("logging.yaml")),
        validation=ValidationSettings(**loader.load("validation.yaml")),
        pipeline=PipelineSettings(**loader.load("pipeline.yaml")),
    )

    resolved.data.input_path = _resolve_project_relative_path(resolved.data.input_path)
    resolved.data.output_path = _resolve_project_relative_path(resolved.data.output_path)

    return resolved


settings = load_settings()