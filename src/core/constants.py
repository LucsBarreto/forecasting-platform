"""
constantes da aplicação.

valores estáticos utilizados em diferentes módulos do projeto.
"""

from pathlib import Path

PROJECT_NAME = "forecasting-platform"

CONFIGS_DIR = Path("configs")

DATA_DIR = Path("data")

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = Path("models")

OUTPUTS_DIR = Path("outputs")

REPORTS_DIR = Path("reports")

LOGS_DIR = Path("logs")

SUPPORTED_INPUT_EXTENSIONS = (

    ".xlsx",

    ".xls",

    ".xlsb",

    ".csv",

    ".parquet"

)

DEFAULT_TARGET_COLUMNS = (

    "VOLUME",

    "VALOR"

)

DEFAULT_RANDOM_STATE = 42