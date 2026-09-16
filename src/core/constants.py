"""constantes globais da aplicação e caminhos absolutos do projeto."""

from pathlib import Path

PROJECT_NAME = "forecasting-platform"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

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