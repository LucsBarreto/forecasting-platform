"""
Tests for application constants.
"""

from pathlib import Path

from src.core.constants import (
    CONFIGS_DIR,
    DATA_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_COLUMNS,
    INTERIM_DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_NAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SUPPORTED_INPUT_EXTENSIONS,
)


def test_project_name() -> None:
    """Test the application project name."""

    assert PROJECT_NAME == "forecasting-platform"


def test_configs_directory() -> None:
    """valida o diretório de configuração do projeto."""

    assert CONFIGS_DIR == Path(__file__).resolve().parents[2] / "configs"


def test_data_directory() -> None:
    """valida o diretório de dados do projeto."""

    assert DATA_DIR == Path(__file__).resolve().parents[2] / "data"


def test_raw_data_directory() -> None:
    """valida o diretório de dados brutos."""

    assert RAW_DATA_DIR == Path(__file__).resolve().parents[2] / "data" / "raw"


def test_interim_data_directory() -> None:
    """valida o diretório de dados intermediários."""

    assert INTERIM_DATA_DIR == Path(__file__).resolve().parents[2] / "data" / "interim"


def test_processed_data_directory() -> None:
    """valida o diretório de dados processados."""

    assert PROCESSED_DATA_DIR == Path(__file__).resolve().parents[2] / "data" / "processed"


def test_models_directory() -> None:
    """valida o diretório de modelos."""

    assert MODELS_DIR == Path(__file__).resolve().parents[2] / "models"


def test_outputs_directory() -> None:
    """valida o diretório de outputs do projeto."""

    assert OUTPUTS_DIR == Path(__file__).resolve().parents[2] / "outputs"


def test_reports_directory() -> None:
    """valida o diretório de relatórios."""

    assert REPORTS_DIR == Path(__file__).resolve().parents[2] / "reports"


def test_logs_directory() -> None:
    """valida o diretório de logs."""

    assert LOGS_DIR == Path(__file__).resolve().parents[2] / "logs"


def test_supported_input_extensions() -> None:
    """Test the supported input file extensions."""

    assert SUPPORTED_INPUT_EXTENSIONS == (
        ".xlsx",
        ".xls",
        ".xlsb",
        ".csv",
        ".parquet",
    )


def test_supported_input_extensions_is_tuple() -> None:
    """Test that supported extensions are immutable."""

    assert isinstance(
        SUPPORTED_INPUT_EXTENSIONS,
        tuple,
    )


def test_default_target_columns() -> None:
    """Test the default target columns."""

    assert DEFAULT_TARGET_COLUMNS == (
        "VOLUME",
        "VALOR",
    )


def test_default_target_columns_is_tuple() -> None:
    """Test that target columns are immutable."""

    assert isinstance(
        DEFAULT_TARGET_COLUMNS,
        tuple,
    )


def test_default_random_state() -> None:
    """Test the default random state."""

    assert DEFAULT_RANDOM_STATE == 42


def test_directory_constants_are_paths() -> None:
    """Test that directory constants use pathlib.Path."""

    directories = (
        CONFIGS_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    )

    assert all(
        isinstance(directory, Path)
        for directory in directories
    )