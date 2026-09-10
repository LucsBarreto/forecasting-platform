"""
logger da aplicação.

configura o logger usando o loguru e disponibiliza
o acesso ao logger configurado.
"""

from __future__ import annotations

import sys
from datetime import datetime
from functools import cache
from pathlib import Path

from loguru import logger

from src.config import settings
from src.core.constants import LOGS_DIR


def _build_log_file_path() -> Path:
    """constrói o caminho do arquivo de log da execução."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"forecast_{datetime.now():%Y%m%d_%H%M%S}.log"

    return LOGS_DIR / filename


@cache
def _configure_logger() -> None:
    """configura o logger uma única vez durante a execução da aplicação."""

    logger.remove()

    logger.add(
        sink=sys.stdout,
        level=settings.logging.level,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    logger.add(
        sink=_build_log_file_path(),
        level=settings.logging.level,
        rotation=settings.logging.rotation,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level:<8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
    )


def get_logger():
    """retorna o logger configurado pelo loguru."""

    _configure_logger()

    return logger