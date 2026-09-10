"""
configurações relacionadas aos logs.
"""

from __future__ import annotations

from pydantic import BaseModel


class LoggingSettings(BaseModel):
    """configuração de logging."""

    level: str

    console: bool

    file: bool

    directory: str

    filename: str

    rotation: str