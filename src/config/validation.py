"""
configuração da validação de dados.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.core.enums import ValidationActionType


class DuplicateSettings(BaseModel):
    """configuração da validação de duplicatas."""

    enabled: bool = True

    subset: list[str] | None = None

    keep: str = "first"

    action: ValidationActionType = ValidationActionType.REMOVE


class FutureDateSettings(BaseModel):
    """configuração da validação de datas futuras."""

    enabled: bool = True

    column: str

    action: ValidationActionType

    allow_today: bool = False


class ZeroVolumeSettings(BaseModel):
    """configuração da validação de volume zero."""

    enabled: bool = True

    column: str

    action: ValidationActionType


class ZeroValueSettings(BaseModel):
    """configuração da validação de valor zero."""

    enabled: bool = True

    column: str

    action: ValidationActionType


class NegativeSalesSettings(BaseModel):
    """configuração da validação de vendas negativas."""

    enabled: bool = True

    column: str

    action: ValidationActionType


class PositiveReturnSettings(BaseModel):
    """configuração da validação de devoluções positivas."""

    enabled: bool = True

    column: str

    action: ValidationActionType


class NullCustomerSettings(BaseModel):
    """configuração da validação de clientes nulos."""

    enabled: bool = True

    column: str

    action: ValidationActionType


class NullProductSettings(BaseModel):
    """configuração da validação de produtos nulos."""

    enabled: bool = True

    column: str

    action: ValidationActionType


class MinimumHistorySettings(BaseModel):
    """configuração da validação do histórico mínimo."""

    enabled: bool = True

    column: str

    minimum_years: int = 2

    action: ValidationActionType


class ValidationSettings(BaseModel):
    """configuração do módulo de validação."""

    duplicate: DuplicateSettings

    future_date: FutureDateSettings

    zero_volume: ZeroVolumeSettings

    zero_value: ZeroValueSettings

    negative_sales: NegativeSalesSettings

    positive_return: PositiveReturnSettings

    null_customer: NullCustomerSettings

    null_product: NullProductSettings

    minimum_history: MinimumHistorySettings