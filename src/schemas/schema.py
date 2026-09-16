"""módulo de compatibilidade para a API de schema do projeto."""

from .models import ColumnSchema, DataSchema
from .sales_schema import SALES_SCHEMA
from .validator import SchemaValidator

__all__ = [
    "ColumnSchema",
    "DataSchema",
    "SALES_SCHEMA",
    "SchemaValidator",
]
