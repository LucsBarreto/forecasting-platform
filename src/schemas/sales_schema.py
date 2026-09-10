"""
schema de vendas derivado do contrato oficial em ``configs/data.yaml``.

este módulo define o schema de vendas a partir dos tipos de dados
configurados no contrato oficial do dataset.
"""

from src.config import settings

from .models import ColumnSchema, DataSchema

SALES_SCHEMA = DataSchema(
    columns=[
        ColumnSchema(name=name, dtype=dtype)
        for name, dtype in settings.data.schema_config.dtypes.items()
    ]
)
