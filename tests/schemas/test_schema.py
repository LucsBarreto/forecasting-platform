"""testes do módulo de schema do projeto."""

from __future__ import annotations


def test_schema_module_can_be_imported() -> None:
    """valida que o módulo de schema está disponível para importação."""

    import src.schemas.schema as schema_module

    assert schema_module is not None