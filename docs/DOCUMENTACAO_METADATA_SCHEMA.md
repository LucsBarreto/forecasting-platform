# Documentação do `MetadataSchema`

## Objetivo

Formalizar o sidecar de metadados produzido por `ForecastExporter` como um contrato Pydantic validável.

O contrato se aplica ao `forecast_metadata.json` associado ao forecast exportado. O `metadata.json` raiz criado por `OutputManager` continua sendo o resumo da execução e possui responsabilidade diferente.

## Campos obrigatórios

```json
{
  "run_id": "RUN_20260114_000000",
  "target": "VOLUME",
  "model": "LinearModel",
  "horizon": 2,
  "period": {
    "start": "2026-01-15",
    "end": "2026-01-16"
  },
  "version": "2.0.0",
  "timestamps": {
    "started_at": "2026-01-14T00:00:00Z",
    "finished_at": "2026-01-14T00:10:00Z"
  }
}
```

Regras:

- `run_id`: formato `RUN_YYYYMMDD_HHMMSS`;
- `target`: somente `VOLUME` ou `VALOR`;
- `model`: string não vazia;
- `horizon`: inteiro positivo;
- `period`: datas ISO com início menor ou igual ao fim;
- `version`: versão simples `major.minor.patch`;
- `timestamps`: datetimes ISO timezone-aware, com término posterior ou igual ao início.

`forecast_rows` e `final_metric` permanecem opcionais para preservar informações de auditoria já usadas pelo exporter.

O `ForecastExporter` valida o payload antes de escrever o CSV ou o sidecar. `FutureForecastContract`, `PredictPipeline` e os demais componentes de ML não conhecem JSON nem filesystem.