# Guia de utilização da API REST

## Estado atual

A API FastAPI está implementada em `src/api` e expõe artefatos já produzidos pelo pipeline. Ela não executa treinamento, seleção de modelos ou geração de forecast.

A aplicação é criada por `src.api.app:create_app` e, localmente, pode ser iniciada com:

```powershell
.\venv_forecast\Scripts\python.exe -m uvicorn src.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

A documentação OpenAPI fica disponível em:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

Não existe prefixo `/api/v1` na implementação atual. A versão futura da API deve ser introduzida deliberadamente, com migração e compatibilidade definidas.

## Arquitetura

```text
HTTP/JSON
   ↓
src/api/routes
   ↓
src/api/services
   ↓
outputs/runs/<run_id>
   ├── metadata.json
   ├── metrics/metrics.json
   └── forecasts/*.{csv,parquet}
```

O router recebe parâmetros HTTP e delega. O service resolve runs, targets e artefatos. O `ForecastArtifactReader` oculta CSV/Parquet. Nenhuma rota chama `PredictPipeline` ou `FutureForecastContract`.

## Contratos disponíveis

### Health

```http
GET /health
```

Resposta:

```json
{
  "status": "ok",
  "service": "forecasting-platform"
}
```

### Listar runs

```http
GET /runs
```

Resposta inicial:

```json
{
  "runs": [
    {"run_id": "RUN_20260911_090311"}
  ]
}
```

A rota não expõe caminhos absolutos nem conteúdo bruto do filesystem.

### Detalhar run

```http
GET /runs/{run_id}
```

Resposta atual:

```json
{"run_id": "RUN_20260911_090311"}
```

Run inexistente retorna `404`.

### Ler métricas

```http
GET /runs/{run_id}/metrics
```

A API lê `metrics/metrics.json` já materializado:

```json
{
  "metrics": {
    "mae": 12.3,
    "rmse": 17.4
  }
}
```

A API não recalcula métricas.

### Descobrir forecasts

```http
GET /runs/{run_id}/forecasts
```

Resposta:

```json
{
  "forecasts": [
    {"filename": "forecast_volume.csv", "target": "VOLUME"},
    {"filename": "forecast_valor.csv", "target": "VALOR"}
  ]
}
```

### Selecionar forecast por target

```http
GET /runs/{run_id}/forecasts/{target}
```

Targets são identidades de negócio. Atualmente, `VOLUME` e `VALOR` são reconhecidos. Um filename arbitrário não é aceito.

Resposta:

```json
{
  "filename": "forecast_volume.csv",
  "target": "VOLUME"
}
```

Target, run ou artefato inexistente retorna `404`.

### Ler dados paginados

```http
GET /runs/{run_id}/forecasts/{target}/data?offset=0&limit=100
```

Resposta:

```json
{
  "run_id": "RUN_20260911_090311",
  "target": "VOLUME",
  "data": [
    {
      "date": "2026-01-01",
      "prediction": 100.0
    }
  ],
  "pagination": {
    "offset": 0,
    "limit": 100,
    "returned": 1,
    "has_next": false
  }
}
```

Regras:

- `offset` padrão: `0`;
- `limit` padrão: `100`;
- limite máximo: `1000`;
- registros são ordenados cronologicamente por `date`;
- `offset < 0` ou limite fora de `1..1000` retorna `422`;
- run/target/artefato inexistente retorna `404`;
- artefato vazio retorna página vazia;
- artefato ilegível retorna erro controlado sem caminho físico.

## Como adicionar uma rota

1. Defina ou atualize o router em `src/api/routes/`.
2. Mantenha a função HTTP fina e livre de pandas, CSV, Parquet e filesystem.
3. Coloque descoberta, leitura e normalização em `src/api/services/`.
4. Crie ou reutilize schemas Pydantic em `src/api/schemas/`.
5. Registre o router em `src/api/app.py`.
6. Adicione testes com `fastapi.testclient.TestClient`.
7. Atualize a documentação em `docs/`.
8. Execute a suíte focada e depois `pytest -q`.

## Verificação atual

A suíte API cobre as sete fronteiras HTTP, o reader CSV/Parquet, paginação, OpenAPI e hardening:

```text
20 passed, 0 warnings
```

A suíte completa do projeto, após transformar os diretórios de testes em pacotes para evitar colisões de basenames, apresenta:

```text
893 passed, 0 warnings
```

O warning do `TestClient` foi removido com a dependência suportada `httpx2`.

## Limites da API atual

Ainda não fazem parte do contrato:

- criação de runs;
- disparo de treinamento ou forecast;
- autenticação e autorização;
- filtros dimensionais;
- paginação por cursor/keyset;
- catálogo enriquecido de modelos e targets;
- dashboard Next.js.

A evolução deve continuar orientada por contratos, testes e artefatos já persistidos.

## Dependencias de teste validadas

O warning anterior do `TestClient` vinha do fallback do Starlette para `httpx`.
O projeto agora declara `httpx2` para usar o caminho suportado pelo Starlette.

Combinacao validada no ambiente `venv_forecast`:

```text
FastAPI 0.140.13
Starlette 1.3.1
httpx2 2.13.0
pytest 9.1.1
```

O `pyproject.toml` e a fonte declarativa principal. O repositorio ainda nao
possui `poetry.lock` e o executavel Poetry nao esta instalado neste ambiente;
portanto, a combinacao acima foi registrada a partir do ambiente efetivamente
validado, sem criar um lockfile manual.
