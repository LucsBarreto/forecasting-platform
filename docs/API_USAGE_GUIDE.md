# Guia de utilização da API REST

Este documento descreve como iniciar, consumir e ampliar a camada REST criada no diretório `api/` do projeto `forecasting-platform`.

## 1. Estrutura da API

A aplicação FastAPI é registrada em `api/app.py` e expose as rotas com prefixo `/api/v1`.

Arquivos principais:

- `api/app.py`: criação do app FastAPI, CORS e inclusão de routers.
- `api/routers/runs.py`: endpoints de execução/listagem de runs.
- `api/routers/metrics.py`: endpoints de métricas por execução.
- `api/routers/forecasts.py`: endpoints de artefatos de previsão.
- `api/services/run_service.py`: leitura de metadata e métricas JSON.
- `api/services/forecast_service.py`: listagem de candidatos de previsão.

## 2. Como iniciar a API

No diretório do projeto, instale as dependências do ambiente e execute:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Depois disso, a aplicação fica disponível em:

- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/api/v1/runs`
- `http://127.0.0.1:8000/api/v1/runs/{run_id}`
- `http://127.0.0.1:8000/api/v1/runs/{run_id}/metrics`
- `http://127.0.0.1:8000/api/v1/runs/{run_id}/forecasts`

## 3. Endpoints incluídos

### Health

```http
GET /api/v1/health
```

Retorna:

```json
{"status": "ok"}
```

### Listar runs

```http
GET /api/v1/runs
```

Lista os diretórios de execução disponíveis sob `outputs/runs`.

### Obter metadata de uma execução

```http
GET /api/v1/runs/{run_id}
```

Lê o arquivo `metadata.json` da execução.

### Obter métricas de uma execução

```http
GET /api/v1/runs/{run_id}/metrics
```

Lê o arquivo `metrics/metrics.json` e devolve uma lista serializável.

### Obter previsões candidatas de uma execução

```http
GET /api/v1/runs/{run_id}/forecasts
```

Varre arquivos `csv` e `parquet` dentro da execução.

## 4. Como adicionar novas rotas

Para adicionar um novo endpoint, siga esta convenção:

1. Crie ou edite um router em `api/routers/`.
2. Declare o objeto `APIRouter` com o prefixo correto.
3. Adicione a função endpoint com o decorador `@router.get`, `@router.post`, `@router.put` ou `@router.delete`.
4. Quando necessário, mova lógica de acesso a arquivos para `api/services/`.
5. Inclua o router em `api/app.py` com `app.include_router(...)`.

Exemplo de rota nova:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/runs", tags=["runs"])

@router.get("/{run_id}/summary")
def get_run_summary(run_id: str):
    return {"run_id": run_id, "summary": "ok"}
```

Depois, registre no `app.py`:

```python
from api.routers.summary import router as summary_router
app.include_router(summary_router, prefix="/api/v1")
```

## 5. Boas práticas

- Manter o acesso a arquivos em service functions e não diretamente nos endpoints.
- Preferir retorno JSON serializável com dicionários simples e listas.
- Centralizar a localização de diretórios de execução com `src.config.settings`.
- Guardar lógica de parsing de artefatos em serviços do módulo `api/services/`.
- Documentar o novo endpoint com docstring e exemplo de resposta.

## 6. Exemplo de uso com Python

```python
import requests

base = "http://127.0.0.1:8000/api/v1"
response = requests.get(f"{base}/runs", timeout=5)
print(response.status_code)
print(response.json())
```

## 7. Observação

A API foi criada como camada de exposição para artefatos já produzidos pelo projeto. Ela não substitui a interface Streamlit nem o fluxo de treinamento/exportação do pipeline.
