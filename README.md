# forecasting-platform

Plataforma modular de previsao comercial com ingestao, validacao, engenharia de features, treinamento temporal, forecast futuro, exportacao auditavel e consulta HTTP de artefatos.

## Estado atual:

O projeto possui um pipeline de machine learning separado de uma camada FastAPI de consulta. A API le artefatos ja produzidos e nao executa treinamento, selecao de modelos ou geracao de forecast.

Evidencia da ultima auditoria tecnica:

```text
893 passed, 0 warnings
```

O `TestClient` usa a dependência suportada `httpx2` no ambiente virtual.

## Fluxo principal

```text
entrada de dados
    -> validacao e schema
    -> preprocessamento
    -> feature engineering
    -> split temporal
    -> backtesting
    -> selecao do modelo vencedor
    -> treinamento final
    -> features futuras
    -> forecast futuro
    -> exportacao CSV + metadata
    -> consulta FastAPI / dashboard
```

## Componentes principais

- `main.py`: ponto de composicao do pipeline principal.
- `app_streamlit.py`: dashboard Streamlit para consulta e visualizacao.
- `src/config/`: carregamento dos contratos YAML e configuracoes Pydantic.
- `src/data_sources/`: adaptadores CSV, Excel, Parquet e SQL.
- `src/data_validation/`: regras e relatorios de qualidade.
- `src/preprocessing/`: limpeza, tipagem e transformacao.
- `src/feature_engineering/`: features temporais e de negocio.
- `src/ml/training/`: backtesting, selecao e treinamento final.
- `src/ml/forecast/`: contratos de forecast futuro e resultado final.
- `src/pipelines/`: pipelines de treino, avaliacao e predicao.
- `src/export/`: materializacao de forecasts e modelos.
- `src/api/`: API FastAPI para consulta de runs e artefatos.
- `forecasting-platform-ops/`: readiness e smoke checks operacionais.
- `tests/`: suite de regressao do projeto.

## Contrato de dados

A fonte oficial e [configs/data.yaml](configs/data.yaml). As colunas obrigatorias sao:

```text
ANO, MES, DATA, FILIAL DESTINO, FILIAL ORIGEM, REGIONAL, UF,
COD CLIENTE, NOME CLIENTE, COD ITEM, PRODUTO, MARCA, CATEGORIA,
SUBCANAL GTM, CANAL GTM, ATENDIMENTO, VOLUME, VALOR
```

`DATA` e o eixo temporal. `VOLUME` e `VALOR` sao os targets principais. O contrato nao aplica aliases automaticos: bases de entrada devem respeitar os nomes definidos no schema.

## Requisitos

- Python `>=3.13,<3.15`, conforme `pyproject.toml`.
- Ambiente virtual recomendado: `venv_forecast`.
- Dependencias principais: pandas, numpy, PyArrow, scikit-learn, CatBoost, LightGBM, XGBoost, FastAPI e Streamlit.

## Instalacao

```powershell
.\venv_forecast\Scripts\python.exe -m pip install -r requirements.txt
```

Para criar o ambiente:

```powershell
python -m venv venv_forecast
.\venv_forecast\Scripts\python.exe -m pip install -r requirements.txt
```

## Execucao

### Pipeline

```powershell
.\venv_forecast\Scripts\python.exe main.py
```

### Dashboard

```powershell
.\venv_forecast\Scripts\python.exe -m streamlit run app_streamlit.py
```

### API FastAPI

A aplicacao e criada por `src.api.app:create_app`:

```powershell
.\venv_forecast\Scripts\python.exe -m uvicorn src.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

Documentacao interativa:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

A implementacao atual nao usa prefixo `/api/v1`.

## API disponivel

```text
GET /health
GET /runs
GET /runs/{run_id}
GET /runs/{run_id}/metrics
GET /runs/{run_id}/forecasts
GET /runs/{run_id}/forecasts/{target}
GET /runs/{run_id}/forecasts/{target}/data
```

A rota de dados aceita `offset` e `limit`:

```text
GET /runs/RUN_20260911_090311/forecasts/VOLUME/data?offset=0&limit=100
```

Regras atuais:

- ordenacao cronologica crescente por `date`;
- `offset` padrao igual a `0`;
- `limit` padrao igual a `100`;
- limite maximo igual a `1000`;
- parametros invalidos retornam `422`;
- run, target ou artefato inexistente retornam `404`;
- CSV e Parquet sao lidos por `ForecastArtifactReader`;
- a API nao expoe caminhos absolutos do filesystem.

Consulte [docs/API_USAGE_GUIDE.md](docs/API_USAGE_GUIDE.md) para exemplos de respostas e regras completas.

## Artefatos de uma execucao

As execucoes sao organizadas em `outputs/runs/<run_id>/`:

```text
outputs/runs/<run_id>/
├── metadata.json
├── metrics/
│   └── metrics.json
├── forecasts/
│   ├── forecast_*.csv
│   └── forecast_*.parquet
├── models/
├── reports/
├── explainability/
└── logs/
```

O `ForecastExporter` e o unico ponto responsavel por materializar forecasts. O resultado futuro segue a fronteira:

```text
FutureForecastResult
    -> ForecastExporter
    -> forecast.csv + forecast_metadata.json
```

Os metadados devem identificar, conforme o contrato do artefato, `run_id`, target, modelo, horizonte, periodo e metrica final.

## Arquitetura de responsabilidades

```text
Backtester
    -> BacktestModelSelector
    -> FinalModelTrainer
    -> PredictPipeline
    -> FutureForecastContract
    -> FutureForecastResult
    -> ForecastExporter
    -> artefatos persistidos
    -> FastAPI / Streamlit
```

Regras importantes:

- `Backtester` produz evidencia; nao escolhe o modelo.
- `BacktestModelSelector` seleciona o vencedor com base na evidencia.
- `FutureForecastContract` nao conhece filesystem.
- `PredictPipeline` nao persiste artefatos.
- `main.py` compoe o fluxo, sem concentrar regras de dominio.
- FastAPI representa artefatos; nao executa ML.
- O frontend nao deve acessar `outputs/` diretamente.

## Testes e validacao

Executar a suite completa:

```powershell
.\venv_forecast\Scripts\python.exe -m pytest -q
```

Executar somente a API:

```powershell
.\venv_forecast\Scripts\python.exe -m pytest tests/api -q
```

A suite de API cobre health, runs, metricas, descoberta de forecasts, selecao por target, dados paginados, CSV/Parquet, OpenAPI, erros de artefato e protecao contra path traversal.

## Documentacao tecnica

- [docs/API_USAGE_GUIDE.md](docs/API_USAGE_GUIDE.md): contratos e uso da API.
- [docs/RELATORIO_AUDITORIA_TECNICA_2026-09-15.md](docs/RELATORIO_AUDITORIA_TECNICA_2026-09-15.md): auditoria, riscos e roadmap.
- [docs/ARQUITETURA_E_RESUMO_PROJETO.md](docs/ARQUITETURA_E_RESUMO_PROJETO.md): arquitetura do projeto.
- [docs/DOCUMENTACAO_API_HARDENING.md](docs/DOCUMENTACAO_API_HARDENING.md): validacoes operacionais da API.
- [docs/DOCUMENTACAO_FORECAST_ARTIFACT_READER.md](docs/DOCUMENTACAO_FORECAST_ARTIFACT_READER.md): leitura limitada de artefatos.
- [docs/DOCUMENTACAO_FORECAST_SERVICE_COMPOSITION.md](docs/DOCUMENTACAO_FORECAST_SERVICE_COMPOSITION.md): composicao do service.
- [docs/DOCUMENTACAO_FORECAST_DATA_CONTRACT.md](docs/DOCUMENTACAO_FORECAST_DATA_CONTRACT.md): schema e paginacao.
- [docs/DOCUMENTACAO_MELHORIAS_FUTURAS.md](docs/DOCUMENTACAO_MELHORIAS_FUTURAS.md): prioridades de evolucao.

## Proximas prioridades

1. Corrigir o warning de compatibilidade do `TestClient/httpx`.
2. Formalizar schemas de `metadata.json` e `metrics.json`.
3. Validar uma execucao real end-to-end com artefatos consultaveis.
4. Definir autenticacao, autorizacao, CORS e limites operacionais.
5. Substituir heuristica de filename por manifesto oficial de artefatos.
6. Integrar Next.js somente apos fechar o contrato de consumo do frontend.
7. Avaliar cursor pagination e particionamento para forecasts muito grandes.

## Limitacoes atuais

Ainda nao fazem parte do contrato:

- disparo de treinamento pela API;
- disparo de forecast pela API;
- autenticacao e autorizacao;
- filtros dimensionais;
- paginacao por cursor;
- catalogo enriquecido de modelos e targets;
- integracao com Next.js.
