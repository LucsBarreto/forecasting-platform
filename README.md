# forecasting-platform

Plataforma modular de previsão comercial com ingestão, validação, preprocessamento, engenharia de features, treinamento temporal, avaliação de modelos, exportação de artefatos e painel analítico em Streamlit.

## Visão geral

A solução foi desenhada para operar como um pipeline de previsão orientado a negócio, com foco em dados de vendas, volume e valor por cliente, produto, canal, filial e período. A arquitetura separa claramente:

- entrada e contrato de dados
- validação e qualidade
- preprocessamento e feature engineering
- treinamento e comparação de modelos
- execução de previsões
- exportação de artefatos
- consumo em dashboard executivo

A lógica central segue o fluxo:

fonte -> validação -> preprocessamento -> feature engineering -> split temporal -> treinamento -> avaliação -> previsão -> exportação -> dashboard

## Objetivo do projeto

O projeto permite:

- ingerir bases históricas de vendas em CSV, Excel e Parquet
- validar qualidade e consistência dos dados de entrada
- preparar features temporais e de contexto comercial
- treinar múltiplos modelos e comparar desempenho
- gerar previsões para horizonte futuro
- materializar artefatos em diretórios de execução
- disponibilizar o resultado em um painel de observabilidade

## Estrutura principal

```text
forecasting-platform/
├── app.py
│   └── dashboard Streamlit para exploração histórica e consumo de previsões.
├── main.py
│   └── ponto de entrada principal do pipeline end-to-end.
├── requirements.txt
│   └── dependências do ambiente de execução.
├── pyproject.toml
│   └── metadados do projeto e configuração de ferramentas.
├── configs/
│   ├── data.yaml
│   │   └── contrato oficial do schema de entradas e caminhos.
│   ├── features.yaml
│   │   └── configuração de features temporais e de negócio.
│   ├── forecast.yaml
│   │   └── horizonte, seed e estratégia de split temporal.
│   ├── models.yaml
│   │   └── modelos habilitados e regras de AutoML.
│   ├── pipeline.yaml
│   │   └── etapas e ordem de execução do pipeline.
│   ├── validation.yaml
│   │   └── regras de qualidade e integridade dos dados.
│   ├── logging.yaml
│   │   └── configuração de logs do runtime.
│   ├── holidays.json
│   │   └── calendário de feriados e regras de influência sazonal.
│   └── holidays.yaml
│       └── variação de calendário utilizada pelo projeto.
├── src/
│   ├── config/
│   │   └── carregadores de YAML, contratos Pydantic e settings globais.
│   ├── core/
│   │   └── utilidades compartilhadas, enums, constantes e exceções.
│   ├── data_loader/
│   │   └── descoberta e leitura de fontes de dados.
│   ├── data_sources/
│   │   └── adaptadores para csv, excel, parquet, SQL e fábrica de origem.
│   ├── data_validation/
│   │   └── regras de validação, relatórios e execução das verificações.
│   ├── feature_engineering/
│   │   └── geração de features temporais e de calendário.
│   ├── feature_selection/
│   │   └── seleção de variáveis relevantes.
│   ├── ml/
│   │   ├── evaluation/
│   │   │   └── métricas e comparação de modelos.
│   │   ├── forecast/
│   │   │   └── previsão e output final.
│   │   ├── models/
│   │   │   └── modelos, adaptadores e baseline.
│   │   ├── monitoring/
│   │   │   └── observabilidade de execução.
│   │   └── training/
│   │       └── treinamento, split temporal e seleção de modelos.
│   ├── pipelines/
│   │   └── train, evaluate e predict pipelines.
│   ├── preprocessing/
│   │   └── limpeza, tipagem e transformação dos dados.
│   ├── schemas/
│   │   └── contratos de schema para a camada de dados.
│   └── visualization/
│       └── gráficos e relatórios de análise.
├── tests/
│   └── suíte de regressão para validação do comportamento dos módulos.
├── outputs/
│   └── runs/
│       └── diretórios de execução com metadata, métricas e previsões.
├── forecasting-platform-ops/
│   └── scripts de readiness e smoke checks operacionais.
├── docs/
│   └── material técnico, arquitetura e roadmap de evolução.
└── venv_forecast/
    └── ambiente virtual do projeto para execução local e testes.
```

## Contrato de dados

A fonte oficial de schema é [configs/data.yaml](configs/data.yaml). O contrato exige que as colunas abaixo existam no input:

- ANO
- MES
- DATA
- FILIAL DESTINO
- FILIAL ORIGEM
- REGIONAL
- UF
- COD CLIENTE
- NOME CLIENTE
- COD ITEM
- PRODUTO
- MARCA
- CATEGORIA
- SUBCANAL GTM
- CANAL GTM
- ATENDIMENTO
- VOLUME
- VALOR

A coluna DATA funciona como eixo temporal. VOLUME e VALOR são as métricas principais. O projeto não realiza mapeamento de alias automático; o contrato precisa ser respeitado para evitar inconsistências.

## Requisitos e execução

### Ambiente recomendado

- Python 3.11+ ou 3.12+, com ambiente equivalente ao usado no projeto
- ambiente virtual isolado para dependências

### Instalação

```powershell
python -m venv venv_forecast
.\venv_forecast\Scripts\python.exe -m pip install -r requirements.txt
```

### Execução do pipeline

```powershell
.\venv_forecast\Scripts\python.exe main.py
```

### Execução do dashboard

```powershell
.\venv_forecast\Scripts\python.exe -m streamlit run app.py
```

## Configuração por módulo

- [configs/data.yaml](configs/data.yaml): contrato de dados, paths, schema e extensões aceitas
- [configs/features.yaml](configs/features.yaml): ativação de feature engineering temporal e de negócio
- [configs/forecast.yaml](configs/forecast.yaml): horizonte, alvo, split e parâmetros de forecast
- [configs/models.yaml](configs/models.yaml): modelos disponíveis e regras de automação
- [configs/pipeline.yaml](configs/pipeline.yaml): sequência de etapas do pipeline
- [configs/validation.yaml](configs/validation.yaml): regras de qualidade e validação
- [configs/logging.yaml](configs/logging.yaml): nível e canal de logs
- [configs/holidays.json](configs/holidays.json): feriados e contexto sazonal

## Arquitetura do runtime

A camada de configuração centraliza regras de negócio e comportamento do pipeline. Em tempo de execução, as classes de `settings`, loaders e modelos consomem esse contrato e transformam a entrada em um fluxo previsível.

As responsabilidades principais são:

- [src/config/settings.py](src/config/settings.py): objeto central de configuração do runtime
- [src/data_sources/factory.py](src/data_sources/factory.py): criação da fonte correta conforme a extensão do arquivo
- [src/data_validation/validator.py](src/data_validation/validator.py): execução das regras de qualidade
- [src/preprocessing/preprocessor.py](src/preprocessing/preprocessor.py): normalização e limpeza dos dados
- [src/feature_engineering/temporal.py](src/feature_engineering/temporal.py): geração de features temporais e calendárias
- [src/ml/training/training_manager.py](src/ml/training/training_manager.py): orquestração do treinamento
- [src/ml/forecast/forecast_manager.py](src/ml/forecast/forecast_manager.py): execução de prerredições e exportação
- [src/core/utils/output_manager.py](src/core/utils/output_manager.py): criação de execuções e organização de artefatos

## Artefatos produzidos

Cada execução gera uma estrutura em [outputs/runs](outputs/runs) com:

- metadata.json: resumo da execução e contexto da run
- metrics/metrics.json: resultados por modelo e métricas avaliadas
- forecasts/: saídas previstas por modelo e alvo
- models/: modelos treinados serializados, quando aplicável
- logs/: logs e rastreabilidade operacional
- reports/: relatórios e resumos de execução

Esses artefatos são o ponto de integração entre backend e dashboard.

## API FastAPI

A camada HTTP está em `src/api` e consulta somente artefatos já materializados.
Ela não executa treinamento, seleção de modelos ou geração de forecast.

Contratos disponíveis:

```text
GET /health
GET /runs
GET /runs/{run_id}
GET /runs/{run_id}/metrics
GET /runs/{run_id}/forecasts
GET /runs/{run_id}/forecasts/{target}
GET /runs/{run_id}/forecasts/{target}/data
```

O endpoint de dados suporta paginação por `offset` e `limit`, com ordenação
cronológica e limite máximo de 1000 registros. Consulte o
[guia de utilização da API](docs/API_USAGE_GUIDE.md) e o
[relatório de auditoria técnica](docs/RELATORIO_AUDITORIA_TECNICA_2026-09-15.md)
para contratos, riscos e próximos passos.

## Manutenção e boas práticas

- manter o contrato em [configs/data.yaml](configs/data.yaml) como fonte única de verdade
- validar mudanças de schema em todos os módulos sensíveis antes de executar pipeline completo
- preservar o split temporal e o alinhamento entre treino e validação
- manter a documentação técnica sincronizada com o código
- garantir que os artefatos de execução sejam reprodutíveis e rastreáveis
- evitar que a camada de dashboard reproduza a lógica do backend em vez de apenas consumir resultados

## Como utilizar com outras bases

A plataforma foi projetada para ser reutilizável com outras bases, desde que o contrato técnico seja compatível. O modelo operacional é simples:

1. preparar uma nova base no mesmo formato do contrato ou adaptar as colunas do schema
2. ajustar os caminhos e file pattern em [configs/data.yaml](configs/data.yaml)
3. verificar se os valores de dimensão e target continuam consistentes
4. revisar os modelos habilitados em [configs/models.yaml](configs/models.yaml)
5. executar o pipeline e verificar os artefatos gerados em [outputs/runs](outputs/runs)
6. consumir o resultado no dashboard sem necessidade de reescrever a UI

Se a nova base tiver outra granularidade ou outra natureza operacional, o projeto ainda funciona com adaptação de schema e regras do domain. O ponto crítico é manter a mesma semântica temporal, de cliente, produto e de agregado financeiro.

Para bases muito diferentes, recomenda-se:

- mapear colunas em termos equivalentes aos do contrato
- verificar se o target continua sendo VOLUME e/ou VALOR
- revisar feriados, sazonalidade e presença de zero ou faltantes
- testar a validade do pipeline com uma amostra representativa

## Melhorias futuras

As melhorias esperadas para evolução do projeto incluem:

- padronização ainda maior do contrato de execução por run
- catálogo centralizado de execuções e comparações históricas
- observabilidade com logs estruturados e monitoramento operacional
- melhoria de performance do dashboard para bases muito grandes
- suporte a múltiplos targets e cenários concorrentes
- integração com orquestradores e pipelines automatizados em produção

## Conclusão

O projeto já está em uma etapa madura de implementação: ele combina arquitetura modular, contrato YAML, pipeline previsível, modelagem temporal e dashboard funcional. O principal diferencial está na combinação de governança de configuração com execução operacional, permitindo evoluir para um ambiente mais robusto e reutilizável em cenários empresariais reais.

