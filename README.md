# forecasting-platform

plataforma modular de previsão de demanda com ingestão, validação, pré-processamento, engenharia de features, treinamento temporal, avaliação, exportação e painel de observabilidade.

## fluxo técnico

```text
fonte
  ↓
validação
  ↓
preprocessamento
  ↓
seleção de features
  ↓
split temporal
  ↓
treinamento
  ↓
previsão
  ↓
avaliação
  ↓
exportação
```

## entrypoint

`main.py` é o ponto de entrada principal. a orquestração dos componentes é composta em memória e os artefatos de execução são materializados em `outputs/runs/<run_id>/`.

## contrato de dados

`config/data.yaml` é a fonte oficial de schema. o contrato exige as colunas:

```text
ANO
MES
DATA
FILIAL DESTINO
FILIAL ORIGEM
REGIONAL
UF
COD CLIENTE
NOME CLIENTE
COD ITEM
PRODUTO
MARCA
CATEGORIA
SUBCANAL GTM
CANAL GTM
ATENDIMENTO
VOLUME
VALOR
```

a coluna `DATA` é a data temporal e `VOLUME`/`VALOR` são as medidas principais. o projeto não cria aliases automáticos; o contrato deve ser respeitado por fonte de entrada.

## ambiente e execução

requisitos: python 3.13+ com ambiente de teste validado em python 3.14.

```powershell
python -m venv venv_forecast
.\venv_forecast\Scripts\python.exe -m pip install -r requirements.txt
python main.py
```

## configuração

* `configs/data.yaml`: contrato principal, paths, extensões, colunas e schema.
* `configs/features.yaml`: ativação de temporal, lag, rolling, trend, business e holidays.
* `configs/forecast.yaml`: alvos, horizonte, seed e proporções de split.
* `configs/models.yaml`: modelos e regressão automática.
* `configs/pipeline.yaml`: etapas do pipeline.
* `configs/validation.yaml`: regras de validação.
* `configs/logging.yaml`: nível e saída de logs.
* `configs/holidays.yaml`: feriados e regras sazonais.

## arquitetura principal

```text
forecasting-platform/
├── main.py
│   └── ponto de entrada principal que monta o fluxo de orquestração.
├── app.py
│   └── dashboard streamlit de exploração, ranking e análise de previsão.
├── configs/
│   └── contratos YAML de dados, features, forecast, models, validation, logging e holidays.
├── src/
│   ├── config/
│   │   └── contratos de configuração e carregadores de yaml.
│   ├── core/
│   │   └── constantes, enums, exceções e utilidades compartilhadas.
│   ├── data_loader/
│   │   └── descoberta e carga de arquivos de entrada.
│   ├── data_sources/
│   │   └── leitores de csv, excel, parquet, factory e sql.
│   ├── data_validation/
│   │   └── regras, relatórios e execução de validação.
│   ├── feature_engineering/
│   │   └── recurso temporal e calendário de feriados.
│   ├── feature_selection/
│   │   └── seleção de atributos e priorização.
│   ├── ml/
│   │   └── avaliação, forecast, modelos, treinamento e monitoring.
│   ├── pipelines/
│   │   └── train, evaluate e predict pipelines.
│   ├── preprocessing/
│   │   └── limpeza, dtype, data time e estratégia de missing.
│   ├── schemas/
│   │   └── contratos de schema de dados forçado ao runtime.
│   └── visualization/
│       └── gráficos, feature importance e relatórios de análise.
├── tests/
│   └── suíte automatizada de regressão e integração.
└── docs/
    └── documentação arquitural e de manutenção.
```

## manutenção

* o schema deve partir de `configs/data.yaml`;
* o pipeline operacional depende do modelo de config em yaml;
* a camada de validação deve manter compatibilidade com a estrutura do contrato;
* o módulo de modelos deve receber entrada consistente de features;
* o fluxo principal deve manter o histórico de saída em `outputs/runs/`.
