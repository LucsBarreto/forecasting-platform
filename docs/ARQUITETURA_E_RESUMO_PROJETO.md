arvore arquitetural resumida do projeto forecasting-platform

estrutura principal

forecasting-platform/
├── main.py
│   └── ponto de entrada da aplicação. compõe o pipeline principal e conecta o contrato de configuração com os componentes de preprocessamento, seleção, treinamento, avaliação e exportação.
├── app.py
│   └── painel streamlit para consulta, visualização e diagnóstico de previsões e métricas.
├── requirements.txt
│   └── dependências de execução do runtime do projeto.
├── pyproject.toml
│   └── metadados do projeto e ferramentas de desenvolvimento.
├── configs/
│   ├── data.yaml
│   │   └── contrato de entrada, schema, colunas obrigatórias, file_pattern e granularidade.
│   ├── features.yaml
│   │   └── ativação e família de features: temporal, lag, rolling, trend, business e holidays.
│   ├── forecast.yaml
│   │   └── alvos, horizonte, divisão temporal e seed.
│   ├── holidays.yaml
│   │   └── calendário de feriados com regras fixas, móveis e observacionais.
│   ├── logging.yaml
│   │   └── contrato de saída de logs em console e arquivo.
│   ├── models.yaml
│   │   └── modelos habilitados e métrica de comparação.
│   ├── pipeline.yaml
│   │   └── etapas e contrato de pipeline.
│   └── validation.yaml
│       └── validações de integridade, qualidade e qualidade histórica.
├── src/
│   ├── config/
│   │   ├── data.py
│   │   │   └── schema pydantic de dados com histórico mínimo e dtypes.
│   │   ├── features.py
│   │   │   └── contrato das families de feature engineering.
│   │   ├── forecast.py
│   │   │   └── horizonte e configuração de validação temporal.
│   │   ├── loader.py
│   │   │   └── leitura do contrato yaml e conversão para dicionário.
│   │   ├── logging.py
│   │   │   └── contrato de logs do runtime.
│   │   ├── models.py
│   │   │   └── registro de modelos e automl.
│   │   ├── pipeline.py
│   │   │   └── configuração de preprocessamento e engenharia de features.
│   │   └── settings.py
│   │       └── objeto central de settings construído pelo loader yaml.
│   ├── core/
│   │   ├── constants.py
│   │   │   └── caminhos e nomes do projeto.
│   │   ├── enums/
│   │   │   └── enums de validação e categorias de regras.
│   │   ├── exceptions/
│   │   │   └── base de erro e classes de comunicação de erro do domínio.
│   │   └── utils/
│   │       ├── datetime.py
│   │       │   └── conversão de data e número serial excel.
│   │       └── output_manager.py
│   │           └── roteamento de diretórios de execução e artefatos.
│   ├── data_loader/
│   │   ├── discovery.py
│   │   │   └── identificação de arquivos com extensão segura e supportada.
│   │   └── loader.py
│   │       └── carregamento e consolidação dos datasets.
│   ├── data_sources/
│   │   ├── base.py
│   │   │   └── base abstrata para leitores de dados.
│   │   ├── csv_source.py
│   │   │   └── ligação csv para o loader de dados.
│   │   ├── excel_source.py
│   │   │   └── driver de leitura compatível com xlsx, xls e xlsb pelo conteúdo.
│   │   ├── factory.py
│   │   │   └── fábrica de fontes de leitura.
│   │   ├── parquet_source.py
│   │   │   └── leitura de parquet.
│   │   ├── registry.py
│   │   │   └── registro das fontes suportadas.
│   │   ├── sql_source.py
│   │   │   └── leitura de dados via sql.
│   │   └── writer.py
│   │       └── exportação de dataframe em csv/parquet/excel.
│   ├── data_validation/
│   │   ├── base/
│   │   │   └── regras e máscara de validação.
│   │   ├── models.py
│   │   │   └── modelos e estruturas de resposta.
│   │   ├── report.py
│   │   │   └── relatório consolidado da validação.
│   │   ├── rules/
│   │   │   └── regras de duplicidade, datas futuras, zero volume, zero valor, retorno positivo, clientes nulos e produtos nulos.
│   │   └── validator.py
│   │       └── executor principal das regras de qualidade.
│   ├── export/
│   │   └── módulos de serialização de forecast, modelos e relatórios.
│   ├── feature_engineering/
│   │   └── temporal.py
│   │       └── geração de data, holiday, lag e resources temporais.
│   ├── feature_selection/
│   │   └── seleção das features de entrada.
│   ├── ml/
│   │   ├── evaluation/
│   │   │   └── métricas de avaliação e comparação de modelos.
│   │   ├── forecast/
│   │   │   └── previsão final e orquestração do forecaster.
│   │   ├── models/
│   │   │   └── baseline, linear, random forest, lightgbm, catboost, xgboost, sarima, prophet e adapters de features.
│   │   ├── monitoring/
│   │   │   └── monitoramento de modelos e observabilidade.
│   │   └── training/
│   │       └── treinamento, split temporal, seleção e validação cruzada.
│   ├── pipelines/
│   │   ├── train_pipeline.py
│   │   │   └── pipeline de preparação, treino e exportação.
│   │   ├── evaluate_pipeline.py
│   │   │   └── avaliação post-treino.
│   │   └── predict_pipeline.py
│   │       └── execução de previsão e produção de artefatos.
│   ├── preprocessing/
│   │   ├── cleaning.py
│   │   │   └── limpeza de linhas, colunas e strings.
│   │   ├── datetime.py
│   │   │   └── criação de features temporais de calendário.
│   │   ├── missing.py
│   │   │   └── estratégia de preenchimento e ausência de valor.
│   │   ├── typing.py
│   │   │   └── conversão e coerção dos tipos de coluna.
│   │   └── preprocessor.py
│   │       └── fluxo principal de preprocessamento.
│   ├── schemas/
│   │   └── sales_schema.py
│   │       └── schema de vendas derivado do contrato de configuração.
│   └── visualization/
│       └── análise exploratória, importância de feature e gráficos de forecast.
├── tests/
│   └── suíte de cobertura para validade do contrato, módulo de dados e pipelines.
├── forecasting-platform-ops/
│   └── scripts operacionais de smoke e readiness.
└── docs/
    └── documentação principal de regras e arquitetura.

resumo de arquivos centrais

main.py: orquestra o pipeline de produção e liga o fluxo de preprocessing, treinamento, avaliação e exportação.
app.py: oferece o dashboard streamlit de exploração visual e entrada de consultas.
configs/data.yaml: fonte oficial do contrato de entrada e schema runtime.
configs/holidays.yaml: calendário e regras de feriado.
src/config/settings.py: carregamento do objeto settings.
src/data_sources/excel_source.py: solução de leitura de excel e xlsb compatível.
src/feature_engineering/temporal.py: produção de features temporais e calendário.
src/ml/models/baseline.py: modelo de referência e previsão ingênua.
src/ml/models/prophet.py: embrulho do runtime prophet para produção.
src/pipelines/train_pipeline.py: fluxo de treinamento e serialização de modelos.

manutenção

- mantenha o contrato yaml como fonte de verdade para o projeto;
- faça mudanças de schema em data.yaml, e reflita isso em training, validation e modelos;
- preserve o fluxo de features e split temporal; e
- mantenha a documentação e os testes compatíveis com a mesma política de entrada.
