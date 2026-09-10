# DOCUMENTAÇÃO DE MELHORIAS FUTURAS - DATA ATUALIZAÇÃO: 09/09/2026

este documento consolida as principais melhorias identificadas durante a revisão
técnica do projeto e define uma ordem de evolução para transformar a plataforma
de forecasting em um fluxo operacional, confiável, reproduzível e auditável.

a prioridade atual não é adicionar novos modelos, mas garantir que os modelos
existentes sejam utilizados dentro de um processo temporalmente correto e
compatível com um cenário real de previsão de demanda.

## VISÃO GERAL

a arquitetura atual já possui os principais componentes necessários para
ingestão, validação, preprocessamento, feature engineering, treinamento,
avaliação, forecast, exportação e visualização.

a próxima etapa de evolução deve concentrar-se na integração correta desses
componentes e na validação do fluxo operacional.

os principais pontos de evolução são:

- garantir separação temporal correta entre treino, validação e teste;
- evitar vazamento de informações futuras;
- trabalhar com dados agregados e deduplicados;
- implementar previsão para períodos realmente futuros;
- selecionar o melhor modelo de forma consistente;
- aumentar a rastreabilidade das execuções;
- preparar a plataforma para maior volume de dados e operação contínua.

## PRIORIDADE P-0

1. GARANTIR BACKTESTING TEMPORAL CORRETO:

o processo de avaliação deve respeitar a ordem temporal dos dados.

a divisão atual baseada em linhas deve ser revisada para garantir que períodos
completos sejam utilizados em treino, validação e teste.

exemplo:

```
treino:
    até dezembro/2024

validação:
    janeiro/2025 a março/2025

teste:
    abril/2025 a junho/2025
```

a evolução posterior deve implementar rolling-origin backtesting para permitir
a avaliação do modelo em múltiplos períodos de corte.

critério esperado:

- nenhuma observação futura pode participar do treinamento de um período anterior;
- cada fold deve respeitar o cutoff temporal;
- as métricas devem ser comparáveis entre os períodos avaliados.

2. EVITAR VAZAMENTO DE DADOS:

toda feature utilizada durante o treinamento deve estar disponível no momento
em que a previsão seria realizada.

o problema não se limita ao uso de VALOR como feature para previsão de VOLUME.
qualquer informação futura indisponível no momento do forecast pode gerar
vazamento.

devem ser revisados:

- medianas e encoders calculados somente com dados de treino;
- rolling features calculadas somente até o cutoff;
- shares e agregações dependentes de períodos futuros;
- features derivadas do próprio target;
- variáveis que não estarão disponíveis no horizonte futuro;
- dependências entre VOLUME e VALOR;
- disponibilidade temporal de cada feature.

princípio:

```
uma feature só pode ser utilizada se estiver disponível no momento em que a previsão é realizada.
```

3. IMPLEMENTAR FORECAST FUTURO EXPLÍCITO:

a avaliação sobre o conjunto de teste representa uma etapa estatística de
validação e não deve ser confundida com o forecast operacional.

a plataforma deve possuir um fluxo específico para gerar previsões futuras.

a execução deve definir explicitamente:

- data de corte;
- horizonte de previsão;
- datas futuras;
- entidades previstas;
- features futuras disponíveis;
- modelo utilizado;
- target;
- previsão gerada.

exemplo:

```
cutoff:
    2026-08-31

horizonte:
    30 dias

previsão:
    2026-09-01 até 2026-09-30
```

o arquivo final deve conter, no mínimo:

```
DATA
dimensões da série
TARGET
MODELO
PREVISAO
```

4. IMPLEMENTAR SELEÇÃO REAL DE MODELOS:

a seleção de modelos deve separar claramente baseline, candidatos, avaliação e promoção.

fluxo esperado:

```
baseline
    +
modelos candidatos
    ↓
mesmo protocolo de backtesting
    ↓
avaliação
    ↓
ranking
    ↓
modelo vencedor
    ↓
promoção
```

o baseline deve ser utilizado como referência obrigatória.

a configuração de AutoML não deve ser confundida com seleção de modelos.
mesmo com AutoML desativado, os modelos habilitados podem ser avaliados e comparados de forma determinística.

a seleção deve considerar métricas previamente definidas e critérios de
estabilidade e viés.

5. INTEGRAR O FLUXO DECLARATIVO COM A EXECUÇÃO:

o arquivo `configs/pipeline.yaml` deve representar corretamente as etapas habilitadas na execução da plataforma.

o `main.py` deve atuar como orquestrador, conectando os componentes sem duplicar suas responsabilidades.

a integração deve contemplar, conforme habilitação da configuração:

- DataValidator;
- SchemaValidator;
- Aggregator;
- FeatureEngineeringPipeline;
- Profiler;
- Explainer;
- ModelMonitor;
- demais componentes previstos pela arquitetura.

as responsabilidades internas devem permanecer nos componentes especializados.

princípio:

```
pipelines orquestram o fluxo;
componentes implementam suas responsabilidades;
configurações definem comportamento;
regras de negócio não devem ser duplicadas no orquestrador.
```

6. IMPLEMENTAR SUPORTE EXPLÍCITO A MÚLTIPLOS TARGETS:

o arquivo `configs/forecast.yaml` pode definir mais de um target, como:

```
VOLUME
VALOR
```

a execução deve permitir selecionar explicitamente o target utilizado.

Exemplo:

```
python main.py --target VOLUME

python main.py --target VALOR
```

inicialmente, cada target deve possuir uma execução independente, simplificando treinamento, avaliação, seleção e rastreabilidade.

cada execução deve registrar o target utilizado nos artefatos da run.

## PRIORIDADE P-1

7. AGREGAR OS DADOS ANTES DO TREINAMENTO:

as bases possuem aproximadamente 4,7 milhões de registros transacionais.

O treinamento deve trabalhar com uma granularidade compatível com o problema de previsão, reduzindo o volume de dados e transformando as transações em séries temporais coerentes.

uma possível granularidade é:

```
DATA + COD CLIENTE + COD ITEM
```

com:

```
VOLUME = soma
VALOR  = soma
```

a granularidade definitiva deve ser definida de acordo com o objetivo do forecast e com a disponibilidade das dimensões necessárias.

modelos de séries temporais, como Prophet e SARIMA, devem receber séries agregadas por entidade e frequência compatível com suas premissas.

8. RESOLVER DUPLICIDADES ENTRE ARQUIVOS:

arquivos de períodos diferentes podem possuir sobreposição temporal.

a deduplicação deve considerar a chave de negócio e não somente linhas fisicamente idênticas.

o processo deve:

- registrar o arquivo de origem;
- calcular hash dos arquivos;
- identificar o intervalo temporal;
- definir a chave de negócio;
- identificar duplicidades físicas;
- identificar duplicidades lógicas;
- remover registros duplicados conforme regra definida;
- registrar a quantidade de linhas removidas.

exemplo de possível chave:

```
DATA + COD CLIENTE + COD ITEM
```

a chave definitiva deve ser definida conforme a granularidade utilizada pelo pipeline.

9. IMPLEMENTAR BASELINE POR GRUPO:

o baseline atual pode evoluir de uma referência global para uma estratégia por entidade.

exemplo:

```
Cliente A + Produto X
    último valor conhecido = 120
    previsão = 120
```

uma implementação possível utiliza:

```
date_column="DATA"
target_column="VOLUME"
group_columns=["COD CLIENTE", "COD ITEM"]
```

o baseline deve servir como referência para determinar se os modelos
mais complexos realmente agregam valor.

10. REVISAR O USO DE PROPHET E SARIMA:

Prophet e SARIMA não devem operar diretamente sobre a base transacional.

seu uso deve ser avaliado em séries que possuam:

- entidade bem definida;
- frequência temporal consistente;
- agregação adequada;
- tratamento de períodos sem venda;
- horizonte definido;
- quantidade de séries compatível com o custo computacional.

modelos tabulares como LightGBM, CatBoost e XGBoost devem ser avaliados como candidatos prioritários para o cenário atual, sem estabelecer previamente qual modelo será superior.

11. EVOLUIR O CONJUNTO DE MÉTRICAS:

MAPE apresenta limitações quando existem valores iguais a zero.

a avaliação deve evoluir para um conjunto de métricas complementares:

- MAE;
- RMSE;
- WAPE;
- sMAPE;
- viés médio;
- métricas por produto;
- métricas por cliente;
- métricas por horizonte.

a escolha do modelo não deve depender obrigatoriamente de uma única métrica.

o critério de seleção deve considerar erro, viés e estabilidade de acordocom o objetivo do forecast.

## PRIORIDADE P-2

12. MELHORAR A ESCALABILIDADE:

o volume atual exige atenção ao consumo de memória e ao tempo de processamento.

possíveis evoluções:

- converter XLSB para Parquet;
- particionar dados por ano e mês;
- reduzir colunas antes da concatenação;
- evitar cópias desnecessárias;
- utilizar tipos adequados, incluindo categorias quando aplicável;
- processar dados em lotes quando necessário;
- medir tempo de execução por etapa;
- medir consumo máximo de memória;
- avaliar operações que materializam grandes estruturas;
- evitar `apply(axis=1)` em operações que possam ser vetorizadas.

a otimização deve ser baseada em medições e benchmarks, evitando otimizações prematuras.

13. AVALIAR CAMADA BRONZE / SILVER / GOLD:

conforme o volume e a frequência de execução aumentarem, pode ser avaliada uma separação entre diferentes níveis de processamento.

```
Bronze:
    arquivos originais

Silver:
    dados consolidados, tipados e deduplicados

Gold:
    séries agregadas e features prontas para modelagem
```

essa arquitetura deve ser adotada somente se houver benefício operacional suficiente para justificar sua complexidade.

14. MELHORAR OS ARTEFATOS DAS RUNS:

o `metadata.json` deve registrar informações suficientes para reproduzir e auditar uma execução.

informações esperadas:

- `run_id`;
- timestamp da execução;
- arquivos utilizados;
- hash dos arquivos;
- período carregado;
- número de linhas carregadas;
- número de linhas removidas;
- schema;
- target;
- features;
- modelo vencedor;
- candidatos avaliados;
- parâmetros;
- seed;
- cutoff;
- horizonte;
- métricas;
- versão do código;
- versão das configurações.

a rastreabilidade deve permitir responder:

```
quais dados foram utilizados?
qual configuração foi aplicada?
qual modelo foi selecionado?
quais métricas foram obtidas?
qual período foi utilizado?
qual código gerou o resultado?
```

15. INTEGRAR MLOPS:

as dependências de MLflow e Evidently podem ser utilizadas para evoluir a rastreabilidade e o monitoramento da plataforma.

possíveis evoluções:

- tracking de experimentos no MLflow;
- versionamento de modelos;
- comparação entre runs;
- registro de parâmetros;
- registro de métricas;
- promoção de modelos;
- monitoramento de drift;
- monitoramento de degradação;
- alertas;
- governança de modelos em produção.

a integração deve ocorrer após a estabilização do fluxo de treinamento, avaliação e forecast.

16. EVOLUIR O STREAMLIT:

o dashboard deve atuar como camada de consumo dos resultados produzidos pela plataforma.

possíveis funcionalidades:

- seleção do dataset;
- seleção do target;
- comparação de modelos;
- visualização do backtesting;
- visualização do forecast futuro;
- comparação entre observado e previsto;
- visualização de métricas;
- visualização de drift;
- identificação do modelo promovido;
- consulta aos artefatos das runs.

o dashboard não deve duplicar regras de negócio existentes no pipeline.

## ORDEM RECOMENDADA:

a evolução recomendada é:

1. corrigir o split temporal e implementar backtesting;
2. eliminar vazamentos de dados;
3. agregar e deduplicar os dados;
4. implementar forecast futuro explícito;
5. implementar seleção real de modelos;
6. implementar baseline por cliente/produto;
7. implementar suporte explícito a múltiplos targets;
8. evoluir métricas e critérios de seleção;
9. otimizar escalabilidade e avaliar migração para Parquet;
10. melhorar artefatos e rastreabilidade das runs;
11. integrar MLflow e Evidently;
12. evoluir o Streamlit.

a integração das etapas declaradas em `configs/pipeline.yaml` deve ocorrer de forma transversal durante essa evolução, mantendo o `main.py` como orquestrador e os componentes especializados responsáveis por suas próprias funções.

## FLUXO OPERACIONAL IDEAL:

o fluxo final esperado é:

```
dados brutos
    ↓
schema e qualidade
    ↓
deduplicação
    ↓
agregação temporal
    ↓
features disponíveis no futuro
    ↓
backtesting temporal
    ↓
baseline + candidatos
    ↓
avaliação
    ↓
seleção do modelo
    ↓
treinamento final
    ↓
forecast futuro
    ↓
artefatos auditáveis
    ↓
monitoramento
    ↓
dashboard
```

## PRINCÍPIOS DE EVOLUÇÃO:

as melhorias futuras devem preservar os princípios arquiteturais definidos para a plataforma:

- cada componente deve possuir uma responsabilidade clara;
- configurações devem permanecer separadas da lógica de processamento;
- pipelines devem orquestrar, não implementar regras de negócio;
- modelos devem respeitar o contrato definido pela camada de modelos;
- validações devem ocorrer antes das etapas que dependem dos dados validados;
- features devem respeitar a disponibilidade temporal;
- avaliação deve respeitar a ordem cronológica dos dados;
- artefatos devem permitir rastreabilidade das execuções;
- novas funcionalidades devem ser implementadas incrementalmente;
- cada evolução deve possuir testes correspondentes;
- otimizações de desempenho devem ser baseadas em medições;
- novas abstrações devem ser introduzidas somente quando houver necessidade real.

o objetivo final não é apenas aumentar a quantidade de modelos disponíveis, mas construir uma plataforma capaz de produzir previsões futuras de forma consistente, reproduzível, mensurável e auditável.
