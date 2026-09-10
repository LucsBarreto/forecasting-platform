# Desenvolvimento do Dashboard Streamlit — Inteligência Comercial

Desenvolva a interface Streamlit da plataforma forecasting-platform, seguindo as decisões de arquitetura e layout descritas abaixo.

## 1. Objetivo

Criar um dashboard de Inteligência Comercial para exploração dos dados históricos de vendas e análise das previsões geradas pela plataforma de forecasting.

O Streamlit deve atuar como camada de visualização e consumo dos resultados, evitando duplicar regras de negócio, processamento, treinamento ou forecasting que pertencem ao backend.

A interface deve ser profissional, limpa, intuitiva e adequada para apresentação de um projeto de Data Science/Analytics.

## 2. Layout geral

Utilizar layout amplo (`wide`).

A aplicação deve ser dividida visualmente em:

```text
┌──────────────────────┬──────────────────────────────────────────────┐
│                      │                                              │
│   FILTROS GLOBAIS    │             INTELIGÊNCIA COMERCIAL          │
│      SIDEBAR         │                                              │
│                      │                 VISÃO GERAL                  │
│                      │                                              │
│                      │                 ANÁLISE HISTÓRICA             │
│                      │                                              │
│                      │                 PREVISÕES                    │
│                      │                                              │
│                      │                 MODELOS / MÉTRICAS            │
│                      │                                              │
│                      │                 EXECUÇÕES                     │
│                      │                                              │
└──────────────────────┴──────────────────────────────────────────────┘
```

Os filtros globais devem ficar na barra lateral esquerda (`st.sidebar`).

A sidebar deve ser colapsável, permitindo que o usuário esconda os filtros e utilize praticamente toda a largura da tela para os gráficos.

## 3. Filtros globais

A sidebar deve conter os filtros que afetam a exploração dos dados históricos.

Organizar os filtros por categorias para evitar uma lista visualmente confusa.

### Período

Disponibilizar:

- Todo o período
- Últimos 12 meses
- Últimos 90 dias
- Período personalizado

### Localização

Filtros para:

- Filial
- Regional
- UF
- Cliente / Produto

### Cliente / Produto

Filtros para:

- Cliente
- Produto
- Marca
- Categoria

### Comercial

Filtros para:

- Canal
- Subcanal
- Atendimento

Os filtros devem ser aplicados aos dados históricos e refletidos nos KPIs e gráficos históricos.

Não utilizar `VOLUME` ou `VALOR` como filtros de dimensão.

`VOLUME` e `VALOR` devem ser tratados como métricas.

## 4. Visão Geral

Criar uma seção de visão geral com cards contendo:

### Card 1 — Volume total

Mostrar o volume total do período selecionado.

### Card 2 — Valor total

Mostrar o valor total do período selecionado.

### Card 3 — Preço médio

Calcular:

`Preço médio = VALOR total / VOLUME total`

Não utilizar uma média simples dos preços individuais.

### Card 4 — Clientes

Mostrar a quantidade total de clientes no período/filtros selecionados.

Os cards devem reagir aos filtros globais.

## 5. Análise histórica

Criar uma seção específica para análise dos dados históricos.

### Seleção da dimensão

Disponibilizar um seletor para escolher a dimensão de agrupamento.

Exemplos:

- Data
- Filial
- Regional
- UF
- Cliente
- Produto
- Marca
- Categoria
- Canal
- Subcanal
- Atendimento

Não incluir as métricas `VOLUME` e `VALOR` como dimensões.

### Seleção de métricas

Disponibilizar:

- Volume
- Valor
- Volume + Valor

O usuário deve poder escolher se deseja visualizar:

- somente volume;
- somente valor;
- volume e valor simultaneamente.

Quando as duas métricas forem selecionadas, considerar escalas diferentes e utilizar uma representação que permita a interpretação adequada das duas grandezas.

### Gráfico

Criar um gráfico de barras verticais mostrando o resultado agrupado pela dimensão selecionada.

Exemplos:

- Dimensão = Produto
- Métrica = Valor

ou:

- Dimensão = Regional
- Métrica = Volume + Valor

O gráfico deve respeitar todos os filtros globais.

### Evolução temporal

Além do gráfico de barras por dimensão, disponibilizar uma visualização temporal para acompanhar a evolução dos indicadores.

Permitir seleção de frequência, quando aplicável:

- Diário
- Semanal
- Mensal

## 6. Execuções do forecasting

Criar uma seção que disponibilize uma lista das execuções anteriores do aplicativo/forecasting.

As execuções devem ser obtidas a partir dos artefatos existentes em:

`outputs/runs/`

Cada execução deve apresentar, quando essas informações estiverem disponíveis:

- identificador da execução;
- data/hora;
- target;
- horizonte;
- granularidade;
- modelos avaliados;
- status.

### Exemplo visual

```text
RUN_20260910_143522
10/09/2026 14:35
Target: VOLUME
Horizonte: 6 meses
Status: concluída
```

O usuário deve poder selecionar uma execução anterior e revisitar seus resultados.

A execução selecionada deve determinar quais artefatos de forecast, métricas e modelos serão apresentados.

## 7. Previsões

Criar uma seção específica para análise das previsões.

Disponibilizar os seguintes controles:

- Execução
- Dimensão

### Execução

```text
[ RUN_20260910_143522 ▼ ]
```

### Dimensão

Permitir selecionar a dimensão de análise, seguindo as mesmas dimensões disponíveis para os dados históricos.

Exemplos:

- Filial
- Regional
- UF

## Atualizações recentes

- A tabela de comparação de execução passou a mostrar a métrica real configurada (ex.: MAPE) em vez de um rótulo genérico.
- O ranking de modelos agora é ordenado pelo melhor desempenho conforme a métrica: menor valor para métricas de erro e maior valor para métricas de benefício.
- O script `clear_cache.py` foi adicionado na raiz para limpar caches do projeto com tolerância a blocos do Windows/OneDrive.
