# Documentação do `MetricsSchema`

## Auditoria do contrato anterior

A API lia `outputs/runs/<run_id>/metrics/metrics.json` com `json.load` e retornava o conteúdo dentro de `{"metrics": ...}`. O teste existente usava um mapa plano com métricas como `mae` e `rmse`.

O projeto também possui o enum `MetricType`, que define as métricas oficiais:

```text
mae, rmse, mape, smape, r2
```

Não foi localizado um produtor persistente explícito de `metrics.json` no código atual. O contrato foi formalizado no ponto de consumo, sem criar um novo mecanismo de persistência ou mover a lógica de avaliação.

## Estrutura oficial

O arquivo persistido passa a ser validado como:

```json
{
  "run_id": "RUN_20260911_090311",
  "target": "VOLUME",
  "model": "LinearModel",
  "metrics": {
    "mae": 12.3,
    "rmse": 17.4,
    "mape": 0.12
  }
}
```

### Campos

- `run_id`: obrigatório, string no formato `RUN_YYYYMMDD_HHMMSS`.
- `target`: obrigatório, somente `VOLUME` ou `VALOR`.
- `model`: obrigatório, string não vazia.
- `metrics`: obrigatório, objeto não vazio com nomes oficiais e valores numéricos finitos.

As chaves permitidas em `metrics` são as já existentes no projeto: `mae`, `rmse`, `mape`, `smape` e `r2`. Nenhuma métrica nova foi adicionada.

## Integração

`MetricsService` agora executa:

```text
metrics.json
    ↓
MetricsSchema
    ↓
validação
    ↓
{"metrics": schema.metrics}
    ↓
JSON HTTP
```

A resposta HTTP existente foi preservada: a API continua devolvendo apenas o mapa interno em `metrics`, sem expor obrigatoriamente o envelope de persistência.

Se o JSON estiver malformado, possuir campos extras, métrica desconhecida, valor não numérico, `NaN`, `Infinity` ou `run_id` inconsistente com a URL, o serviço retorna `422` sem traceback ou caminho físico.

## Responsabilidades preservadas

- avaliação/ML produz os valores;
- persistência grava `metrics.json`;
- `MetricsSchema` valida o contrato;
- `MetricsService` consulta e normaliza a resposta;
- a API não recalcula métricas;
- nenhuma rota, pipeline ou componente de forecast foi alterado.

## Testes

Foram adicionados testes para:

- payload válido;
- campos obrigatórios ausentes;
- `run_id` inválido;
- targets `VOLUME` e `VALOR`;
- target inválido;
- modelo vazio ou com tipo inválido;
- métricas vazias, desconhecidas, aninhadas ou não numéricas;
- `NaN` e `Infinity`;
- JSON persistido inválido;
- inconsistência de `run_id` entre artefato e URL.
