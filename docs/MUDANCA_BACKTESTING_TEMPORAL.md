# Mudança de contrato — Backtesting temporal

## Data
2026-09-14

## Objetivo
Documentar a primeira alteração formal de desenho para o item 2 do projeto: introduzir um contrato técnico de `Backtester` e manter `TemporalSplitter` como responsável pelo split único de treino/validação/teste.

## Alterações realizadas

### Arquivos alterados
- `src/ml/training/__init__.py`
- `src/ml/training/backtesting.py` (novo arquivo)
- `tests/ml/training/test_backtesting.py` (novo arquivo)

### Resumo técnico
A implementação adiciona o desenho inicial do componente `Backtester` com as responsabilidades de:

- aceitar o bloco de entrada do contrato:
  - `dataframe`
  - `date_column`
  - `horizon`
  - `n_folds`
  - `min_training_history`
  - `gap`
  - `strategy = "expanding"`
- gerar folds de forma temporal e expandindo o treino por janela;
- separar a geração dos folds da execução e da agregação de resultados;
- manter o componente de split cronológico único em `TemporalSplitter` intacto;
- manter o teste final fora do ciclo de backtesting;
- registrar em um `BacktestResult` a estrutura de `folds`, `metrics_by_fold` e `aggregated_metrics`.

### Regras do contrato implementado
- padrão de folds: 3;
- horizonte padrão: 3 períodos;
- treino mínimo: 24 períodos;
- gap padrão: 0;
- estratégia: `expanding`.

### Comportamento previsto
Com o exemplo de entrada abaixo:

```text
min_training_history = 24
horizon = 3
n_folds = 3
```

o gerador deve construir folds com a seguinte expansão:

```text
24 train -> 3 validation
27 train -> 3 validation
30 train -> 3 validation
```

Ou seja, o treino cresce por janela e o horizonte de validação permanece fixo.

## Observações de arquitetura

- O `TemporalSplitter` continua responsável por dividir a base em treino/validação/teste de forma cronológica e única.
- O `Backtester` é um componente novo de avaliação temporal para múltiplos folds.
- A escolha de `strategy="expanding"` já está codificada como o primeiro contrato válido.
- Ainda não há suporte a `fixed`, conforme acordo técnico para esta primeira versão.

## Impacto futuro

A partir daqui, a próxima etapa deve mover a seleção de features para o interior de cada fold usando o treino correspondente apenas, para garantir que a seleção não veja a validação inteira antes da janela temporal. Isso é o principal ajuste de correção de vazamento de informação.
