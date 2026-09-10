# documentação: calendário de feriados e datas relevantes

**data atualização:** 09/09/2026

o projeto utiliza YAML como formato principal para configuração de feriados e datas relevantes usadas na engenharia de features temporais e comerciais.

**arquivo de configuração:**

`configs/holidays.yaml`

---

## 1. estrutura

o calendário é organizado por categoria:

* `national`:
  feriados nacionais oficiais.

* `regional`:
  feriados estaduais e municipais.

* `commercial`:
  datas com potencial impacto comercial nas vendas.

* `observance`:
  datas comemorativas ou de conscientização.

---

## 2. datas fixas

datas fixas utilizam os campos:

* `name`:
  nome do evento.

* `month`:
  mês do evento.

* `day`:
  dia do evento.

**exemplo:**

```yaml
- name: "Ano Novo"
- month: 1
- day: 1
```

---

## 3. datas móveis

datas móveis utilizam uma data-base e um deslocamento.

* `base`:
  base de cálculo da data.

* `offset_days`:
  deslocamento em dias a partir da base.

**exemplo:**

```yaml
- name: "Sexta-feira Santa"
- base: "easter"
- offset_days: -2
```

O valor `"easter"` representa a data da páscoa.

---

## 4. regras de calendário

eventos que não podem ser representados apenas por `month/day` utilizam o campo `rule`.

**exemplo:**

```yaml
- name: "Dia das Mães"
- rule: "second_sunday_of_may"
```

as regras são interpretadas pelo componente responsável pelo calendário antes da geração das features.

---

## 5. feriados regionais

feriados estaduais são definidos por UF:

```yaml
- name: "Data Magna do Ceará"
- state: "CE"
- month: 3
- day: 19
```

feriados municipais são definidos por UF e município:

```yaml
- name: "Aniversário de Fortaleza"
- state: "CE"
- city: "Fortaleza"
- month: 4
- day: 13
```

a aplicação de um feriado municipal depende da localização da filial.

Um evento configurado para Fortaleza não deve ser aplicado a uma filial localizada em outro município.

---

## 6. normalização

o calendário configurado em YAML é convertido para datas reais durante o processamento da janela temporal utilizada pelo pipeline.

datas fixas são calculadas diretamente a partir de `month/day`.

datas móveis são calculadas a partir da base configurada e de `offset_days`.

regras específicas são resolvidas pelo mecanismo de calendário.

---

## 7. manutenção

novos eventos devem seguir a estrutura existente e utilizar apenas os campos necessários para representar sua regra.

**campos principais:**

* `name`
* `month`
* `day`
* `base`
* `offset_days`
* `rule`
* `state`
* `city`

a categoria do evento é determinada pela seção do YAML e não deve ser repetida dentro de cada registro.

a lógica de cálculo das datas deve permanecer no código.

O YAML deve conter apenas as configurações e regras necessárias para representar o calendário.
