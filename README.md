# Case Técnico — Especialista de Dados I | E-commerce

Previsão de receita diária do e-commerce (Seção 1) e proposta de arquitetura de IA para análise de banners (Seção 2).

## Resumo executivo

**O que os dados mostram (nov/2025 a jun/2026, R$ 490 mi de receita)**
- **Novembro é outro negócio:** 26% da receita do período em um único mês promocional. Fora dele, a venda é estável, com picos em datas de presente.
- **Sexta é o melhor dia, domingo o pior** (-18% no App, -27% no Site). O App segura melhor o fim de semana e já responde por 79% da receita.
- **Pico de vendas às 11h** nos dois canais; o App tem um segundo pico às 20h-21h.
- **Menos desconto, ticket maior:** o ticket médio sobe de R$ 59 (nov) para R$ 93 (jun) enquanto o desconto cai de 55% para 35% do preço cheio.
- **O salário move a venda:** dias 5 a 7 vendem ~22% acima do esperado; o adiantamento (dia 20) não tem efeito.
- **Estornos estruturais** de ~4% da receita todo mês, independentes de campanha.

**Previsão de 7 dias à frente**
- Cinco modelos competiram nas mesmas condições, com junho guardado como teste cego.
- **Recomendação: LightGBM**, que erra cerca de 20% da receita diária em junho (R$ 318 mil por dia), contra 31% de uma média móvel simples e 35% de repetir a semana anterior.
- A vantagem está nas **semanas com data especial**: na semana do Dia dos Namorados, a média móvel errou 54% e o LightGBM, 19%. Em semanas comuns, os dois empatam, por isso a média móvel fica como régua de monitoramento.
- Com só 5 semanas de teste, a diferença não é estatisticamente conclusiva; a escolha e o critério estão explícitos em `docs/decisoes_tecnicas.md` (D07).

**Seção 2: scoring de banners**
- Posição: não seguir com o projeto como está. Primeiro definir a decisão de negócio e o KPI (recomendado: receita por impressão, não CTR), depois PoC e MVP com teste A/B.
- A ferramenta no-code não homologada é tratada como questão de compliance, com trilha formal junto a Segurança da Informação e Jurídico.
- Demo com CLIP em banners sintéticos: o embedding captura cor e tipo de mensagem; com 24 exemplos rotulados, um classificador simples acerta 100% da mensagem; detalhes pequenos (preço) pedem OCR.

## Estrutura

```
notebooks/
  01_eda.ipynb              análise exploratória e insights
  03_modelagem.ipynb        torneio de modelos, validação e interpretação
  04_demo_clip.ipynb        demo de embeddings de imagem (Seção 2)
src/
  data_loader.py            leitura, limpeza e agregação
  eda.py                    análises e figuras da EDA
  calendario.py             feriados e datas comerciais
  features.py               variáveis do modelo (calendário, eventos, histórico)
  models.py                 baselines, Ridge, XGBoost, LightGBM
  train.py                  walk-forward, holdout, contraprova e decisão
  evaluate.py               métricas e bootstrap por semana
  explain.py, visual.py     SHAP, importância e visualizações
  clip_demo.py              banners sintéticos, CLIP, zero-shot e clusters
docs/
  decisoes_tecnicas.md      decisões técnicas (contexto, decisão, consequência)
  secao2_arquitetura_ia.md  business case da Seção 2
reports/
  figures/                  gráficos usados na apresentação
  metrics/                  métricas do torneio e da demo
tests/                      testes automatizados (inclui teste de vazamento temporal)
```

## Como reproduzir

O arquivo de dados do case não está no repositório. Coloque-o em `data/raw/vendas.csv`.

**Local (Python 3.10)**
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt   # Linux/Mac: .venv/bin/python
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -m src.train      # torneio completo
```

**Google Colab:** cada notebook tem uma célula inicial de preparação. Clone o repositório, envie o CSV para `data/raw/` e execute. O notebook 04 usa GPU automaticamente quando disponível.

## Como o modelo foi validado

- **Sem split aleatório:** treino sempre no passado e teste na semana seguinte, avançando semana a semana (walk-forward). Junho ficou isolado como teste final.
- **Horizonte de 7 dias:** só entram variáveis conhecidas com uma semana de antecedência (calendário, datas comerciais, histórico de 7 dias atrás).
- **Teste de vazamento:** corromper todos os dados posteriores à data da previsão não altera nenhuma variável, em todas as 27 semanas previstas.
- **Comparação justa:** mesmas variáveis, mesmas semanas e mesmo orçamento de ajuste para todos os modelos; intervalos de confiança reamostrando semanas inteiras.
- **Auditoria independente** do pipeline antes da rodada final (D09).

## Limitações

- 8 meses de histórico: cada data especial aparece uma vez; a sazonalidade anual não é observável.
- Base agregada: sem cliente, SKU, margem, estoque, mídia ou calendário de campanhas.
- 6% da receita tem categoria ilegível na origem (tratada como "Não mapeada").
- Junho teve Copa do Mundo, fora da base.
