# Case Técnico — Especialista de Dados I | E-commerce

Previsão de receita diária do e-commerce (Seção 1) e proposta de arquitetura de IA para análise de banners (Seção 2).

## Resumo executivo

**O que os dados mostram (nov/2025 a jun/2026, R$ 490 mi de receita)**
- **Novembro é outro negócio:** 26% da receita do período em um único mês promocional. Depois dele, não há tendência de longo prazo; o nível varia com as datas de presente. O Natal vende o mês (semanas de R$ 13–16 mi em dezembro), não o dia 25, que é o pior da base.
- **Sexta é o melhor dia, domingo o pior** (-18% no App, -27% no Site). O App segura melhor o fim de semana e já responde por 79% da receita.
- **Pico de vendas às 11h** nos dois canais; o App tem um segundo pico às 20h-21h.
- **Menos desconto, ticket maior:** o ticket médio sobe de R$ 59 (nov) para R$ 93 (jun) enquanto o desconto cai de 55% para 35% do preço cheio.
- **O salário move a venda:** dias 5 a 7 vendem ~22% acima do esperado; o adiantamento (dia 20) não tem efeito.
- **Estornos estruturais** de ~4% da receita todo mês, independentes de campanha.

**Como o consumidor reage (associações controladas por calendário, não causalidade)**
- **Desconto compra volume, não valor:** cada ponto a mais de desconto vem com +2,8% de pedidos, −1,9% no preço por item e só +1,6% de receita. Se compensa depende da margem, que a base não tem.
- **Promoção é evento de perfume:** em dias de desconto alto, perfumaria vai de 46% para 60% da receita; Gifts cai de 14% para 7% (presente vende por ocasião, não por preço).
- **O presente tem destinatário:** antes do Dia das Mães, a perfumaria feminina sobe para ~25% da receita; antes do Dia dos Namorados, a masculina chega a 34%–42%. A curva acelera cerca de uma semana antes.
- **O ticket sobe por dois motivos:** menos desconto e mix mais caro (preço cheio por item de R$ 80 para R$ 93).

**Previsão de 7 dias à frente**
- Cinco modelos competiram nas mesmas condições, com junho guardado como teste cego.
- **Recomendação: LightGBM**, que erra cerca de 20% da receita diária em junho (R$ 318 mil por dia), contra 31% de uma média móvel simples e 35% de repetir a semana anterior.
- A vantagem aparece quando **a venda muda de patamar** de uma semana para outra (datas e ressacas): nessas semanas o LightGBM erra ~26% e a média móvel ~37% (semana do Dia dos Namorados: 26% × 43%; semana seguinte: 19% × 54%). Em semanas estáveis, a média móvel erra menos, por isso ela fica como régua de monitoramento e gatilho de reversão.
- XGBoost (18,6%) e LightGBM empatam tecnicamente; o LightGBM era o candidato definido antes do teste e foi mantido.
- Com só 5 semanas de teste, nenhuma diferença contra a média móvel é estatisticamente conclusiva. A regra fixada antes, aplicada literalmente, aponta a média móvel (`reports/metrics/decisao.json`); a decisão final e o critério estão em `docs/decisoes_tecnicas.md` (D07).
- **Planejar pela semana:** o erro do total semanal é de 12% (contra 20% no dia), e o modelo elimina 43% do erro de repetir a semana anterior. Uma faixa P10–P90 ajustada por conformal cobre 90% dos dias de junho (meta de 80%: é conservadora, ≈ ±50% do previsto) e explicita a incerteza diária (D10).

**Seção 2: scoring de banners**
- Posição: não seguir com o projeto como está. Primeiro definir a decisão de negócio e o KPI (recomendado: receita por impressão, não CTR), depois PoC e MVP com teste A/B.
- A ferramenta no-code não homologada é tratada como questão de compliance, com trilha formal junto a Segurança da Informação e Jurídico; o time do hackathon segue dono do problema e começa já a rotular banners e definir o KPI.
- Custo × retorno sobre margem, com custo recorrente e três cenários (payback de ~12 meses no cenário base; perda limitada ao MVP no pessimista). Como 0,1% da receita não é mensurável em A/B, o teste usa um KPI intermediário detectável com a receita por impressão como guarda.
- Demo com CLIP em banners sintéticos: o embedding captura cor e tipo de mensagem; com 24 exemplos rotulados, um classificador simples acerta 100% da mensagem; detalhes pequenos (preço) pedem OCR.

## Como ler este repositório em 5 minutos

1. Este README (resumo executivo acima).
2. `notebooks/03_modelagem.ipynb`: a decisão do modelo, da pergunta à recomendação.
3. `notebooks/02_comportamento_consumo.ipynb`: o que o dado sugere para o negócio.
4. `docs/decisoes_tecnicas.md`: cada escolha com contexto, decisão e consequência.
5. `docs/secao2_arquitetura_ia.md`: a proposta da Seção 2.

## Estrutura

```
notebooks/
  01_eda.ipynb              análise exploratória e insights
  02_comportamento_consumo.ipynb  reação a desconto, mix, datas de presente e salário
  03_modelagem.ipynb        torneio de modelos, validação, interpretação e faixa de previsão
  04_demo_clip.ipynb        demo de embeddings de imagem (Seção 2)
src/
  data_loader.py            leitura, limpeza e agregação
  eda.py                    análises e figuras da EDA
  consumo.py                leitura de comportamento de consumo
  calendario.py             feriados e datas comerciais
  features.py               variáveis do modelo (calendário, eventos, histórico)
  models.py                 baselines, Ridge, XGBoost, LightGBM
  train.py                  walk-forward, holdout, contraprova e decisão
  evaluate.py               métricas e bootstrap por semana
  explain.py, visual.py     SHAP, importância e visualizações
  intervalos.py             ganho sobre a régua, erro semanal e faixa de previsão calibrada
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
.venv/Scripts/python -m src.intervalos # métricas complementares e faixa de previsão
```

**Google Colab:** cada notebook tem uma célula inicial de preparação. Clone o repositório, envie o CSV para `data/raw/` e execute. O notebook 04 usa GPU automaticamente quando disponível.

## Como o modelo foi validado

- **Sem split aleatório:** treino sempre no passado e teste na semana seguinte, avançando semana a semana (walk-forward). Junho ficou isolado como teste final.
- **Horizonte de 7 dias:** só entram variáveis conhecidas com uma semana de antecedência (calendário, datas comerciais, histórico de 7 dias atrás).
- **Teste de vazamento:** corromper todos os dados posteriores à data da previsão não altera nenhuma variável, em todas as 27 semanas previstas.
- **Comparação justa:** mesmas variáveis, mesmas semanas e mesmo orçamento de ajuste para todos os modelos; intervalos de confiança reamostrando semanas inteiras.
- **Revisão independente** do pipeline antes da rodada final (D09).
- **Limitações de validação** declaradas em D12 (semanas de ajuste intercaladas, 5 semanas de teste).

## Limitações

- 8 meses de histórico: cada data especial aparece uma vez; a sazonalidade anual não é observável.
- Base agregada: sem cliente, SKU, margem, estoque, mídia ou calendário de campanhas.
- 6% da receita tem categoria ilegível na origem (tratada como "Não mapeada").
- Junho teve Copa do Mundo, fora da base.
- Datas que o modelo nunca viu (Dia dos Pais, nova Black Friday) devem ser acompanhadas contra a média móvel.
- As leituras de consumo são associações: dias de desconto também têm campanha e mídia.
