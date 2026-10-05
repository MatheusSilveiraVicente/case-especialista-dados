# Decisões técnicas

Formato: contexto → decisão → consequência.

## D01 — Alvo do modelo
- Contexto: o enunciado pede "previsão de venda diária" sem definir a métrica.
- Decisão: receita aprovada líquida diária como alvo principal; pedidos como alvo secundário para separar efeito preço de efeito volume.
- Consequência: estornos (receita negativa) ficam dentro do alvo, refletindo o caixa real.

## D02 — Horizonte de 7 dias
- Contexto: previsão de amanhã (d+1) tem pouco uso para planejamento operacional.
- Decisão: prever 7 dias à frente; só lags >= 7 e variáveis de calendário conhecidas antecipadamente.
- Consequência: acurácia menor que d+1, mas utilizável para escala, estoque e campanhas.

## D03 — Categorias corrompidas
- Contexto: 5.341 linhas (6% da receita) têm categoria = número aleatório, irrecuperável.
- Decisão: agrupar como "Nao mapeada"; manter no total, excluir das análises por categoria.
- Consequência: totais corretos; análise por categoria cobre 94% da receita.

## D04 — Receita negativa
- Contexto: 3.513 linhas com receita < 0 (-R$ 21 mi, 4,3%), espalhadas no período; significado não documentado.
- Decisão: tratar como estorno/ajuste; manter no líquido e reportar à parte.
- Consequência: KPIs de ticket calculados no agregado diário, não por linha.

## D05 — Taxa de desconto
- Contexto: `desconto / receita` passa de 100% em algumas categorias.
- Decisão: `taxa_desconto = desconto / (receita + desconto)`, assumindo receita líquida pós-desconto.
- Consequência: taxa limitada a [0, 1]; premissa declarada nas apresentações.

## D07 — Escolha do algoritmo
- Contexto: 242 dias × 2 canais (484 pontos diários; ~212 dias antes do holdout de junho). Cada evento (Black Friday, Dia das Mães, Namorados) ocorre uma única vez. Tendência quase plana em 2026; sazonalidade semanal forte (domingo -23%); novembro é um regime promocional à parte. O enunciado pede explicitamente as variáveis mais importantes.
- Decisão: LightGBM global (um modelo para os dois canais, com canal como feature), com calendário, eventos e lags >= 7, fortemente regularizado. Torneio com as mesmas features e folds: seasonal naive (t-7), Ridge, XGBoost e LightGBM, mais média móvel 7d como referência.
- XGBoost fica no torneio como controle da mesma família: se empatar, LightGBM vence por categóricas nativas e velocidade de retreino no walk-forward; se XGBoost ganhar com margem fora do IC, adotamos XGBoost.
- Descartados antes do torneio: Prophet (sazonalidade anual não identificável com 8 meses; changepoints reagem demais a novembro; fica opcional como baseline); SARIMA/SARIMAX/ETS (lineares, exigem estacionariedade que novembro quebra, eventos únicos viram dummies de 1 observação e previsão recursiva em 7 passos acumula erro); LSTM/Transformer (centenas de pontos não sustentam o número de parâmetros; sem interpretabilidade).
- Consequência: árvores não extrapolam tendência, aceitável dado o nível plano; eventos únicos são aprendidos com 1 exemplo, então o erro em datas especiais é reportado à parte.
- Resultado (WAPE da receita total; junho cego / walk-forward de avaliação): naive 35,1% / 30,9%; média móvel 31,2% / 22,8%; Ridge 22,8% / 25,3%; XGBoost 18,6% / 23,3%; LightGBM 20,1% / 22,3%. Bias de XGBoost e LightGBM em junho ≈ 0.
- Veredito da regra pré-fixada: com 5 semanas de teste (block bootstrap), as árvores superam o Ridge com significância, mas não se separam estatisticamente da média móvel (IC da diferença encosta em zero), e XGBoost e LightGBM empatam. A regra, aplicada literalmente, aponta a média móvel; ela também se mostrou não transitiva no walk-forward, o que registramos como limite do critério.
- Decisão de negócio: LightGBM em produção, com a média móvel como régua de monitoramento. Critério: em semanas comuns os modelos empatam; em semanas com evento, a média móvel quebra (Namorados: 54% × 19%; pós-Dia das Mães: 51% × 29%), e é nelas que a operação depende da previsão. Desempate com o XGBoost por engenharia (retreino ~2,5× mais rápido, categorias nativas). Gatilho de reversão: se o LightGBM não superar a média móvel nas próximas datas especiais, volta-se ao modelo simples.

## D08 — Payday separado em dia 5 e dia 20
- Contexto: o handoff definia payday como dias 5 e 20 (+2 dias). Na EDA (dez-jun, sem janelas de evento, controlando mês e dia da semana), os dias 5-7 vendem +22% acima do esperado e os dias 20-22 ficam neutros (Mann-Whitney p = 0,005 para o payday combinado).
- Decisão: duas features, `is_payday_5` e `is_payday_20`.
- Consequência: o modelo pode usar o efeito do salário sem diluí-lo com o adiantamento; o SHAP mostra a diferença. Ressalva: o efeito pode refletir campanhas recorrentes de início de mês, não só renda.

## D09 — Validação de integridade do pipeline
- Contexto: com pouco histórico e eventos únicos, o maior risco de um forecast é parecer bom no teste por motivos errados (vazamento, extrapolação, comparação desigual).
- Decisão: antes da avaliação final, o pipeline passou por:
  1. Teste de vazamento por perturbação: corromper todos os dados posteriores à origem de previsão não altera nenhuma feature da semana prevista (todas as origens).
  2. Revisão independente do código e dos resultados, separada da etapa de implementação.
  3. Regra de decisão fixada antes da rodada final: menor WAPE no holdout; empate estatístico (IC pareado contém zero) favorece o modelo mais simples.
- O que a auditoria mudou: variáveis de calendário que assumiam valores nunca vistos no treino (mês e semana do ano) foram removidas; feriados ausentes da biblioteca nacional (Carnaval, Corpus Christi) foram incluídos; todos os modelos de ML passaram a usar o mesmo alvo (log da receita); folds de ajuste e de avaliação passaram a ser disjuntos; intervalos de confiança passaram a reamostrar semanas inteiras.
- Consequência: os números apresentados vêm exclusivamente da rodada final, com as correções aplicadas.

## D06 — Dado bruto fora do Git
- Contexto: repositório será público; dado é da empresa.
- Decisão: `data/` no `.gitignore`; README explica onde colocar o CSV.
- Consequência: reprodução exige o arquivo do case.
