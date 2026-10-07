# Decisões técnicas

Formato: contexto → decisão → consequência.

## D01 — Alvo do modelo

- Contexto: “venda diária” pode significar pedidos, receita bruta ou receita líquida.
- Decisão: prever receita aprovada líquida por dia e canal, com desconto já abatido e estornos incluídos; pedidos ficam como leitura secundária de volume.
- Consequência: o alvo se aproxima do caixa realizado, mas mistura demanda e estorno. Ticket é receita ÷ pedidos; preço pago é receita ÷ itens; preço cheio é (receita + desconto) ÷ itens.

## D02 — Horizonte de 7 dias

- Contexto: d+1 oferece pouco tempo para ajustar estoque, escala e campanha.
- Decisão: horizonte de 7 dias, apenas com calendário conhecido e histórico defasado por pelo menos 7 dias.
- Consequência: a previsão serve ao planejamento sem usar informação futura.

## D03 — Categorias corrompidas

- Contexto: 5.341 linhas, equivalentes a 6% da receita, têm categoria irrecuperável.
- Decisão: agrupar como “Não mapeada”, manter no total e excluir dos recortes por categoria.
- Consequência: o total é preservado e a análise por categoria cobre 94% da receita.

## D04 — Receita negativa

- Contexto: 3.513 linhas somam −R$ 21,05 mi e são compatíveis com estorno ou ajuste.
- Decisão: manter no alvo líquido e reportar separadamente.
- Consequência: estornos representam R$ 2,63 mi/mês no período e 3,9%–4,6% da receita por mês; abrir os motivos vira uma frente de receita retida.

## D05 — Taxa de desconto

- Contexto: `desconto / receita` pode superar 100%.
- Decisão: usar `desconto / (receita + desconto)`, assumindo receita após desconto.
- Consequência: a taxa fica limitada a [0, 1], com a premissa declarada.

## D07 — Escolha do algoritmo e previsão diária

- Contexto: há 242 dias e dois canais, cada evento aparece uma vez e novembro é um regime promocional distinto. A operação planeja por semana, enquanto o enunciado pede a previsão diária e os direcionadores da venda.
- Decisão: testar de fato oito modelos nas mesmas origens: naive sazonal, média móvel de 7 dias, Ridge, XGBoost, LightGBM, ETS, SARIMA e SARIMAX. Prophet não entra porque, sem um ano completo, não aprende a sazonalidade anual; redes neurais não são proporcionais ao volume de dados.
- Resultado diário: nas 16 semanas, LightGBM faz WAPE de 21,71% e viés de −2,45%, contra 22,29%/−13,27% do ETS, 22,03%/−8,81% do SARIMA e 23,38%/−6,64% do SARIMAX. O LightGBM é o melhor modelo puro no erro diário e o menos enviesado entre esses quatro.
- Resultado semanal: nas semanas cheias das 16 semanas, SARIMA faz 13,14% contra 13,95% do LightGBM; em junho, faz 6,10% contra 10,10%. Apesar de vencer nessa métrica, o SARIMA não foi adotado: seu viés é −8,81% nas 16 semanas e seu ajuste registrou falha de convergência. A família estatística somou três falhas — uma no SARIMA e duas no SARIMAX —, um risco para o retreino semanal. O SARIMAX chega a 5,84% em junho, mas piora para 17,03% nas 16 semanas; o ganho não generaliza.
- Entrega: usar um único LightGBM para a previsão diária de 7 dias à frente nos dois canais. Além do menor erro diário e do viés mais baixo que o dos modelos estatísticos, ele incorpora calendário e datas comerciais e explica os direcionadores por variável.
- Experimento concluído: combinar o total semanal do SARIMA com o perfil diário do LightGBM produz WAPE diário de 20,62% nas 16 semanas, o melhor número do torneio, e preserva o erro semanal de 13,14% do SARIMA. Porém, também preserva seu viés de −8,81%; frente ao LightGBM, a diferença é −1,08 p.p., IC 95% [−4,77; +2,37], sem ganho estatisticamente demonstrável. No walk-forward de jan–mai, ainda perde por pouco para o SARIMA no diário, 21,31% contra 21,27%, e para o LightGBM no semanal de semanas cheias, 15,70% contra 15,35%.
- Consequência: a combinação fica como caminho testado e não adotado por ora. Ela será reconsiderada somente se um fator de correção estimado nas semanas de ajuste reduzir o viés sem eliminar o ganho de erro e se o resultado se sustentar fora do ajuste.

## D08 — Salário separado do adiantamento

- Contexto: na base única sem novembro e sem janelas de evento, com controle por mês e dia da semana, os dias 5–7 ficam +21,61% no total, +20,32% no App e +24,85% no Site. Para o total, Mann–Whitney p = 0,000057; os dias 20–22 ficam neutros.
- Decisão: manter `is_payday_5` e `is_payday_20` como variáveis separadas.
- Consequência: o modelo pode representar início do mês sem atribuir o mesmo efeito ao adiantamento; campanha recorrente continua sendo explicação alternativa.

## D09 — Integridade e histórico das rodadas

- Contexto: junho fica fora do ajuste, mas foi consultado como diagnóstico na rodada 1. Nessa rodada, LightGBM fez 23,48%, Ridge 23,63%, XGBoost 29,71%, média móvel 31,20% e naive sazonal 35,09% em junho; LightGBM menos XGBoost foi −6,22 p.p., IC 95% [−12,97; −0,47].
- Decisão: registrar a rodada 1 e corrigir o pipeline antes de fixar a regra da rodada final. Entre as rodadas, foram removidas variáveis de calendário que extrapolavam, igualado o alvo em log dos modelos de ML, separados os folds de ajuste e avaliação, completados feriados, reduzida a redundância de features, ampliado o tuning de folhas e trocada a reamostragem por blocos semanais. O teste de perturbação do futuro permaneceu obrigatório.
- Decisão: regra fixada antes da rodada final — menor WAPE diário; se o IC pareado incluir zero, vence o modelo mais simples; o walk-forward de jan–mai verifica consistência com junho.
- Consequência: a rodada final é a fonte dos números de decisão. A regra literal com oito modelos aponta a média móvel em junho e no walk-forward de jan–mai, enquanto a decisão operacional é reenquadrada em D13.

## D10 — Incerteza e folga logística

- Contexto: WAPE diário não dimensiona sozinho a capacidade semanal, e a faixa diária é larga.
- Decisão: usar como principal a faixa semanal empírica do LightGBM seed 42, calibrada nas semanas de ajuste: erro relativo P10 de −35,34% e P90 de +36,20%. Ela cobre 90,91% dos blocos do walk-forward de jan–mai e 80,00% dos blocos de junho.
- Decisão: dimensionar folga logística pelo P80 do erro absoluto semanal, 28,69% nas 25 semanas cheias de ajuste e avaliação, e não pela média do erro.
- Consequência: estoque e capacidade partem do total semanal mais a folga; atendimento usa o perfil por dia e hora. A faixa conformal diária fica como detalhe diagnóstico: cobertura de 93,33% dos dias no walk-forward e 90,00% dos dias em junho, com largura média próxima do previsto.

## D11 — Leitura de comportamento de consumo

- Contexto: forecast responde quanto; o negócio também precisa entender como desconto, mix e ocasião se associam à venda.
- Decisão: regressões diárias controladas por dia da semana e mês, erros HAC e comparações de mix em janelas definidas.
- Consequência: os resultados são hipóteses, não causalidade. Antes do Dia das Mães, feminina e Gifts sobem enquanto masculina cai; em Namorados, masculina concentra o lift. Margem, campanha e cliente são necessários para decidir.

## D12 — Limitações de validação

- Contexto: ajuste e avaliação do walk-forward de jan–mai usam semanas intercaladas; junho tem cinco blocos, sendo o último parcial; datas especiais aparecem uma vez.
- Decisão: declarar o potencial otimismo do walk-forward, o baixo poder de junho e a necessidade de teste prospectivo.
- Consequência: nenhuma vantagem pequena é tratada como prova definitiva; a próxima evidência vem de novas datas.

## D13 — Torneio ampliado e critério de decisão

- Contexto: o torneio de oito modelos expõe um trade-off operacional. Nas 16 semanas, o LightGBM tem 21,71% de erro diário e −2,45% de viés; a combinação reduz o erro diário para 20,62%, mas leva o viés a −8,81%, e o IC 95% da diferença contra o LightGBM, [−4,77; +2,37], inclui zero. A regra literal ainda favorece a média móvel nos dois recortes, mas em junho seu MDE é 19,02 p.p.; ela não tem poder para separar diferenças menores.
- Decisão: entregar um único LightGBM para os próximos 7 dias nos dois canais. SARIMA e combinação ficam registrados como evidência: o primeiro vence no total semanal, e a segunda alcança o menor erro diário do torneio, mas ambos carregam subestimação sistemática. A média móvel permanece como régua. O critério de calendário B — data comercial a até 7 dias — fica publicado como hipótese prospectiva: nas 16 semanas, LightGBM e média móvel fizeram 22,34% e 28,91% nas oito semanas com data, e 20,82% e 19,76% nas oito demais. O desempate LightGBM × XGBoost usa as 16 semanas, 21,71% contra 22,03%, e engenharia: categorias nativas, retreino frequente e explicação já integrada.
- Consequência: abre-se mão de cerca de 1 p.p. de erro diário para não carregar mais 6,4 p.p. de viés sistemático. A combinação volta à mesa se a correção estimada apenas nas semanas de ajuste reduzir esse viés sem eliminar o ganho e se o resultado se repetir fora do ajuste. Dia dos Pais e Black Friday são os próximos testes da escolha contra a média móvel; se o LightGBM não a superar nessas janelas, a previsão diária volta à régua simples. O viés do LightGBM continua monitorado por segmento: em junho, −15,53% em evento/janela e +8,39% em dia normal; no walk-forward de jan–mai, −26,15% e +1,05%. Até haver ajuste validado, a ação é reforço de estoque nas semanas de data.

## D06 — Dado bruto fora do Git

- Contexto: o repositório pode ser público e o dado pertence à empresa.
- Decisão: manter `data/` no `.gitignore` e documentar o caminho `data/raw/vendas.csv`.
- Consequência: a reprodução requer acesso autorizado ao CSV.
