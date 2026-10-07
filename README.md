# Case Técnico — Especialista de Dados I | E-commerce

Previsão de receita diária do e-commerce (Seção 1) e proposta de arquitetura de IA para análise de banners (Seção 2).

## Resumo executivo

### O que os dados mostram

- A base cobre nov/2025 a jun/2026 e soma R$ 490,31 mi de receita aprovada líquida. Novembro concentra 26% da receita; a Black Friday chega a 7,5× a mediana diária.
- Pela média, sexta é o melhor dia e domingo o pior. Sem novembro, domingo fica 20% abaixo da média no total, 18% no App e 27% no Site.
- O App representa 76% da receita no período e 79% em junho. Os dois canais têm pico às 11h; à noite, o App sustenta mais volume.
- Ticket é receita ÷ pedidos; preço pago é receita ÷ itens; preço cheio é (receita + desconto) ÷ itens. O ticket vai de R$ 59 em novembro a R$ 93 em junho e, fora de novembro, de aproximadamente R$ 78 a R$ 93. O preço cheio atinge o pico em janeiro. Pedidos estornados continuam contando como pedidos.
- Na mesma base — sem novembro nem janelas de evento e com controle de mês e dia da semana — os dias 5–7 ficam 21,61% acima do esperado no total, 20,32% no App e 24,85% no Site.
- Antes do Dia das Mães, perfumaria feminina e Gifts sobem enquanto a masculina cai; feminina volta ao patamar usual depois de um vale. Na janela de 03–09/05, os lifts são R$ 2,77 mi em feminina e R$ 4,07 mi em Gifts. Na janela de 06–12/06, perfumaria masculina gera lift de R$ 2,93 mi e o pico de Namorados ocorre no próprio dia.
- Perfumaria feminina combina 51% de desconto com 26% da receita. O desconto está associado a volume, mas a decisão econômica depende da margem, ausente da base.

### Receita retida em estornos

Estornos somam R$ 21,05 mi em oito meses, ou R$ 2,63 mi/mês, e permanecem entre 3,9% e 4,6% da receita mensal. Abrir os motivos de estorno é a primeira ação. Como ordem de grandeza, reduzir a taxa em 0,5 p.p. preservaria cerca de R$ 3,7 mi/ano se o período fosse anualizado; essa extrapolação inclui novembro e tende a superestimar um ano típico, e a fração recuperável ainda é desconhecida.

### Previsão com horizonte de 7 dias

O alvo é a receita aprovada líquida por dia e canal: desconto já abatido e estornos incluídos. Oito modelos foram medidos nas mesmas origens temporais: naive sazonal, média móvel, Ridge, XGBoost, LightGBM, ETS, SARIMA e SARIMAX.

- **Recomendação:** um único LightGBM para prever os próximos 7 dias por canal. Ele tem o menor erro diário entre os oito modelos nas 16 semanas, 21,71%, e viés de −2,45%, contra −6,64% a −13,27% dos modelos estatísticos. Além de evitar uma subestimação sistemática relevante para estoque e escala, explica os direcionadores por variável, incorpora calendário e datas comerciais e atende App e Site com o mesmo modelo.
- O SARIMA é o concorrente que vence no total semanal: faz 13,14% nas semanas cheias das 16 semanas e 6,10% nas semanas cheias de junho, contra 13,95% e 10,10% do LightGBM. Não foi adotado porque subestima a receita em 8,81% nas 16 semanas e porque seu ajuste registrou falha de convergência; a família estatística somou três falhas — uma no SARIMA e duas no SARIMAX —, o que aumenta o risco do retreino semanal.
- O SARIMAX chega ao melhor erro semanal de junho, 5,84%, mas fica em 17,03% nas 16 semanas, quase o pior resultado semanal do torneio. O ganho localizado não generaliza.

A combinação do total semanal do SARIMA com o perfil diário do LightGBM também foi testada. Ela alcança 20,62% de erro diário nas 16 semanas, o melhor número do torneio, mas herda o viés de −8,81% do SARIMA; frente aos 21,71% do LightGBM, o ganho arredondado é 1,09 p.p., e o IC 95% da diferença, [−4,77; +2,37], inclui zero. Por isso, não é a entrega atual. Ela só volta à mesa se um fator de correção estimado exclusivamente nas semanas de ajuste reduzir o viés sem eliminar o ganho de erro e esse resultado se repetir fora do ajuste.

A combinação tampouco vence em todo recorte: no walk-forward de jan–mai, o erro diário é 21,31% contra 21,27% do SARIMA; nas semanas cheias do mesmo recorte, o erro semanal é 15,70% contra 15,35% do LightGBM. O ganho de erro absoluto diário do LightGBM sobre a média móvel, convertido para uma semana, vai de aproximadamente R$ 0,06 mi no walk-forward de jan–mai a R$ 1,23 mi em junho.

A regra estatística fixada antes da rodada final, aplicada literalmente aos oito modelos, continua escolhendo a média móvel tanto em junho quanto no walk-forward de jan–mai. A decisão operacional não segue esse resultado porque a regra tinha baixo poder: em junho, a diferença mínima detectável era 19,02 p.p. Nas 16 semanas, o LightGBM é não inferior à média móvel no limite observado — diferença de −3,41 p.p., IC 95% [−9,67; +2,73] —, tem viés menor que os estatísticos e oferece explicabilidade. O critério de calendário “data comercial a até 7 dias” fica como hipótese prospectiva para Dia dos Pais e Black Friday; se o LightGBM não superar a média móvel nessas próximas janelas, o processo volta à régua simples.

O viés agregado do LightGBM próximo de zero também é compensação: em junho, −15,53% em evento/janela e +8,39% em dia normal; no walk-forward de jan–mai, −26,15% em evento/janela e +1,05% em dia normal. A ação é reforçar estoque nas semanas de data e medir o uplift, sem embuti-lo na previsão antes de validá-lo.

Para capacidade e estoque, a faixa principal é semanal. Calibrada nas semanas de ajuste com o LightGBM seed 42, ela vai de −35,34% a +36,20% do previsto e cobre 90,91% dos blocos do walk-forward de jan–mai e 80,00% dos blocos de junho. A folga logística de referência é o P80 do erro absoluto semanal: 28,69% nas 25 semanas cheias de ajuste e avaliação. A faixa conformal diária fica como diagnóstico: cobre 93,33% dos dias do walk-forward e 90,00% dos dias de junho, mas sua largura média é próxima do próprio valor previsto.

### Seção 2: scoring de banners

- A proposta começa pela decisão de negócio e pelo KPI de receita por impressão, com compliance e teste A/B antes da escala. O App responde por 76% da receita no período e 79% em junho, o que orienta a priorização sem apagar o Site.
- O ponto de equilíbrio substitui cenários: o projeto paga a operação com ganho de 0,054% da receita mensal; paga o investimento em 12 meses com 0,100% e em 6 meses com 0,145%. São premissas a substituir por margem, tráfego e receita influenciada aprovados pelo negócio.
- A demo híbrida separa semântica e sinais quantitativos. Com 24 rótulos, o CLIP acerta 100% de mensagem na média de 20 partições; atributos simples acertam 100% de fundo, preço visível e elemento de produto. O conjunto é sintético e serve para validar o pipeline, não para estimar produção.
- Em CPU aquecida, o CLIP em lote mede 32,3 ms/banner; 10 mil banners levam 5,39 min na projeção. O A/B de referência para CTR de 2,0% → 2,1% requer cerca de 315 mil usuários por variante, com randomização persistente por usuário e guarda de receita por impressão.

## Como ler este repositório em 5 minutos

1. Este README.
2. `notebooks/03_modelagem.ipynb`: torneio, combinação e incerteza.
3. `notebooks/02_comportamento_consumo.ipynb`: desconto, mix, datas de presente e salário.
4. `docs/decisoes_tecnicas.md`: decisões no formato contexto → decisão → consequência.
5. `docs/secao2_arquitetura_ia.md`: business case e arquitetura da Seção 2.

## Estrutura

```text
notebooks/
  01_eda.ipynb                    análise exploratória
  02_comportamento_consumo.ipynb  comportamento de consumo
  03_modelagem.ipynb              torneio, combinação e explicação
  04_demo_clip.ipynb              demo híbrida de banners
src/
  data_loader.py, eda.py, consumo.py
  calendario.py, features.py
  models.py, train.py, combinacao.py
  evaluate.py, explain.py, visual.py, intervalos.py
  clip_demo.py
docs/
  decisoes_tecnicas.md
  secao2_arquitetura_ia.md
reports/
  figures/
  metrics/
tests/
```

## Como reproduzir

O dado não é versionado. Coloque o arquivo em `data/raw/vendas.csv`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -m src.train
.venv/Scripts/python -m src.intervalos
.venv/Scripts/python -m src.combinacao
```

O projeto usa Python 3.10. Os notebooks têm preparação para Google Colab; o notebook da Seção 2 usa GPU quando disponível.

## Como o modelo foi validado

- Sem split aleatório: treino no passado e previsão da semana seguinte, em walk-forward.
- Horizonte de 7 dias: somente calendário conhecido e histórico com lags de pelo menos 7 dias.
- Junho ficou fora do ajuste, mas foi usado uma vez como diagnóstico na rodada 1. Depois das correções de integridade, a regra foi fixada antes da rodada final; a evidência principal reúne o walk-forward de jan–mai e junho em 16 blocos semanais.
- O teste de perturbação do futuro cobre todas as origens e os intervalos de confiança usam 2.000 reamostragens de blocos semanais com seed 42.
- Todos os modelos usam as mesmas origens; a comparação sempre inclui baselines.

## Limitações

- Oito meses de histórico: cada data especial aparece uma vez e, sem um ano completo, o Prophet não aprende a sazonalidade anual.
- Sem cliente, SKU, margem, estoque, mídia ou calendário de campanhas.
- Categoria ilegível em 6% da receita, mantida no total como “Não mapeada”.
- Junho contém a Copa do Mundo, ausente das variáveis.
- A correção prospectiva do viés da combinação precisa de validação em novas datas antes que ela possa ser reconsiderada.
- As leituras de consumo são associações, não efeitos causais.
