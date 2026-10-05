# Seção 2 — Inovação e Arquitetura de IA: scoring de banners

## Resumo executivo

- **Posição: eu não seguiria com o projeto como ele está.** A ideia é boa, mas está crua: não há KPI definido, nem critério do que é um banner "bom", nem decisão de negócio clara que o score vá apoiar. Construir o pipeline agora seria otimizar uma métrica que ninguém escolheu. O primeiro passo é entender o escopo; só depois uma PoC e, se ela provar valor, um MVP.
- O problema não é a ferramenta, é a pergunta. Antes de escolher tecnologia, definimos qual decisão o score vai apoiar e qual KPI prova que funcionou: recomendamos **receita por impressão**, não CTR nem "beleza".
- A ferramenta no-code não homologada é uma restrição de compliance, não um bloqueio. Seguimos em duas trilhas: MVP dentro do ambiente já homologado e, em paralelo, pedido formal de avaliação da ferramenta com Segurança da Informação e Jurídico.
- Tecnicamente: embeddings de imagem (CLIP) transformam cada banner em uma "impressão digital numérica". Sobre ela classificamos atributos, agrupamos estilos visuais e treinamos um score de receita por impressão, controlando posição e semana.
- Sequência de 7–8 semanas: escopo (1–2), PoC (2) e MVP (4), com critério de parada em cada etapa. O score só vale se ganhar um teste A/B.
- Custo × retorno sobre **margem**, com custo recorrente e três cenários: no cenário base o projeto se paga em cerca de um ano; no pessimista, a perda fica limitada ao custo do MVP porque os critérios de parada interrompem o investimento (seção 3.3).

---

## 1. Condução do problema com o time

### 1.1 Do "qual ferramenta" para "qual decisão"

O time chegou com uma solução (pipeline no-code). O primeiro passo é voltar ao problema, numa sessão de discovery curta:

| Pergunta | Por que importa |
|---|---|
| Qual decisão o score vai mudar? (escolher banner da home, priorizar criativos, orientar briefing da agência) | Define se precisamos de ranking, classificação ou diagnóstico. |
| Como sabemos que deu certo? | Define o KPI: CTR, conversão pós-clique, receita por sessão. |
| O que é um banner "bom" hoje e quem decide? | Revela o critério atual (gosto, guideline de marca, dado). |
| Quem usa o resultado e com que frequência? | Define se é batch semanal ou checagem antes de publicar. |
| Que dados existem? (banners históricos, impressões, cliques, posição, campanha) | Sem histórico de performance, não há score supervisionado, só descritivo. |

### 1.2 Sequência: escopo → PoC → MVP

| Etapa | Duração | Pergunta que responde | Sai com |
|---|---|---|---|
| Entendimento de escopo | 1-2 semanas | Qual decisão, qual KPI, que dados existem? | KPI escolhido, inventário de dados, critério de sucesso |
| PoC | 2 semanas | Existe sinal? Atributos visuais se relacionam com o KPI? | Análise descritiva com banners históricos; go/no-go |
| MVP | 4 semanas | O score melhora o KPI de verdade? | Score + teste A/B; go/no-go para escala |

Se o entendimento de escopo mostrar que não há dado de performance por banner, a PoC muda de "score preditivo" para "catálogo visual descritivo" (clusters e atributos), que já tem valor para o time de criação.

### 1.3 A ferramenta não homologada: compliance, não bloqueio

A homologação existe por bons motivos: imagens de campanha podem conter pessoas (direito de imagem, contratos com embaixadores), material não lançado (sigilo comercial) e dados de performance. Enviar isso a uma ferramenta não avaliada é risco jurídico e de marca.

Caminho proposto, em paralelo:
1. **Trilha MVP:** construir no ambiente de dados e nuvem já homologados pelo grupo, com o time de dados como executor técnico.
2. **Trilha homologação:** se a ferramenta no-code for essencial para a autonomia do time, abrir o processo formal de avaliação de fornecedor com Segurança da Informação e Jurídico, com o MVP servindo de caso de uso.
3. **Sandbox:** enquanto não há aprovação, testes só com banners já publicados (públicos), sem dados de cliente.

### 1.4 Roteiro da primeira conversa com o time

1. **Reconhecer a iniciativa:** a ideia nasceu de quem conhece os banners por dentro; é exatamente o tipo de inovação que queremos.
2. **Separar problema de ferramenta:** "a ferramenta não está homologada; o problema que vocês acharam continua de pé, e vamos resolvê-lo".
3. **Fazer as perguntas de escopo** (seção 1.1) juntos, saindo com a decisão que o score apoia e o KPI.
4. **Combinar o que começa já, sem esperar a homologação:** rotular uma amostra de banners (o que é "bom" para vocês?), levantar o histórico de performance com o time de web analytics, e explorar o que dá para fazer nas ferramentas já homologadas (planilhas e painéis corporativos).
5. **Definir papéis e prazos:**

| Quem | Responsável por |
|---|---|
| Time do hackathon | Dono do problema: KPI, rótulos, hipóteses, acompanhamento do A/B |
| Time de dados | Pipeline, embeddings, modelo, validação |
| Segurança da Informação | Parecer sobre a ferramenta no-code (prazo combinado na abertura do processo) |
| Jurídico | Direito de imagem e uso de material de campanha |
| Sponsor de negócio | Aprovar o A/B e o espaço de teste |

### 1.5 Um time sem programação continua dono do problema

| Time do hackathon | Time de dados |
|---|---|
| Define critérios de "bom" e rotula uma amostra de banners | Constrói pipeline, embeddings e modelo |
| Formula hipóteses ("banner com pessoa converte mais?") | Testa as hipóteses com dado |
| Desenha e acompanha o A/B | Mede e reporta significância |
| Usa o resultado num dashboard ou planilha | Mantém, monitora e retreina |

O entusiasmo do hackathon é ativo da cultura de inovação. A mensagem para o time é "vamos chegar lá por um caminho seguro", não "não pode".

### 1.6 Que score? Cenários possíveis

"Score de banner" pode significar coisas muito diferentes. Cada cenário apoia uma decisão distinta e exige dados distintos. Escolher um é a principal saída do entendimento de escopo.

| Score | O que mede | Decisão que apoia | Dado necessário | Armadilha |
|---|---|---|---|---|
| CTR | Cliques ÷ impressões | Qual banner vai para a home | Impressões e cliques por banner e posição | Premia o chamativo, não o que vende |
| Conversão pós-clique | Pedidos ÷ cliques | Qual banner traz quem compra | Clique ligado à sessão e ao pedido | Volume baixo por banner; ruído alto |
| Receita por impressão (RPM) | Receita atribuída ÷ impressões | Priorizar espaço nobre da home | CTR × conversão × ticket | Atribuição: a compra pode ter outra origem |
| Engajamento | Tempo na página de destino, rolagem, rejeição | Banner que leva a uma página que prende | Analytics da página de destino | Tempo alto pode ser confusão, não interesse |
| Adição ao carrinho | Add-to-cart ÷ cliques | Banner de produto/lançamento | Eventos de carrinho por origem | Carrinho abandonado não é venda |
| Fadiga criativa | Queda de CTR ao longo dos dias | Quando trocar o banner | Série diária de CTR por banner | Confunde fadiga com sazonalidade |
| Aderência à marca (qualitativo) | Distância do banner ao guideline | Revisão antes de publicar | Guideline + banners aprovados como referência | Subjetivo; precisa de rótulo humano |
| Acessibilidade / legibilidade | Contraste, tamanho e proporção de texto | Checagem técnica pré-publicação | Só a imagem | Não diz nada sobre venda |

Recomendação inicial: **receita por impressão** como KPI de negócio, com CTR e conversão como diagnósticos. Os três qualitativos/técnicos (marca, acessibilidade, fadiga) entram como checagens, não como score principal.

### 1.7 Provocações para a discussão (hipóteses, sem dado ainda)

Sem dados de banners, o valor está em fazer as perguntas certas. Algumas, ligadas ao que vimos na Seção 1:

1. **CTR alto pode vender pior.** Na base de vendas, novembro teve desconto médio de 55% e ticket de R$ 59; junho, desconto de 35% e ticket de R$ 93. Um score de CTR tende a premiar banners de desconto agressivo, que atraem clique mas, pelo padrão observado, trazem pedidos menores. Por isso receita por impressão, e não CTR.
2. **O App é 76% da receita.** Banner pensado para desktop pode ser o menos visto. O score deveria ser por canal e formato, não um só.
3. **O horário muda o público.** O Site vende mais pela manhã e o App é mais noturno. O mesmo banner pode performar diferente conforme o horário em que é exibido; rotação por horário é um teste barato.
4. **O "banner bonito" pode ser só o banner bem posicionado.** Sem controlar posição e período, o score aprende o calendário promocional, não a imagem.
5. **Datas especiais se comportam como outro negócio.** Dia das Mães teve pico na véspera (sábado), não no domingo. A troca de banner precisa acompanhar a curva de compra, não a data comemorativa.
6. **Pessoa no banner vende mais?** Hipótese clássica, mas com risco de viés: se o histórico favoreceu um perfil de modelo, o score reproduz. Testar com A/B e auditar por atributo.

---

## 2. Implementação técnica (sem impeditivos ferramentais)

### 2.1 Arquitetura

```mermaid
flowchart LR
    A[CMS / DAM<br/>banners + metadados] --> B[Pré-processamento<br/>resize, dedupe, OCR]
    W[Web analytics<br/>impressões, cliques,<br/>conversão, posição] --> F
    B --> C[Embeddings CLIP<br/>impressão digital da imagem]
    B --> Q[Atributos quantitativos<br/>cor, contraste, % texto,<br/>presença de pessoa]
    C --> Z[Classificação zero-shot<br/>promoção? produto? pessoa?]
    C --> K[Clusters de estilo visual]
    C --> F[Base de features]
    Q --> F
    Z --> F
    F --> S[Score de receita por impressão<br/>regressão regularizada / LightGBM<br/>controlando posição e semana]
    S --> D[Dashboard semanal +<br/>checagem pré-publicação]
    S --> T[Teste A/B<br/>valida causalidade]
    D --> M[Monitoramento<br/>drift e retreino]
```

### 2.2 Componentes

| Etapa | Escolha | Por quê |
|---|---|---|
| Embeddings | CLIP (open-source; ex.: ViT-B/32) | Coloca imagem e texto no mesmo espaço: permite perguntar "este banner mostra uma promoção?" sem treinar nada (zero-shot). Roda em CPU para centenas de imagens. |
| Atributos quantitativos | OpenCV + OCR | Brilho, contraste, cor dominante, proporção de texto, preço visível. Explicáveis para o time de criação. |
| Atributos qualitativos | Zero-shot CLIP com rótulos definidos pelo time | "Produto em destaque", "pessoa", "fundo claro", "preço/desconto", "lançamento". |
| Estilos visuais | K-means sobre embeddings | Revela famílias de banner e permite comparar performance por família. |
| Score | Regressão regularizada (MVP) → LightGBM | Prever receita por impressão a partir de embeddings + atributos + contexto. Começar simples. |
| Busca por similaridade | FAISS (fase de escala) | "Mostre banners parecidos com este que performaram bem." |
| Serviço | Batch semanal (MVP) → API (escala) | O MVP não precisa de tempo real. |
| Monitoramento | Drift dos embeddings e da receita por impressão | Nova identidade visual ou sazonalidade mudam o padrão. |

Opção complementar: um LLM multimodal (via provedor homologado) pode gerar uma crítica em linguagem natural de cada banner frente ao guideline de marca. Útil para o lado qualitativo e para o time sem programação; custo por imagem e consistência precisam ser avaliados.

### 2.3 Prova de conceito: o que uma demo de 32 banners já mostra

Para testar a peça central da arquitetura sem usar dado da marca, geramos 32 banners sintéticos com gabarito conhecido (fundo, cor, mensagem, preço, frasco) e rodamos o CLIP (`notebooks/04_demo_clip.ipynb`).

| Atributo | Zero-shot (sem treino) | Clusters (sem rótulo) | Classificador com 24 exemplos rotulados |
|---|---|---|---|
| Cor dominante | 100% | — | 94% |
| Mensagem (promoção × lançamento) | 31% | separa 100% | 100% |
| Fundo claro × escuro | 56% | — | 66% |
| Preço visível | 50% | — | 19% |
| Frasco de produto | 50% | — | 25% |

Acaso = 50% nos atributos de duas classes. Velocidade: 38 ms por banner em CPU (10 mil banners ≈ 6 minutos).

O que isso ensina para o MVP:
1. **O embedding captura o que é grande e dominante** (cor, tipo de mensagem). Os clusters separaram promoção de lançamento sem nenhum rótulo.
2. **Zero-shot depende de fazer a pergunta certa.** Prompts em inglês não casaram com "50% OFF" e "NOVO" em português. Com só 24 exemplos rotulados, um classificador simples acerta 100%. Rotular uma amostra pequena é exatamente o papel do time do hackathon.
3. **Detalhes pequenos não aparecem no embedding global** (preço, frasco). Por isso a arquitetura combina embeddings com OCR e atributos quantitativos.

Por que 19% e 25%, abaixo do acaso? Quando não há sinal, a validação cruzada com poucos exemplos tende a ficar abaixo de 50%: ao separar exemplos de uma classe para teste, o treino fica levemente enviesado para a outra. Leia como "atributo não detectado". E, para zero-shot com texto em português, a escolha natural é um CLIP multilíngue (ou SigLIP multilíngue), em vez do CLIP original treinado com legendas em inglês.

Limite: banners sintéticos são mais simples que os reais, e a demo não mede performance comercial (não há dado de clique). Ela valida componentes, não o score.

### 2.4 Como o embedding vira nota

| Peça | Especificação do MVP |
|---|---|
| Unidade | Um banner exibido em um espaço (posição) numa semana |
| Alvo | Receita atribuída ÷ impressões (receita por impressão), em log |
| Variáveis da imagem | Componentes principais do embedding (≈ 20–50), atributos de OCR (preço visível, % de texto) e atributos quantitativos (cor, contraste) |
| Controles | Efeitos fixos de posição e de semana, desconto da campanha e canal: o score compara banners dentro do mesmo espaço e período |
| Modelo | Regressão regularizada no MVP (explicável); LightGBM se houver volume |
| Validação | Temporal: treinar em campanhas passadas e medir, nas semanas seguintes, se o ranking do score acerta a ordem de receita por impressão dos banners de cada espaço (correlação de ranking) |
| Saída | Nota relativa (percentil dentro do espaço) e os atributos que mais pesaram, para o time de criação |
| Infraestrutura | Vetores guardados num índice vetorial (FAISS) junto ao catálogo do DAM; ~38 ms por banner em CPU, custo de cálculo desprezível |

### 2.5 O ponto técnico que mais importa: causalidade

Um banner na primeira posição da home durante a Black Friday tem CTR alto por causa da posição e do momento, não da imagem. Se o modelo não controlar isso, ele aprende "banner de Black Friday é bom". Por isso:
- posição, período, campanha e desconto entram como variáveis de controle;
- o score é validado com teste A/B (mesma posição, mesmo período, criativos diferentes) antes de virar critério de decisão.

---

## 3. Justificativa para a gestão sênior

### 3.1 Tradução

| Termo técnico | Como explicar |
|---|---|
| Embedding | Uma impressão digital numérica da imagem: banners parecidos têm impressões parecidas. |
| Zero-shot | O modelo reconhece "tem uma pessoa" ou "tem preço" sem precisar de exemplos rotulados. |
| Score | A receita esperada por impressão deste banner, comparada a banners no mesmo espaço e período. |
| Teste A/B | Metade do público vê o banner A, metade vê o B; a diferença medida é a prova. |

### 3.2 MVP de 4 semanas (depois de escopo e PoC: 7–8 semanas no total)

| Semana do MVP | Entrega | Segue se… |
|---|---|---|
| 1 | Base de banners históricos com impressões, cliques e receita atribuída, por espaço e semana | Há dado suficiente (ex.: 200+ banners com impressões) |
| 2 | Embeddings, atributos, clusters | Os grupos fazem sentido para o time de criação |
| 3 | Score de receita por impressão com validação temporal | O ranking do score bate a régua (média histórica do espaço) |
| 4 | Desenho e início do A/B | O A/B decide a escala |

Equipe: 1 cientista de dados dedicado, apoio parcial de engenharia de dados e o time do hackathon como dono do negócio. Infraestrutura: nuvem homologada existente; CLIP roda em CPU nesse volume, então custo de GPU é marginal.

### 3.3 Custo × retorno

Todos os valores de custo, margem e ganho abaixo são **premissas ilustrativas** a substituir pelos números de Finanças. A receita mensal vem da base do case (R$ 490 mi em 8 meses ≈ R$ 61 mi/mês) e pode ser só parte do total real.

**Premissas:** custo do escopo + PoC + MVP = R$ 100 mil (uma vez); custo recorrente de operação = R$ 10 mil/mês; margem de contribuição = 30% da receita incremental.

| Cenário | Ganho de receita (% da receita mensal) | Margem incremental/mês | Menos o custo recorrente | Payback |
|---|---|---|---|---|
| Pessimista | 0% (o A/B não mostra efeito) | R$ 0 | −R$ 10 mil | Não se paga: o projeto para no critério de parada, e a perda fica limitada ao custo do MVP |
| Base | 0,1% (≈ R$ 61 mil) | ≈ R$ 18 mil | ≈ R$ 8 mil | ≈ 12 meses |
| Otimista | 0,3% (≈ R$ 183 mil) | ≈ R$ 55 mil | ≈ R$ 45 mil | ≈ 2 meses |

**O A/B não consegue medir 0,1% da receita total.** Por isso o teste usa um **KPI intermediário detectável**, com a receita por impressão como guarda (não pode piorar):
- Exemplo: CTR base de 2% no espaço testado; para detectar +5% relativo (2,0% → 2,1%) com 80% de poder e 5% de significância, são precisos ~310 mil impressões por variante. Em um espaço de home com tráfego alto, isso leva dias, não meses.
- O ganho financeiro se confirma depois, acompanhando a receita por impressão do espaço por algumas semanas.

**Custo de não fazer:** decisões de criativo continuam sem dado; o time do hackathon, desmotivado, tende a buscar ferramentas paralelas sem governança, exatamente o risco que a homologação quer evitar.

Retornos não financeiros: briefing de criação baseado em dado, menos tempo de revisão manual, e a mesma infraestrutura de embeddings reaproveitável para busca no DAM, detecção de duplicados e checagem de guideline de marca.

### 3.4 Riscos, levantados antes da pergunta

| Risco | Mitigação |
|---|---|
| LGPD e direito de imagem (pessoas nos banners) | Detectar presença de pessoa, nunca identificar quem é; nenhum dado de cliente no pipeline; validação do Jurídico. |
| Viés (corpos, etnias, faixas de preço) | Se o histórico favoreceu um perfil, o score repete. Auditar o score por atributo e nunca usar como critério único de veto. |
| Brand safety | Score apoia a decisão; aprovação final segue com marca e criação. |
| Correlação ≠ causa | Controles de posição/período e A/B obrigatório antes de escalar. |
| Drift | Nova identidade visual ou campanha muda o padrão: monitorar e retreinar mensalmente. |
| Dependência do time de dados | Interface simples para o time do hackathon e documentação; trilha de homologação para autonomia futura. |

### 3.5 Roadmap

Entendimento de escopo (1-2 semanas, KPI e dados) → PoC (2 semanas, banners históricos, descritivo) → MVP (4 semanas, score + A/B) → Escala (API pré-publicação, busca por similaridade, novos canais como App e e-mail). Cada passagem depende de um gate com critério numérico.

### 3.6 O que pedimos à liderança

1. Acesso aos dados de performance de banners (impressões, cliques, posição).
2. Um sponsor de negócio para o A/B.
3. Aval para a trilha de homologação com Segurança da Informação e Jurídico.
4. Sete a oito semanas de um cientista de dados (escopo, PoC e MVP), com critério de parada em cada etapa.
