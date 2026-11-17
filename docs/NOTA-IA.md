# Nota de estudo: embeddings, RAG e temperature/top-k/top-p no Cortaí

Esta nota liga os conceitos do módulo ao código do projeto. Cada seção termina com **"no projeto"**:
onde você vê o conceito funcionando e o que dá para mexer.

---

## 0. Mapa rápido

| Conceito | Onde aparece | Arquivo |
|---|---|---|
| Embedding de texto | fichas das frutas viram vetores (Transformers.js) | `rag/ingest.js` |
| Embedding de imagem | recorte da fruta vira vetor (CLIP) | `rag/identify.js` |
| Banco vetorial + cosseno | Neo4j guarda os chunks e acha os mais parecidos | `rag/search.js` |
| RAG | pergunta → trechos → LLM responde com fontes | `src/ml/llm.js`, `rag/ask.js` |
| Temperature / top-k no LLM | Gemini Nano no Chrome | `src/ml/llm.js` |
| Temperature / top-k / top-p no LLM | Ollama (experimento) | `rag/ask.js`, `rag/experiments/` |
| Temperature / top-k / top-p no agente | escolha do corte | `src/ml/sampling.js` |
| Masking (proibir opção) | corte que pega bomba = proibido | `src/ml/planner.js` |
| Thresholds de visão (não é sampling!) | confiança e IoU do YOLO | `src/ml/worker.js` |

---

## 1. Embeddings

**O que é:** uma função que transforma algo (texto, imagem) em uma lista de números, um **vetor**, de
forma que **coisas parecidas em significado ficam perto** no espaço.

```
"açaí é roxo e cresce em cachos"     → [0.12, -0.80, 0.33, ...]  (384 números num MiniLM)
"fruto escuro de palmeira em cacho"  → [0.10, -0.75, 0.30, ...]  ← perto
"bomba explode se cortada"           → [-0.60, 0.20, 0.90, ...]  ← longe
```

O modelo não procura palavras iguais: ele aprendeu, com muito texto, a posicionar significados.
"Fruto escuro de palmeira" fica perto de "açaí" sem compartilhar nenhuma palavra.

### Similaridade cosseno

Mede o ângulo entre dois vetores: 1 = mesma direção (muito parecido), 0 = nada a ver, −1 = oposto.

```
cos(a, b) = (a · b) / (|a| · |b|)
```

Exemplo com 3 dimensões: `a = [1, 2, 0]`, `b = [2, 3, 1]`

- `a · b = 1·2 + 2·3 + 0·1 = 8`
- `|a| = √5 ≈ 2,236` e `|b| = √14 ≈ 3,742`
- `cos = 8 / (2,236 · 3,742) ≈ 0,956`, ou seja, muito parecidos

Se os vetores forem normalizados (comprimento 1), o cosseno vira só o produto escalar. Por isso o
`ingest.js` gera os embeddings com `normalize: true`, e o índice vetorial do Neo4j é criado com
`vector.similarity_function: 'cosine'`.

### Texto × imagem

- **Modelo de texto multilíngue** no Transformers.js (ex.: `Xenova/paraphrase-multilingual-MiniLM-L12-v2`):
  usado no RAG. A Unidade 10 usa a mesma biblioteca; trocamos por um modelo multilíngue porque as
  fichas estão em português.
- **CLIP** (imagem **e** texto no mesmo espaço): o recorte de uma fruta e a frase "foto de um guaraná"
  ficam perto. Isso permite identificar a fruta comparando com exemplos, sem treinar um classificador.

**No projeto:** `rag/identify.js` pega o recorte que o YOLO encontrou, gera o embedding CLIP e procura o
vizinho mais próximo numa galeria de exemplos rotulados. Dá para ver os embeddings num gráfico 2D (PCA):
as frutas formam "ilhas" separadas. Uma fruta nova (ex.: taperebá) entra só adicionando exemplos.

---

## 2. RAG (Retrieval-Augmented Generation)

**Problema:** o LLM só sabe o que viu no treino. Ele não conhece as regras do seu jogo nem o que
aconteceu na partida de ontem, e pode **inventar** (alucinar).

**Solução:** antes de perguntar ao LLM, **recupere** os trechos relevantes dos seus documentos e
coloque no prompt. O LLM responde **com base neles**.

Na linguagem da Unidade 10:
- **Memória paramétrica:** o que o modelo aprendeu no treino e ficou nos pesos.
- **Memória não paramétrica:** a base externa pesquisável (aqui, o Neo4j com as fichas e os logs).

O RAG junta as duas. Como o transformer só prevê o próximo token a partir do contexto, se a
informação certa não estiver no contexto ele completa com padrões, e aí vem a alucinação.

```
1. INDEXAR (uma vez)    documentos → chunks → embeddings → índice
2. RECUPERAR (por vez)  pergunta → embedding → k trechos mais parecidos
3. GERAR                prompt = instrução + trechos + pergunta → LLM → resposta com fontes
```

### Decisões que mudam a qualidade

| Decisão | Efeito | No projeto |
|---|---|---|
| Tamanho do chunk | pequeno = preciso mas sem contexto; grande = contexto mas com ruído | começar com ~300 tokens e 50 de sobreposição |
| **k da recuperação** | poucos = pode faltar informação; muitos = ruído e prompt caro | começar com k = 4 |
| Modelo de embedding | define o que é "parecido" | MiniLM multilíngue (português) |
| Metadados no chunk | permitem filtrar e citar | `source`, fruta, tipo (ficha ou log) como propriedades do nó |
| Instrução do prompt | "responda só com base nos trechos; se não souber, diga" | reduz a alucinação |
| Citar fontes | permite conferir | cada trecho carrega `source` |

⚠️ **Não confunda:** o *k da recuperação* (quantos trechos buscar) **não tem nada a ver** com o
*top-k da amostragem* (seção 3). São nomes parecidos para coisas diferentes, e é uma pergunta clássica
de prova.

### RAG × MCP × fine-tuning

- **RAG:** muda **o que o modelo lê** na hora. Barato, atualiza na hora, cita fonte. Bom para fatos.
  O conhecimento é descartado quando a conversa acaba.
- **MCP:** protocolo para o modelo acessar ferramentas e fontes. Pode ser o **canal** de um RAG, mas
  não é um método de recuperação.
- **Fine-tuning:** muda **os pesos** do modelo. Caro, bom para estilo e formato, ruim para fatos que mudam.

Curiosidade: o **treino do YOLO na Etapa 1 é um fine-tuning**. Partimos de pesos pré-treinados no
COCO e ajustamos para fruta e bomba, exatamente a "tarefa extra" da Aula 5 da Unidade 4.

**No projeto:** dois usos de RAG.
1. **Fruti-pédia:** `knowledge/*.md` tem fichas (origem, sabor, usos, curiosidades). Ao cortar uma fruta,
   o jogo pergunta "conte uma curiosidade sobre o cupuaçu".
2. **Treinador:** `logs/partidas.jsonl` também é indexado. Pergunta: "por que a IA perdeu mais com
   temperature 1.5?" → o RAG recupera as partidas e o LLM analisa. Aqui o RAG trabalha sobre **dados
   gerados pelo próprio sistema**.

---

## 3. Temperature, top-k e top-p

### De onde vêm

A cada passo, o LLM dá uma **nota (logit)** para cada token possível do vocabulário. A função
**softmax** transforma as notas em probabilidades, e um token é **sorteado**. Os três parâmetros
mexem nesse sorteio. **Eles não mudam o que o modelo sabe, só como ele escolhe.**

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)          z = logits, T = temperature
```

### Exemplo numérico (o mesmo que o `sampling.js` usa nos testes)

Quatro opções com logits `z = [2,0  1,0  0,5  −1,0]`:

| T | Probabilidades | Leitura |
|---|---|---|
| 0,5 | **0,842**  0,114  0,042  0,002 | quase sempre a melhor: conservador |
| 1,0 | **0,609**  0,224  0,136  0,030 | distribuição original do modelo |
| 2,0 | **0,434**  0,263  0,205  0,097 | mais achatado: arrisca mais |
| → 0 | 1  0  0  0 | *greedy*: sempre a maior nota (determinístico) |

**Temperature** divide os logits antes do softmax. T baixo exagera as diferenças; T alto as apaga.

**Top-k** mantém só as **k opções de maior probabilidade**, zera o resto e renormaliza.
Com T = 1 e k = 2: `[0,609  0,224]` → `[0,731  0,269  0  0]`.
Problema: k é fixo. Quando o modelo está muito seguro, k = 40 ainda deixa 39 opções ruins; quando está
em dúvida, k = 2 corta opções boas.

**Top-p (nucleus)** mantém o **menor grupo cuja soma chega a p**. É adaptativo.
Com T = 1 e p = 0,9: acumulado 0,609 → 0,833 → **0,969 ≥ 0,9**, então ficam 3 opções →
`[0,628  0,231  0,140  0]`. Se o modelo estivesse seguro (ex.: 0,95 na primeira), sobraria só uma.

**Ordem usual:** temperature → top-k → top-p → sorteio. A ordem muda o resultado: top-p calculado
depois da temperature enxerga a distribuição já achatada ou afiada.

### Receitas

| Tarefa | temperature | top_p | top_k | Por quê |
|---|---|---|---|---|
| RAG factual, extrair dados | 0 a 0,3 | 0,9 | 40 | quer a resposta mais provável e fiel aos trechos |
| Explicar ou resumir | 0,5 a 0,7 | 0,9 | 40 | um pouco de variação no texto |
| Curiosidade criativa, narração | 0,8 a 1,0 | 0,95 | 50+ | variedade |
| Depurar ou reproduzir | 0 + seed fixa | n/a | n/a | mesma saída sempre |

Regra prática: mexa em **um de cada vez**. Temperature e top-p juntos se confundem.

É o mesmo experimento da Unidade 5 ("O céu é..." com temperature 0 dá quase sempre "azul"; com
temperature 2 e topK 10 aparecem "vasto", "cheio de estrelas"...), agora medido.

**No projeto (LLM):** `rag/experiments/` roda a **mesma pergunta 10 vezes** para cada combinação e
mede (a) quantas respostas diferentes saíram e (b) quantas inventaram algo fora dos trechos. Você vê
na prática a temperature alta aumentando variedade **e** alucinação.

---

## 4. A mesma matemática no agente do jogo

O agente não é um LLM, mas a decisão tem a **mesma forma**: várias opções (cortes candidatos), uma nota
para cada, e um sorteio. Por isso `sampling.js` implementa exatamente a seção 3.

| LLM | Agente do jogo |
|---|---|
| token | corte candidato (segmento a→b) |
| logit | score do corte: frutas atingidas, prioridade das que vão cair |
| token proibido (logit = −∞) | corte que atravessa bomba: **masking**, nunca é sorteado |
| T = 0 (greedy) | sempre o melhor corte: previsível, pode travar em situações ruins |
| T alta | explora cortes "piores" e às vezes descobre combos |
| top-k | considera só os k melhores cortes |
| top-p | considera os cortes até somar p de probabilidade |

**No projeto:** a Etapa 4 roda 50 partidas com seeds fixas para cada combinação (T ∈ {0; 0,3; 0,7;
1,0; 1,5}) e plota pontuação × temperature. Hipótese a testar: T = 0 tem a melhor média, e T alta
tem mais variância. É o mesmo raciocínio de escolher a temperature de um LLM, só que medido com
pontos do jogo.

---

## 5. Thresholds do YOLO não são sampling

O detector também tem "botões", mas eles **filtram**, não **sorteiam**:

- **confidence threshold** (ex.: 0,4): descarta caixas com pouca certeza. Mais alto = menos falsos
  positivos, mais frutas perdidas.
- **IoU do NMS** (ex.: 0,45): junta caixas sobrepostas do mesmo objeto.
- **max detections**: limite de caixas por frame.

São determinísticos: a mesma imagem sempre dá o mesmo resultado.

---

## 6. Na prática: APIs e parâmetros

- **Gemini Nano no Chrome (Prompt API, Unidade 5):** roda dentro do navegador, sem servidor.
  Aceita `temperature` e `topK`, **mas não top-p**. É o LLM do painel dentro do jogo (`src/ml/llm.js`).
- **Ollama (local):** `options: { temperature, top_k, top_p, seed }`. Os três funcionam, e por isso
  ele é o laboratório do experimento de sampling e a reserva quando o Gemini Nano não estiver ativo.
- **Transformers.js (Unidade 10):** só embeddings, rodando local no Node, sem chave de API.
- **Claude (API da Anthropic):** os modelos atuais **não aceitam mais** `temperature`, `top_p` nem
  `top_k` (a requisição volta com erro 400). O controle passa a ser pelo **prompt** e pelo **effort**
  (quanto o modelo raciocina). Vale como discussão de engenharia: em modelos com raciocínio, a
  variabilidade vem mais do "pensar" do que do sorteio de tokens. A Anthropic também não oferece
  modelo de embedding próprio, então o RAG usaria outro modelo de embedding de qualquer forma.
- **Outras APIs** costumam aceitar temperature e top_p, e às vezes top_k. Confira sempre a
  documentação do modelo exato.

---

## 7. Glossário

- **Logit:** nota bruta antes do softmax.
- **Softmax:** transforma notas em probabilidades que somam 1.
- **Greedy:** sempre a opção de maior probabilidade (T = 0).
- **Chunk:** pedaço de documento indexado.
- **Vector database / índice vetorial:** onde ficam os embeddings. Aqui é o Neo4j (como no curso);
  alternativas: FAISS, Chroma, pgvector.
- **Recall@k:** dos trechos certos, quantos apareceram entre os k recuperados (métrica do retriever).
- **Grounding:** a resposta estar apoiada nos trechos recuperados.
- **mAP50:** precisão média do detector com IoU ≥ 0,5 (métrica do YOLO).
- **IoU:** área de interseção ÷ área de união de duas caixas.

---

## 8. Ligação com o curso (Disciplina 1: Fundamentos de IA e LLMs)

| Unidade do curso | O que ensina | Onde aparece no Cortaí | O que vai além da aula |
|---|---|---|---|
| 4, Aula 1: integração em vez de treino | usar um modelo pronto | ponto de partida | aqui o YOLO **é treinado** (fine-tuning) com fruta e bomba |
| 4, Aula 2: Web Worker + createImageBitmap | IA fora da thread do jogo | `src/ml/worker.js` | igual |
| 4, Aula 3: tf.ready, warmup, fromPixels, resize, /255, expandDims, tf.tidy | pré-processamento | `src/ml/worker.js` | igual |
| 4, Aula 4: limiar de confiança, filtro de classe, centro da caixa | pós-processamento | `worker.js` + `planner.js` | NMS próprio (YOLO11); a ação é um **corte** (segmento), e não um clique |
| 4, Aula 4: "possíveis melhorias" | threshold dinâmico, intervalo de captura | `bench/` mede o efeito | rastreio com velocidade (prever onde a fruta vai estar) |
| 4, Aula 5: desafio "hackeie outro jogo" + estender o YOLO | aplicar em outro jogo, novas classes | **o projeto inteiro** | jogo próprio e dataset gerado automaticamente do estado do jogo |
| 5: Web AI, temperature e topK | Gemini Nano no Chrome | `src/ml/llm.js` | o mesmo sampling aplicado ao agente do jogo |
| 10, Aula 1: RAG, memória paramétrica × não paramétrica | conceito | seção 2 | RAG também sobre os **logs das partidas** |
| 10, Aulas 2 e 3: Transformers.js + Neo4j | embeddings e banco vetorial | `rag/` | embeddings de **imagem** (CLIP) e RAG com imagem + texto |

> Nível desta nota: ajustado à Disciplina 1. Usa os mesmos termos e ferramentas das unidades 4, 5 e
> 10 e acrescenta a matemática (softmax, cosseno) para você conseguir explicar sem consultar o
> material, como pedem os checkpoints do curso.
