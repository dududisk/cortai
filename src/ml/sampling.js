// Amostragem sobre as notas (logits) dos cortes candidatos: a mesma matemática de um LLM
// (docs/NOTA-IA.md, seções 3 e 4). Ordem fixa: temperature → top-k → top-p → sorteio.
//
// - score -Infinity (ou NaN) = opção proibida (masking): probabilidade 0, nunca é sorteada.
// - temperature <= 0 = greedy (argmax). Empate: fica o menor índice.
// - topK <= 0 ou >= nº de opções válidas = desligado.
// - topP >= 1 (ou <= 0) = desligado.

const valid = z => Number.isFinite(z) || z === Infinity;

// Índice do maior score válido, ou -1 se não houver nenhum.
export function argmax(scores) {
  let best = -1;
  for (let i = 0; i < scores.length; i++) {
    if (valid(scores[i]) && (best < 0 || scores[i] > scores[best])) best = i;
  }
  return best;
}

// softmax(z / T) só sobre as opções válidas; as proibidas ficam com 0.
export function softmax(scores, temperature = 1) {
  const p = new Array(scores.length).fill(0);
  const best = argmax(scores);
  if (best < 0) return p;
  if (!(temperature > 0) || scores[best] === Infinity) { p[best] = 1; return p; }
  const zmax = scores[best];
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    if (!valid(scores[i])) continue;
    p[i] = Math.exp((scores[i] - zmax) / temperature); // subtrai o máximo: estável numericamente
    sum += p[i];
  }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return p;
}

// Índices com probabilidade > 0, do mais provável ao menos provável (empate: menor índice).
function ranked(p) {
  return p.map((v, i) => i).filter(i => p[i] > 0).sort((i, j) => p[j] - p[i] || i - j);
}

function renormalize(p, keep) {
  const out = new Array(p.length).fill(0);
  let sum = 0;
  for (const i of keep) sum += p[i];
  for (const i of keep) out[i] = p[i] / sum;
  return out;
}

// Top-k: mantém só as k opções mais prováveis e renormaliza.
export function topKFilter(p, k) {
  const order = ranked(p);
  if (!(k > 0) || k >= order.length) return p.slice();
  return renormalize(p, order.slice(0, Math.floor(k)));
}

// Top-p (nucleus): mantém o menor grupo (dos mais prováveis) cuja soma chega a p.
export function topPFilter(p, topP) {
  if (!(topP > 0) || topP >= 1) return p.slice();
  const order = ranked(p);
  const keep = [];
  let cum = 0;
  for (const i of order) {
    keep.push(i);
    cum += p[i];
    if (cum >= topP - 1e-12) break;
  }
  return renormalize(p, keep);
}

// Distribuição final usada no sorteio (útil para testes e para desenhar na tela).
export function probabilities(scores, { temperature = 1, topK = 0, topP = 1 } = {}) {
  let p = softmax(scores, temperature);
  p = topKFilter(p, topK);
  p = topPFilter(p, topP);
  return p;
}

// sample(scores, { temperature, topK, topP }, rand) → índice sorteado (-1 se tudo for proibido).
// rand: função que devolve um número em [0, 1). Use um gerador com seed para reproduzir partidas.
export function sample(scores, { temperature = 1, topK = 0, topP = 1 } = {}, rand = Math.random) {
  if (!(temperature > 0)) return argmax(scores); // greedy: não consome o gerador
  const p = probabilities(scores, { temperature, topK, topP });
  let last = -1;
  let u = rand();
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue;
    last = i;
    if ((u -= p[i]) < 0) return i;
  }
  return last; // arredondamento: cai na última opção válida
}
