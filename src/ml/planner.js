// Planejador: a partir dos tracks (já previstos para o instante do corte) gera cortes candidatos e dá
// uma nota (logit) a cada um. Corte que passa perto de bomba recebe -Infinity (masking): a amostragem
// nunca o sorteia, seja qual for a temperature.
//
// Candidate = { a: {x, y}, b: {x, y}, hits: number[], bombs: number, score: number }
//   hits  = ids dos tracks de fruta que o corte atravessa
//   bombs = quantas bombas ficam dentro da margem (> 0 ⇒ score = -Infinity)
//   extra: clearance = folga (px) até a margem da bomba mais próxima (negativa se mascarado)
import { W, H } from '../game.js';
import { GRAVITY, timeToY } from './tracker.js';

export const PLANNER_DEFAULTS = {
  bombMargin: 24,      // px além do raio da bomba que o corte não pode invadir (máscara dura)
  // margem extra por segundo de informação velha (latência/coast), conforme a confiança na velocidade:
  bombStaleSpeed: 150, // px/s: bomba rastreada há 2+ frames
  bombYoungSpeed: 500, // px/s: bomba com uma medida de velocidade só (age 1, ruidosa)
  bombNewSpeed: 1000,  // px/s: bomba vista uma vez só (age 0, velocidade desconhecida)
  bombSoft: 60,        // px: faixa além da margem onde o corte perde nota (prefere passar longe)
  bombSoftWeight: 0.6, // nota perdida no limite da margem
  hitShrink: 5,        // px: o corte tem de passar a (r - hitShrink) do centro previsto
  hitStaleSpeed: 150,  // px/s: hitShrink cresce com a idade da informação da fruta
  extend: 18,          // px: o corte passa além da borda da fruta (gesto visível)
  maxLen: 420,         // px: corte mais longo permitido (gesto de uma pessoa)
  angles: 8,           // orientações testadas no corte de uma fruta só
  yMax: H - 22,        // nenhum ponto do corte abaixo disso: bomba que ainda não apareceu nasce embaixo
  unseenRise: 1000,    // px/s: quanto uma bomba invisível no último frame pode ter subido desde então
  perceptionAge: 0,    // s entre o último frame percebido e o corte (o agente preenche)
  reactionS: 0,        // s: fruta só vira alvo depois de rastreada por este tempo (tempo de reação)
  fruitValue: 1,       // nota por fruta atravessada
  urgencyWeight: 1.5,  // nota extra para fruta prestes a cair
  urgencyTau: 0.5,     // s: escala da urgência, exp(-tempo até sair / tau)
  exclude: null,       // Set de ids que não são alvo (ex.: frutas já cortadas esperando confirmação)
};

// distância de um ponto c ao segmento ab
export function segDist(a, b, c) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((c.x - a.x) * dx + (c.y - a.y) * dy) / len2));
  return Math.hypot(a.x + t * dx - c.x, a.y + t * dy - c.y);
}

// Liang-Barsky: recorta o segmento ao retângulo [x0,x1]×[y0,y1]. null se ficar de fora.
function clip(a, b, x0, y0, x1, y1) {
  const dx = b.x - a.x, dy = b.y - a.y;
  let t0 = 0, t1 = 1;
  for (const [p, q] of [[-dx, a.x - x0], [dx, x1 - a.x], [-dy, a.y - y0], [dy, y1 - a.y]]) {
    if (p === 0) { if (q < 0) return null; continue; }
    const r = q / p;
    if (p < 0) { if (r > t1) return null; if (r > t0) t0 = r; }
    else { if (r < t0) return null; if (r < t1) t1 = r; }
  }
  return [{ x: a.x + t0 * dx, y: a.y + t0 * dy }, { x: a.x + t1 * dx, y: a.y + t1 * dy }];
}

// Urgência em [0, 1]: 1 = a fruta está saindo do alcance do corte agora.
export function urgency(f, o = PLANNER_DEFAULTS) {
  if (f.age < 1) return 0; // velocidade desconhecida: acabou de aparecer subindo
  const deadline = o.yMax + f.r - (o.hitShrink ?? 0); // abaixo disso não dá mais para alcançar
  return Math.exp(-timeToY(f, deadline, GRAVITY) / o.urgencyTau);
}

// tracks: Track[] previstos para o instante do corte (t). Devolve um Candidate por conjunto de frutas
// atingidas (o melhor corte geométrico de cada conjunto), incluindo os mascarados.
export function plan(tracks, opts = {}) {
  const o = { ...PLANNER_DEFAULTS, ...opts };
  // bombas nascem abaixo da tela (y = H + 50) e só viram detecção quando aparecem; com percepção
  // velha, uma bomba ainda não vista pode já estar na faixa de baixo → o teto do corte sobe
  o.yMax -= o.unseenRise * Math.max(0, o.perceptionAge);
  const stale = tr => Math.max(0, (tr.t ?? 0) - (tr.seen ?? tr.t ?? 0));
  const pending = tr => tr.cutAt != null && tr.seen <= tr.cutAt;
  const reacted = tr => (tr.t ?? 0) - (tr.born ?? -Infinity) >= o.reactionS - 1e-9;
  const fruits = tracks.filter(tr => tr.cls === 'fruta' && tr.misses === 0 && !pending(tr) && reacted(tr) && !o.exclude?.has(tr.id))
    .map(tr => ({ ...tr, reach: tr.r - o.hitShrink - o.hitStaleSpeed * stale(tr), u: urgency(tr, o) }))
    .filter(f => f.reach > 2);
  const bombs = tracks.filter(tr => tr.cls === 'bomba')
    .map(tr => {
      const speed = tr.age < 1 ? o.bombNewSpeed : tr.age < 2 ? o.bombYoungSpeed : o.bombStaleSpeed;
      return { ...tr, keep: tr.r + o.bombMargin + speed * stale(tr) };
    });
  if (!fruits.length) return [];

  const best = new Map(); // chave = ids atingidos → melhor candidato
  const consider = (a0, b0) => {
    const seg = clip(a0, b0, 0, 0, W, o.yMax);
    if (!seg) return;
    const [a, b] = seg;
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    if (len < 3 || len > o.maxLen) return;
    const hits = fruits.filter(f => segDist(a, b, f) <= f.reach);
    if (!hits.length) return;
    let clearance = Infinity, nb = 0;
    for (const bm of bombs) {
      const c = segDist(a, b, bm) - bm.keep;
      if (c < 0) nb++;
      clearance = Math.min(clearance, c);
    }
    let score = -Infinity;
    if (nb === 0) {
      score = hits.reduce((s, f) => s + o.fruitValue + o.urgencyWeight * f.u, 0);
      if (clearance < o.bombSoft) score -= o.bombSoftWeight * (1 - clearance / o.bombSoft);
    }
    const key = hits.map(f => f.id).sort((x, y) => x - y).join(',');
    const prev = best.get(key);
    if (!prev || score > prev.score || (score === -Infinity && prev.score === -Infinity && clearance > prev.clearance)) {
      best.set(key, { a, b, hits: hits.map(f => f.id), bombs: nb, score, clearance });
    }
  };

  // segmento centrado em p, na direção (ux, uy), meia-largura L
  const centered = (p, ux, uy, L) => consider({ x: p.x - ux * L, y: p.y - uy * L }, { x: p.x + ux * L, y: p.y + uy * L });

  for (const f of fruits) {
    // ponto de mira: o centro, ou o ponto mais baixo permitido se a fruta já está abaixo de yMax
    const p = { x: f.x, y: Math.min(f.y, o.yMax) };
    if (f.y - p.y > f.reach) continue;
    const full = f.r + o.extend, short = Math.max(4, f.r * 0.5);
    for (let k = 0; k < o.angles; k++) {
      const th = (k * Math.PI) / o.angles;
      centered(p, Math.cos(th), Math.sin(th), full);
      centered(p, Math.cos(th), Math.sin(th), short);
    }
    // bomba perto: corte perpendicular à direção da bomba, pelo lado da fruta mais longe dela
    let near = null, nd = Infinity;
    for (const bm of bombs) { const d = Math.hypot(f.x - bm.x, f.y - bm.y); if (d < nd) { nd = d; near = bm; } }
    if (near && nd > 0 && nd < 4 * (f.r + near.keep)) {
      const n = { x: (f.x - near.x) / nd, y: (f.y - near.y) / nd };
      const off = 0.8 * f.reach;
      const q = { x: f.x + n.x * off, y: Math.min(f.y + n.y * off, o.yMax) };
      centered(q, -n.y, n.x, full);
      centered(q, -n.y, n.x, short);
    }
  }

  // pares de frutas: segmento pelos dois centros (e versões deslocadas para os lados)
  for (let i = 0; i < fruits.length; i++) {
    for (let j = i + 1; j < fruits.length; j++) {
      const f = fruits[i], g = fruits[j];
      const d = Math.hypot(g.x - f.x, g.y - f.y);
      if (d < 1 || d > o.maxLen) continue;
      const u = { x: (g.x - f.x) / d, y: (g.y - f.y) / d }, n = { x: -u.y, y: u.x };
      const side = 0.7 * Math.min(f.reach, g.reach);
      for (const s of [0, side, -side]) {
        for (const ext of [[f.r + o.extend, g.r + o.extend], [0, 0]]) {
          consider(
            { x: f.x - u.x * ext[0] + n.x * s, y: f.y - u.y * ext[0] + n.y * s },
            { x: g.x + u.x * ext[1] + n.x * s, y: g.y + u.y * ext[1] + n.y * s },
          );
        }
      }
    }
  }

  return [...best.values()].sort((p, q) => q.score - p.score);
}
