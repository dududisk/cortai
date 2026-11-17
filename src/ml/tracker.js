// Rastreador: associa Detection[] entre frames (vizinho mais próximo), estima a velocidade e prevê
// a posição daqui a Δt com a gravidade do jogo. Só enxerga Detection[], nunca game.entities.
//
// Detection = { cls: 'fruta' | 'bomba', x, y, w, h, score }      x,y = centro em px do canvas 960×540
// Track     = Detection & { id, vx, vy, age }                     vx,vy em px/s
//   extras: r (raio estimado), born (primeira detecção, s), t (instante do estado, s), seen (última
//   detecção, s), misses (frames sem detecção), cutAt (instante em que o agente mirou este track, s),
//   prior (nasceu na borda de baixo com velocidade de lançamento suposta).
//   age = quantas vezes o track foi reencontrado; age 0 = visto uma vez só, velocidade desconhecida.
import { W, H } from '../game.js';

export const GRAVITY = 900; // px/s², igual a src/game.js

// Porta de associação (px) = gateBase + velocidade de incerteza × Δt desde a última atualização.
// Porta justa evita troca de identidade quando uma bomba que sobe cruza uma que cai.
export const TRACKER_DEFAULTS = {
  gateBase: 30,        // px: ruído de posição tolerado
  gateSpeed: 300,      // px/s: erro de velocidade de um track já medido (age >= 1)
  gatePriorSpeed: 400, // px/s: track novo nascido embaixo (velocidade do prior "subindo")
  gateNewSpeed: 1100,  // px/s: track novo sem prior (velocidade desconhecida)
  priorBand: 120,      // px: detecção nova a esta distância da borda de baixo está subindo (lançamento)
  priorVy: -830,       // px/s: velocidade típica de lançamento ao aparecer (jogo: -712 a -946)
  posAlpha: 1,       // 1 = posição vem da detecção (gt é exato); < 1 suaviza ruído do YOLO
  velAlpha: 0.6,     // peso da velocidade medida contra a prevista
  maxCoastS: 0.35,   // s sem detecção até apagar o track
  gravity: GRAVITY,
};

// A caixa do rótulo é cortada na borda do canvas: uma fruta entrando por baixo vira uma caixa larga e
// baixa, com o centro deslocado. O objeto é um círculo, então o raio sai do lado maior da caixa e o
// centro verdadeiro é reconstruído a partir da borda que não foi cortada.
export function unclip(d, { width = W, height = H } = {}) {
  let { x, y } = d;
  const { w, h } = d;
  const r = Math.max(w, h) / 2;
  const e = 1; // px de tolerância para "encostou na borda"
  if (h < 0.92 * w) {
    if (y + h / 2 >= height - e && y - h / 2 > e) y = y - h / 2 + r;      // cortada embaixo
    else if (y - h / 2 <= e && y + h / 2 < height - e) y = y + h / 2 - r; // cortada em cima
  }
  if (w < 0.92 * h) {
    if (x + w / 2 >= width - e && x - w / 2 > e) x = x - w / 2 + r;
    else if (x - w / 2 <= e && x + w / 2 < width - e) x = x + w / 2 - r;
  }
  return { x, y, r };
}

// Estado do track daqui a dt segundos (movimento balístico). age 0 sem prior: fica parado.
export function predict(track, dt, g = GRAVITY) {
  if ((track.age < 1 && !track.prior) || dt === 0) return { x: track.x, y: track.y, vx: track.vx, vy: track.vy };
  return {
    x: track.x + track.vx * dt,
    y: track.y + track.vy * dt + 0.5 * g * dt * dt,
    vx: track.vx,
    vy: track.vy + g * dt,
  };
}

// Tempo (s) até a coordenada y do track chegar a yLimit descendo. Infinity se não chega.
export function timeToY(track, yLimit, g = GRAVITY) {
  if (track.y >= yLimit) return 0;
  // y + vy t + g t²/2 = yLimit  →  t = (-vy + sqrt(vy² + 2 g (yLimit - y))) / g
  return (-track.vy + Math.sqrt(track.vy * track.vy + 2 * g * (yLimit - track.y))) / g;
}

export class Tracker {
  constructor(opts = {}) {
    this.o = { ...TRACKER_DEFAULTS, ...opts };
    this.reset();
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
    this.t = null;
  }

  // detections: Detection[] do frame capturado no instante t (s). Devolve os tracks vivos.
  update(detections, t) {
    const o = this.o;
    const dets = detections.map(d => ({ ...d, ...unclip(d) }));

    // 1. associação gulosa: pares (track, detecção) da mesma classe, do mais perto ao mais longe
    const pairs = [];
    this.tracks.forEach((tr, ti) => {
      const dt = Math.max(0, t - tr.t);
      const p = predict(tr, dt, o.gravity);
      const gate = o.gateBase + dt * (tr.age >= 1 ? o.gateSpeed : tr.prior ? o.gatePriorSpeed : o.gateNewSpeed);
      dets.forEach((d, di) => {
        if (d.cls !== tr.cls) return;
        const dist = Math.hypot(d.x - p.x, d.y - p.y);
        if (dist <= gate) pairs.push({ ti, di, dist, p, dt });
      });
    });
    pairs.sort((a, b) => a.dist - b.dist);
    const usedT = new Set(), usedD = new Set();
    for (const { ti, di, p, dt } of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti); usedD.add(di);
      this.correct(this.tracks[ti], dets[di], p, dt, t);
    }

    // 2. tracks sem detecção seguem pela previsão até maxCoastS; os que saíram por baixo somem
    this.tracks.forEach((tr, ti) => { if (!usedT.has(ti)) tr.misses++; });
    this.tracks = this.tracks.filter(tr => {
      if (tr.misses === 0) return true;
      if (t - tr.seen > o.maxCoastS) return false;
      const p = predict(tr, t - tr.t, o.gravity);
      return !(p.vy > 0 && p.y - tr.r > H);
    });

    // 3. detecções novas viram tracks com age 0. Perto da borda de baixo é um lançamento: prior de
    //    velocidade subindo (e puxando para o meio da tela, como no spawnWave), trocado pela medida no
    //    próximo frame.
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      const prior = d.y > H - o.priorBand;
      this.tracks.push({
        id: this.nextId++, cls: d.cls, x: d.x, y: d.y, w: d.w, h: d.h, score: d.score ?? 1, r: d.r,
        vx: prior ? (W / 2 - d.x) * 0.4 : 0, vy: prior ? o.priorVy : 0, prior,
        age: 0, born: t, t, seen: t, misses: 0, cutAt: null,
      });
    });
    this.t = t;
    return this.tracks;
  }

  correct(tr, d, p, dt, t) {
    const o = this.o;
    if (dt > 0) {
      // velocidade média no intervalo + correção da gravidade = velocidade no fim do intervalo
      const vxm = (d.x - tr.x) / dt;
      const vym = (d.y - tr.y) / dt + 0.5 * o.gravity * dt;
      if (tr.age < 1) { tr.vx = vxm; tr.vy = vym; }
      else {
        tr.vx = p.vx + o.velAlpha * (vxm - p.vx);
        tr.vy = p.vy + o.velAlpha * (vym - p.vy);
      }
      tr.x = p.x + o.posAlpha * (d.x - p.x);
      tr.y = p.y + o.posAlpha * (d.y - p.y);
      tr.age++;
    } else {
      tr.x = d.x; tr.y = d.y;
    }
    tr.w = d.w; tr.h = d.h; tr.score = d.score ?? 1;
    tr.r = Math.max(tr.r * 0.7 + d.r * 0.3, d.r * 0.95); // raio estável, sem encolher na borda
    tr.t = t; tr.seen = t; tr.misses = 0;
  }

  // Cópias dos tracks com o estado previsto para o instante t (s): é o que o planner recebe.
  predicted(t) {
    return this.tracks.map(tr => ({ ...tr, ...predict(tr, Math.max(0, t - tr.t), this.o.gravity), t }));
  }
}
