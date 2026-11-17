// Agente que joga o Cortaí: percepção → rastreador → planejador → amostragem → game.slice(a, b).
// Só enxerga Detection[] (modo 'gt' = game.labels() convertido; modo 'external' = detecções entregues
// por agent.observe(), ex.: o worker YOLO). Nunca lê game.entities.
import { Game, W, H, CLASSES, rng } from '../game.js';
import { Tracker } from './tracker.js';
import { plan } from './planner.js';
import { sample } from './sampling.js';

export const AGENT_DEFAULTS = {
  perception: 'gt',         // 'gt' | 'external'
  temperature: 0.3,
  topK: 5,
  topP: 0.9,
  seed: 1,                  // seed do gerador do sorteio (partidas reproduzíveis)
  minIntervalMs: 200,       // intervalo mínimo entre dois cortes (ritmo de uma pessoa)
  perceptionIntervalMs: 0,  // modo gt: 0 = lê os rótulos a cada step; 100 = imita o worker (~10 Hz)
  leadMs: 0,                // prevê as posições para agora + leadMs (latência entre decidir e cortar)
  planner: {},              // sobrescreve PLANNER_DEFAULTS
  tracker: {},              // sobrescreve TRACKER_DEFAULTS
};

// game.labels() (YOLO normalizado: cls 0/1, cx, cy, w, h em [0,1]) → Detection[] em px do canvas
export function labelsToDetections(labels) {
  return labels.map(l => ({ cls: CLASSES[l.cls], x: l.cx * W, y: l.cy * H, w: l.w * W, h: l.h * H, score: 1 }));
}

export function createAgent(game, opts = {}) {
  const o = {
    ...AGENT_DEFAULTS, ...opts,
    planner: { ...opts.planner }, tracker: { ...opts.tracker },
  };
  const tracker = new Tracker(o.tracker);
  let rand, lastNow, lastPerception, lastCut;

  function reset() {
    tracker.reset();
    rand = rng(o.seed);
    lastNow = null;
    lastPerception = -Infinity;
    lastCut = -Infinity;
    agent.lastCandidates = [];
    agent.lastAction = null;
  }

  const clock = nowMs => (nowMs ?? game.time * 1000) / 1000;

  const agent = {
    options: o,
    tracker,
    lastCandidates: [], // Candidate[] do último planejamento (para desenhar/depurar)
    lastAction: null,   // último corte feito: { a, b, hits, bombs, score, index, t }
    reset,

    // Entrega detecções de um frame capturado em captureMs (use o instante da captura, não o da
    // resposta do worker: assim a latência entra na previsão). Usado no modo 'external'.
    observe(detections, captureMs) {
      const t = clock(captureMs);
      if (tracker.t != null && t < tracker.t) return; // frame atrasado, fora de ordem
      tracker.update(detections, t);
      lastPerception = t;
    },

    // Um passo de decisão. nowMs = relógio em ms (padrão: game.time * 1000, o relógio da física).
    // Faz no máximo um corte e respeita minIntervalMs. Devolve o corte feito ou null.
    step(nowMs) {
      const now = clock(nowMs);
      if (lastNow != null && now < lastNow) reset(); // relógio voltou: partida nova
      lastNow = now;
      if (game.over) return null;

      if (o.perception === 'gt' && now - lastPerception >= o.perceptionIntervalMs / 1000 - 1e-9) {
        agent.observe(labelsToDetections(game.labels()), now * 1000);
      }
      if (now - lastCut < o.minIntervalMs / 1000 - 1e-9) return null;

      const tAct = now + o.leadMs / 1000;
      const cands = plan(tracker.predicted(tAct), { ...o.planner, perceptionAge: tAct - (tracker.t ?? tAct) });
      agent.lastCandidates = cands;
      const i = sample(cands.map(c => c.score), o, rand);
      if (i < 0) return null;

      const c = cands[i];
      game.slice(c.a, c.b);
      lastCut = now;
      for (const tr of tracker.tracks) if (c.hits.includes(tr.id)) tr.cutAt = now;
      return (agent.lastAction = { ...c, index: i, t: now });
    },
  };
  reset();
  return agent;
}

// Partida sem tela (Node), para testes e benchmark. Devolve uma linha no formato de
// logs/partidas.jsonl. O jogo não tem fim natural se a IA não erra, então há um limite de tempo.
export function runEpisode(seed, agentOpts = {}, { maxTimeS = 120, dt = 1 / 60 } = {}) {
  const game = new Game(null, seed);
  const agent = createAgent(game, { seed, ...agentOpts });
  while (!game.over && game.time < maxTimeS) {
    game.update(dt);
    agent.step();
  }
  const { temperature, topK, topP, perception } = agent.options;
  return {
    seed, perception, params: { temperature, topK, topP },
    score: game.score,
    bombHit: game.over && game.lives > 0, // fim por bomba (fim por vidas deixa lives = 0)
    misses: 3 - game.lives,
    durationS: +game.time.toFixed(2),
  };
}
