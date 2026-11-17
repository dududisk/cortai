// node src/ml/agent.test.js
// Integração (percepção gt → tracker → planner → sampling → slice) e benchmark do critério da Etapa 3:
// modo gt, seeds fixas, média >= 40 pontos e bomba em < 10% das partidas.
import assert from 'node:assert';
import { Game } from '../game.js';
import { createAgent, runEpisode, labelsToDetections, AGENT_DEFAULTS } from './agent.js';

// game.labels() (YOLO normalizado) → Detection em px do canvas 960×540
{
  const dets = labelsToDetections([{ cls: 0, cx: 0.5, cy: 0.25, w: 0.075, h: 0.1 }, { cls: 1, cx: 0.1, cy: 0.9, w: 0.05, h: 0.05 }]);
  assert.deepStrictEqual(dets[0], { cls: 'fruta', x: 480, y: 135, w: 72, h: 54, score: 1 });
  assert.strictEqual(dets[1].cls, 'bomba');
}

// O agente só toca em labels, slice, time e over (nunca em game.entities), faz no máximo um corte
// por step e respeita o intervalo mínimo
{
  const real = new Game(null, 11);
  const touched = new Set();
  let slicesThisStep = 0;
  const realSlice = real.slice.bind(real);
  real.slice = (a, b) => { slicesThisStep++; return realSlice(a, b); };
  const game = new Proxy(real, {
    get(t, k) { touched.add(k); const v = t[k]; return typeof v === 'function' ? v.bind(t) : v; },
  });
  const agent = createAgent(game, { seed: 11, minIntervalMs: 250 });
  const cuts = [];
  for (let i = 0; i < 30 * 60 && !real.over; i++) {
    real.update(1 / 60);
    slicesThisStep = 0;
    const r = agent.step();
    assert(slicesThisStep <= 1, 'mais de um corte no mesmo step');
    assert.strictEqual(slicesThisStep, r ? 1 : 0);
    if (r) cuts.push(r.t);
  }
  assert(!touched.has('entities'), 'o agente leu game.entities');
  assert.deepStrictEqual([...touched].sort(), ['labels', 'over', 'slice', 'time']);
  assert(cuts.length > 20 && real.score > 20, `cortou pouco: ${cuts.length} cortes, ${real.score} pontos`);
  for (let i = 1; i < cuts.length; i++) assert(cuts[i] - cuts[i - 1] >= 0.25 - 1e-9, 'intervalo mínimo violado');
}

// Partida nova (relógio do jogo volta a 0) zera o estado do agente
{
  const game = new Game(null, 4);
  const agent = createAgent(game, { seed: 4 });
  for (let i = 0; i < 120; i++) { game.update(1 / 60); agent.step(); }
  assert(agent.tracker.tracks.length > 0 || agent.lastAction);
  game.reset(5);
  game.update(1 / 60);
  agent.step();
  assert(agent.tracker.tracks.every(t => t.born <= game.time + 1e-9 && t.id < 10), 'tracks da partida anterior sobraram');
}

// Modo 'external': detecções entregues por observe() (como fará o worker YOLO), com 10 Hz e latência
{
  const game = new Game(null, 21);
  const agent = createAgent(game, { seed: 21, perception: 'external' });
  const queue = [];
  let lastCap = -Infinity;
  while (!game.over && game.time < 60) {
    game.update(1 / 60);
    const now = game.time * 1000;
    if (now - lastCap >= 100 - 1e-6) { lastCap = now; queue.push({ dets: labelsToDetections(game.labels()), cap: now }); }
    while (queue.length && queue[0].cap + 60 <= now + 1e-6) { const q = queue.shift(); agent.observe(q.dets, q.cap); }
    agent.step(now);
  }
  assert(!(game.over && game.lives > 0), 'modo external cortou bomba');
  assert(game.score >= 40, `modo external (10 Hz, 60 ms de latência): ${game.score} pontos`);
}

// Mesma seed + mesmos parâmetros = mesma partida (o sorteio usa gerador com seed)
{
  const p = { temperature: 1.5, topK: 0, topP: 1, minIntervalMs: 600 };
  assert.deepStrictEqual(runEpisode(7, p, { maxTimeS: 40 }), runEpisode(7, p, { maxTimeS: 40 }));
}

// Benchmark do critério de pronto (parâmetros padrão)
{
  const SEEDS = Array.from({ length: 30 }, (_, i) => i + 1);
  const MAX_S = 120;
  const rows = SEEDS.map(seed => runEpisode(seed, {}, { maxTimeS: MAX_S }));
  const avg = rows.reduce((a, r) => a + r.score, 0) / rows.length;
  const bombRate = rows.filter(r => r.bombHit).length / rows.length;
  const { temperature, topK, topP, minIntervalMs } = AGENT_DEFAULTS;
  console.log(`\nBenchmark modo gt: ${rows.length} seeds, até ${MAX_S} s de jogo, T=${temperature} topK=${topK} topP=${topP}, ${minIntervalMs} ms entre cortes`);
  console.log('seed | pontos | bomba | vidas perdidas | duração (s)');
  for (const r of rows) console.log(`${String(r.seed).padStart(4)} | ${String(r.score).padStart(6)} | ${r.bombHit ? '  sim' : '  não'} | ${String(r.misses).padStart(14)} | ${r.durationS}`);
  console.log(`média ${avg.toFixed(1)} pontos, bomba em ${(bombRate * 100).toFixed(1)}% das partidas\n`);
  assert(avg >= 40, `média ${avg.toFixed(1)} < 40`);
  assert(bombRate < 0.1, `bomba em ${(bombRate * 100).toFixed(1)}% das partidas`);
}

console.log('agent ok');
