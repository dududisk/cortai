// node src/ml/tracker.test.js
// O teste pode olhar game.entities para conferir o rastreador (o agente não pode).
import assert from 'node:assert';
import { Game, W, H } from '../game.js';
import { Tracker, unclip, predict, timeToY, GRAVITY } from './tracker.js';
import { labelsToDetections } from './agent.js';

// unclip: fruta entrando por baixo (r = 36, centro em y = 560) vira caixa baixa encostada na borda
{
  const y1 = 560 - 36, box = { x: 300, y: (y1 + H) / 2, w: 72, h: H - y1 };
  const u = unclip(box);
  assert.strictEqual(u.r, 36);
  assert(Math.abs(u.y - 560) < 1e-9, `centro reconstruído ${u.y}`);
  const inteira = unclip({ x: 300, y: 200, w: 72, h: 72 });
  assert.deepStrictEqual(inteira, { x: 300, y: 200, r: 36 });
}

// previsão balística: y(t) = y + vy t + g t²/2
{
  const p = predict({ x: 100, y: 300, vx: 50, vy: -400, age: 3 }, 0.5);
  assert(Math.abs(p.x - 125) < 1e-9 && Math.abs(p.y - (300 - 200 + 0.5 * GRAVITY * 0.25)) < 1e-9);
  assert.strictEqual(p.vy, -400 + GRAVITY * 0.5);
  const parado = predict({ x: 100, y: 300, vx: 0, vy: 0, age: 0 }, 0.5);
  assert.strictEqual(parado.y, 300, 'age 0 sem prior não tem velocidade: não extrapola');
  // tempo até descer a y = 540 partindo de y = 540 - 450 com vy = 0: sqrt(2·450/900) = 1 s
  assert(Math.abs(timeToY({ y: 90, vy: 0 }, 540) - 1) < 1e-9);
}

// Velocidade estimada a partir de dois frames de um objeto em queda livre
{
  const tr = new Tracker();
  const pos = t => ({ x: 200 + 80 * t, y: 400 - 600 * t + 0.5 * GRAVITY * t * t });
  const det = t => [{ cls: 'fruta', ...pos(t), w: 70, h: 70, score: 1 }];
  tr.update(det(0), 0);
  tr.update(det(0.05), 0.05);
  tr.update(det(0.1), 0.1);
  const [t0] = tr.tracks;
  assert.strictEqual(tr.tracks.length, 1);
  assert(Math.abs(t0.vx - 80) < 1e-6, `vx ${t0.vx}`);
  assert(Math.abs(t0.vy - (-600 + GRAVITY * 0.1)) < 1e-6, `vy ${t0.vy}`);
  const [p] = tr.predicted(0.4);
  assert(Math.abs(p.x - pos(0.4).x) < 1e-6 && Math.abs(p.y - pos(0.4).y) < 1e-6, 'previsão 0,3 s à frente');
}

// Duas bombas se cruzando (uma sobe, outra cai) mantêm os ids, mesmo perdendo um frame de uma delas
{
  const tr = new Tracker();
  const up = t => ({ cls: 'bomba', x: 400 + 30 * t, y: 470 - 700 * t + 450 * t * t, w: 68, h: 68 });
  const down = t => ({ cls: 'bomba', x: 440 - 30 * t, y: 250 + 300 * t + 450 * t * t, w: 68, h: 68 });
  const ids = new Map();
  for (let k = 0; k <= 6; k++) {
    const t = k * 0.1;
    const dets = k === 3 ? [up(t)] : [up(t), down(t)]; // frame 3: a que cai não foi detectada
    tr.update(dets, t);
    for (const [name, f] of [['up', up], ['down', down]]) {
      if (k === 3 && name === 'down') continue;
      const p = f(t);
      const near = tr.tracks.reduce((a, b) => (Math.hypot(a.x - p.x, a.y - p.y) < Math.hypot(b.x - p.x, b.y - p.y) ? a : b));
      if (!ids.has(name)) ids.set(name, near.id);
      assert.strictEqual(near.id, ids.get(name), `troca de identidade no frame ${k} (${name})`);
    }
  }
}

// Jogo real: rastrear 20 s a 60 fps, prever 0,1 s à frente e comparar com a posição verdadeira
{
  const game = new Game(null, 3);
  game.lives = Infinity; // ninguém corta: sem isso o jogo acaba na 3ª fruta perdida
  const tr = new Tracker();
  const errs = [];
  let pending = [];
  for (let i = 0; i < 20 * 60; i++) {
    game.update(1 / 60);
    const t = game.time;
    tr.update(labelsToDetections(game.labels()), t);
    // confere previsões feitas 0,1 s atrás (6 frames)
    pending = pending.filter(q => {
      if (t < q.t - 1e-9) return true;
      const e = game.entities.reduce((a, b) => (Math.hypot(a.x - q.x, a.y - q.y) < Math.hypot(b.x - q.x, b.y - q.y) ? a : b), game.entities[0]);
      if (e) errs.push(Math.hypot(e.x - q.x, e.y - q.y));
      return false;
    });
    if (i % 10 === 0) {
      for (const p of tr.predicted(t + 0.1)) if (p.age >= 2 && p.misses === 0 && p.y < H - 40) pending.push({ x: p.x, y: p.y, t: t + 0.1 });
    }
  }
  errs.sort((a, b) => a - b);
  const median = errs[errs.length >> 1];
  assert(errs.length > 50, `poucas amostras: ${errs.length}`);
  assert(median < 3, `erro mediano da previsão de 0,1 s: ${median.toFixed(2)} px`);
  console.log(`tracker: previsão 0,1 s à frente no jogo, ${errs.length} amostras, erro mediano ${median.toFixed(2)} px`);
}

assert.strictEqual(W, 960);
console.log('tracker ok');
