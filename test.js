// Checagem mínima da lógica que a IA vai usar: corte, bomba, rótulos e seed.
import assert from 'node:assert';
import { Game, segHitsCircle } from './src/game.js';

assert(segHitsCircle({ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 50, y: 10, r: 12 }));
assert(!segHitsCircle({ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 50, y: 20, r: 12 }));
assert(!segHitsCircle({ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 50, y: 0, r: 12 })); // fora do segmento

const g = new Game(null, 1);
g.entities = [
  { kind: 'acai', bomb: false, r: 30, x: 100, y: 100, vx: 0, vy: 0, rot: 0, vr: 0, wait: 0 },
  { kind: 'bomba', bomb: true, r: 30, x: 400, y: 100, vx: 0, vy: 0, rot: 0, vr: 0, wait: 0 },
];
assert.deepStrictEqual(g.labels().map(l => l.cls), [0, 1]);
const l = g.labels()[0];
assert(Math.abs(l.cx - 100 / 960) < 1e-9 && Math.abs(l.w - 60 / 960) < 1e-9);

g.slice({ x: 50, y: 100 }, { x: 150, y: 100 });
assert.strictEqual(g.score, 1);
assert.strictEqual(g.pieces.length, 2);
assert(!g.over);
g.slice({ x: 350, y: 100 }, { x: 450, y: 100 });
assert(g.over, 'cortar bomba encerra o jogo');

// mesma seed => mesmos lançamentos
const run = seed => { const x = new Game(null, seed); for (let i = 0; i < 300; i++) x.update(1 / 60); return x.entities.map(e => e.kind + e.x.toFixed(2)).join(); };
assert.strictEqual(run(42), run(42));
assert.notStrictEqual(run(42), run(43));

// fruta perdida tira vida
const m = new Game(null, 5);
m.entities = [{ kind: 'acai', bomb: false, r: 30, x: 100, y: 600, vx: 0, vy: 10, rot: 0, vr: 0, wait: 0 }];
m.nextWave = 99;
m.update(0.01);
assert.strictEqual(m.lives, 2);
console.log('ok');
