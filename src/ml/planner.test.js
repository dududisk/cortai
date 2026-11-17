// node src/ml/planner.test.js
import assert from 'node:assert';
import { H, rng } from '../game.js';
import { plan, segDist, urgency, PLANNER_DEFAULTS } from './planner.js';
import { sample } from './sampling.js';

let nextId = 1;
const fruta = (x, y, extra = {}) => ({ id: nextId++, cls: 'fruta', x, y, r: 36, w: 72, h: 72, vx: 0, vy: 0, age: 3, t: 0, seen: 0, born: -1, misses: 0, cutAt: null, ...extra });
const bomba = (x, y, extra = {}) => ({ ...fruta(x, y, extra), cls: 'bomba', r: 34, w: 68, h: 68, ...extra });
const finite = cs => cs.filter(c => c.score > -Infinity);
const key = c => [...c.hits].sort((a, b) => a - b).join(',');

// 1 fruta sozinha: o melhor corte passa por ela, dentro da tela e acima de yMax
{
  const f = fruta(300, 300);
  const cs = plan([f]);
  assert.strictEqual(cs.length, 1);
  const [c] = cs;
  assert.deepStrictEqual(c.hits, [f.id]);
  assert.strictEqual(c.bombs, 0);
  assert(c.score > 0 && Number.isFinite(c.score));
  assert(segDist(c.a, c.b, f) <= f.r);
  for (const p of [c.a, c.b]) assert(p.x >= 0 && p.x <= 960 && p.y >= 0 && p.y <= PLANNER_DEFAULTS.yMax + 1e-9);
}

// Mais frutas = nota maior: 3 alinhadas → o melhor corte pega as 3
{
  const fs = [fruta(200, 250), fruta(300, 250), fruta(400, 250), fruta(700, 100)];
  const [best] = plan(fs);
  assert.strictEqual(best.hits.length, 3, `melhor corte pegou ${best.hits.length}`);
  assert(best.score > 2.9);
}

// MASKING: qualquer corte que passe a menos de r + margem de uma bomba vale -Infinity
{
  const a = fruta(200, 300), b = fruta(400, 300), bm = bomba(300, 300);
  const cs = plan([a, b, bm]);
  const keep = bm.r + PLANNER_DEFAULTS.bombMargin;
  for (const c of cs) {
    const near = segDist(c.a, c.b, bm) < keep;
    assert.strictEqual(near, c.score === -Infinity, `corte ${key(c)} a ${segDist(c.a, c.b, bm).toFixed(1)} px da bomba, score ${c.score}`);
    assert.strictEqual(c.bombs > 0, near);
  }
  const par = cs.find(c => c.hits.length === 2);
  assert(par && par.score === -Infinity, 'cortar as duas frutas passando pela bomba tem de ser proibido');
  assert.strictEqual(finite(cs).length, 2, 'as frutas sozinhas continuam permitidas');
  // e a amostragem nunca escolhe o mascarado, nem com temperature alta
  const rand = rng(5), scores = cs.map(c => c.score);
  for (let i = 0; i < 5000; i++) assert(cs[sample(scores, { temperature: 10 }, rand)].score > -Infinity);
}

// Fruta colada na bomba: ainda há corte seguro (perpendicular, pelo lado de fora)
{
  const f = fruta(300, 300), bm = bomba(372, 300);
  const ok = finite(plan([f, bm]));
  assert(ok.length === 1, 'deveria achar um corte seguro');
  assert(segDist(ok[0].a, ok[0].b, bm) >= bm.r + PLANNER_DEFAULTS.bombMargin);
  assert(segDist(ok[0].a, ok[0].b, f) <= f.r);
}

// Fruta escondida atrás da bomba: nenhum corte permitido
{
  const cs = plan([fruta(300, 300), bomba(310, 300)]);
  assert(cs.length > 0 && finite(cs).length === 0);
}

// Bomba vista uma vez só e com informação velha ganha margem maior (velocidade desconhecida)
{
  const f = fruta(300, 300, { t: 0.1, seen: 0.1 });
  // margem: age 0 → 34 + 24 + 1000·0,1 = 158 px; age 5 → 34 + 24 + 150·0,1 = 73 px.
  // O corte mais afastado que ainda pega a fruta fica a 120 + 31 = 151 px da bomba.
  assert(finite(plan([f, bomba(420, 300, { age: 0, t: 0.1, seen: 0 })])).length === 0, 'bomba nova a 120 px com 0,1 s de atraso: arriscado');
  assert(finite(plan([f, bomba(420, 300, { age: 5, t: 0.1, seen: 0 })])).length === 1, 'bomba rastreada a 120 px: ok');
}

// Prioridade: fruta caindo perto do fundo vale mais que fruta subindo
{
  const caindo = fruta(200, 420, { vy: 500 }), subindo = fruta(700, 300, { vy: -400 });
  assert(urgency(caindo) > 0.5 && urgency(subindo) < 0.1, `${urgency(caindo)} ${urgency(subindo)}`);
  const [best] = plan([subindo, caindo]);
  assert.deepStrictEqual(best.hits, [caindo.id]);
}

// Faixa de baixo: corte nunca desce abaixo de yMax (bomba ainda invisível nasce em y = H + 50)
{
  const f = fruta(500, H - 10); // centro abaixo de yMax, mas ainda alcançável pela borda de cima
  const cs = plan([f]);
  assert(cs.length === 1);
  for (const p of [cs[0].a, cs[0].b]) assert(p.y <= PLANNER_DEFAULTS.yMax + 1e-9);
  // com percepção velha o teto sobe (unseenRise × idade): a mesma fruta deixa de ser alcançável
  assert.strictEqual(plan([f], { perceptionAge: 0.05 }).length, 0);
}

// Só fruta vista agora vira alvo: track em coast (misses > 0) ou já cortado (cutAt) não
{
  assert.strictEqual(plan([fruta(300, 300, { misses: 1 })]).length, 0);
  assert.strictEqual(plan([fruta(300, 300, { cutAt: 0, seen: 0 })]).length, 0);
  assert.strictEqual(plan([fruta(300, 300, { cutAt: 0, seen: 0.02, t: 0.02 })]).length, 1, 'visto depois do corte: o corte errou, tenta de novo');
  assert.strictEqual(plan([fruta(300, 300, { born: 0, t: 0.1 })], { reactionS: 0.2 }).length, 0, 'tempo de reação');
}

// Um candidato por conjunto de frutas (sem duplicar opções na amostragem)
{
  const cs = plan([fruta(200, 200), fruta(320, 220), fruta(600, 300)]);
  const keys = cs.map(key);
  assert.strictEqual(new Set(keys).size, keys.length);
}

console.log('planner ok');
