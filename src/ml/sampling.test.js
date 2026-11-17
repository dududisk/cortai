// node src/ml/sampling.test.js
// Reproduz o exemplo numérico da docs/NOTA-IA.md, seção 3 (z = [2, 1, 0.5, -1]).
import assert from 'node:assert';
import { rng } from '../game.js';
import { sample, probabilities, softmax, argmax } from './sampling.js';

const z = [2, 1, 0.5, -1];
const TOL = 1e-3;
const close = (got, want, msg) => {
  assert.strictEqual(got.length, want.length, msg);
  got.forEach((g, i) => assert(Math.abs(g - want[i]) <= TOL, `${msg}: p[${i}] = ${g.toFixed(4)}, esperado ${want[i]}`));
};

// Temperature (tabela da seção 3)
close(softmax(z, 0.5), [0.842, 0.114, 0.042, 0.002], 'T = 0.5');
close(softmax(z, 1), [0.609, 0.224, 0.136, 0.030], 'T = 1');
close(softmax(z, 2), [0.434, 0.263, 0.205, 0.097], 'T = 2');
close(softmax(z, 0), [1, 0, 0, 0], 'T = 0 (greedy)');

// Top-k = 2 com T = 1: [0.609 0.224] → [0.731 0.269 0 0]
close(probabilities(z, { temperature: 1, topK: 2 }), [0.731, 0.269, 0, 0], 'T = 1, top-k = 2');

// Top-p = 0.9 com T = 1: acumulado 0.609 → 0.833 → 0.969 ≥ 0.9, ficam 3 → [0.628 0.231 0.140 0]
close(probabilities(z, { temperature: 1, topP: 0.9 }), [0.628, 0.231, 0.140, 0], 'T = 1, top-p = 0.9');

// Modelo seguro (0.95 na primeira) → top-p = 0.9 deixa só uma opção
const seguro = [Math.log(0.95), Math.log(0.03), Math.log(0.02)];
close(probabilities(seguro, { temperature: 1, topP: 0.9 }), [1, 0, 0], 'top-p adaptativo');

// A ordem importa: top-p depois de uma T alta (achatado) mantém mais opções que depois de T baixa
// T = 0.5: 0.842 → 0.956 (2 opções); T = 2: 0.434 → 0.697 → 0.902 (3); T = 4: 0.340 → 0.605 → 0.839 → 1 (4)
const nucleo = T => probabilities(z, { temperature: T, topP: 0.9 }).filter(p => p > 0).length;
assert.strictEqual(nucleo(0.5), 2);
assert.strictEqual(nucleo(2), 3);
assert.strictEqual(nucleo(4), 4);

// T = 0 é argmax, mesmo com -Infinity, e não consome o gerador
let chamadas = 0;
const contador = () => { chamadas++; return 0.99; };
assert.strictEqual(sample(z, { temperature: 0 }, contador), 0);
assert.strictEqual(sample([-Infinity, 3, 7, -Infinity, 7], { temperature: 0 }, contador), 2, 'empate: menor índice');
assert.strictEqual(chamadas, 0);
assert.strictEqual(argmax([-Infinity, -Infinity]), -1);
assert.strictEqual(sample([-Infinity, -Infinity], { temperature: 1 }, Math.random), -1, 'tudo proibido → -1');
assert.strictEqual(sample([], { temperature: 1 }, Math.random), -1);

// Frequências do sorteio batem com as probabilidades (gerador com seed: resultado reproduzível)
function freq(scores, opts, n = 200000, seed = 1) {
  const rand = rng(seed), f = new Array(scores.length).fill(0);
  for (let i = 0; i < n; i++) f[sample(scores, opts, rand)]++;
  return f.map(c => c / n);
}
const f1 = freq(z, { temperature: 1 });
softmax(z, 1).forEach((p, i) => assert(Math.abs(f1[i] - p) < 0.005, `frequência T = 1, opção ${i}`));
const fp = freq(z, { temperature: 1, topP: 0.9 });
assert.strictEqual(fp[3], 0, 'top-p = 0.9 nunca sorteia a 4ª opção');
assert(Math.abs(fp[0] - 0.628) < 0.005);
const fk = freq(z, { temperature: 1, topK: 2 });
assert.strictEqual(fk[2] + fk[3], 0, 'top-k = 2 nunca sorteia a 3ª e a 4ª');
assert(Math.abs(fk[0] - 0.731) < 0.005);

// -Infinity (masking) nunca é sorteado, nem com temperature alta
const mascarado = [-Infinity, 0.1, -Infinity, 0, -Infinity];
const fm = freq(mascarado, { temperature: 50 }, 50000, 7);
assert.strictEqual(fm[0] + fm[2] + fm[4], 0, '-Infinity sorteado!');
assert(Math.abs(fm[1] - 0.5) < 0.02 && Math.abs(fm[3] - 0.5) < 0.02, 'T alta achata as válidas');
// gerador nos extremos [0, 1) também respeita a máscara
assert.strictEqual(sample(mascarado, { temperature: 1 }, () => 0), 1);
assert.strictEqual(sample(mascarado, { temperature: 1 }, () => 0.999999999), 3);

// top-k = 1 equivale ao greedy
assert.strictEqual(sample(z, { temperature: 3, topK: 1 }, Math.random), 0);

// mesma seed → mesma sequência de escolhas
const seq = seed => { const r = rng(seed); return Array.from({ length: 50 }, () => sample(z, { temperature: 1 }, r)).join(); };
assert.strictEqual(seq(3), seq(3));

console.log('sampling ok');
