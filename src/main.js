import { Game, W, H, loadAssets } from './game.js';

const canvas = document.querySelector('canvas');
canvas.width = W;
canvas.height = H;
await loadAssets();

// window.game é a porta de entrada da IA: game.slice(a, b), game.entities, game.labels()
const game = window.game = new Game(canvas);

const toCanvas = ev => {
  const r = canvas.getBoundingClientRect();
  return { x: (ev.clientX - r.left) * W / r.width, y: (ev.clientY - r.top) * H / r.height };
};
let last = null;
canvas.addEventListener('pointerdown', ev => {
  if (game.over) game.reset();
  last = toCanvas(ev);
  canvas.setPointerCapture(ev.pointerId);
});
canvas.addEventListener('pointermove', ev => {
  if (!last) return;
  const p = toCanvas(ev);
  game.slice(last, p);
  last = p;
});
canvas.addEventListener('pointerup', () => { last = null; });

// IA (Etapa 3): por enquanto enxerga o jogo pelos rótulos (modo 'gt'); na Etapa 2 vira o YOLO
let agent = null;
const aiBtn = document.querySelector('#ia');
const temp = document.querySelector('#temp');
const tempOut = document.querySelector('output[for="temp"]');
temp.oninput = () => {
  tempOut.value = temp.value;
  if (agent) agent.options.temperature = Number(temp.value); // vale já no próximo corte
};
aiBtn.onclick = async () => {
  if (agent) {
    agent = null;
    aiBtn.textContent = 'Ligar IA';
    return;
  }
  const { createAgent } = await import('./ml/agent.js');
  if (game.over) game.reset();
  agent = createAgent(game, { perception: 'gt', temperature: Number(temp.value) });
  aiBtn.textContent = 'Desligar IA';
};

let onFrame = null;
const btn = document.querySelector('#dataset');
btn.onclick = async () => {
  const { startDataset } = await import('./dataset.js');
  onFrame = await startDataset(game, canvas, msg => { btn.textContent = msg; });
  btn.disabled = true;
  agent = null; // o bot do dataset e a IA não jogam juntos
  aiBtn.textContent = 'Ligar IA';
  aiBtn.disabled = true;
};

let prev = performance.now();
requestAnimationFrame(function loop(t) {
  game.update(Math.min(0.05, (t - prev) / 1000));
  prev = t;
  agent?.step(); // relógio do jogo: um game.reset() zera o estado do agente sozinho
  game.draw();
  onFrame?.();
  requestAnimationFrame(loop);
});
