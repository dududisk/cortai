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

let onFrame = null;
const btn = document.querySelector('#dataset');
btn.onclick = async () => {
  const { startDataset } = await import('./dataset.js');
  onFrame = await startDataset(game, canvas, msg => { btn.textContent = msg; });
  btn.disabled = true;
};

let prev = performance.now();
requestAnimationFrame(function loop(t) {
  game.update(Math.min(0.05, (t - prev) / 1000));
  prev = t;
  game.draw();
  onFrame?.();
  requestAnimationFrame(loop);
});
