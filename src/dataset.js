// Gera dataset YOLO direto do estado do jogo: sem rotular nada à mão.
// Grava em uma pasta escolhida pelo usuário (File System Access API, Chrome/Edge):
//   images/{train,val}/000123.jpg  +  labels/{train,val}/000123.txt
import { H } from './game.js';

const EVERY_N_FRAMES = 8; // espaça as capturas para os frames não ficarem quase iguais

export async function startDataset(game, canvas, status) {
  const root = await window.showDirectoryPicker({ mode: 'readwrite' });
  const dirs = {};
  for (const kind of ['images', 'labels']) {
    const k = await root.getDirectoryHandle(kind, { create: true });
    for (const split of ['train', 'val']) dirs[`${kind}/${split}`] = await k.getDirectoryHandle(split, { create: true });
  }

  let n = 0, frame = 0, busy = false;
  game.reset();
  return () => {
    if (game.over) game.reset(); // partidas encadeadas, cada uma com seed nova
    bot(game);
    if (++frame % EVERY_N_FRAMES || busy) return;

    busy = true;
    const labels = game.labels()
      .map(l => `${l.cls} ${l.cx.toFixed(6)} ${l.cy.toFixed(6)} ${l.w.toFixed(6)} ${l.h.toFixed(6)}`)
      .join('\n');
    const name = String(n).padStart(6, '0');
    const split = n % 10 === 0 ? 'val' : 'train'; // 10% validação
    n++;
    canvas.toBlob(async blob => {
      await write(dirs[`images/${split}`], `${name}.jpg`, blob);
      await write(dirs[`labels/${split}`], `${name}.txt`, labels);
      status(`${n} frames salvos`);
      busy = false;
    }, 'image/jpeg', 0.9);
  };
}

async function write(dir, name, data) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.write(data);
  await w.close();
}

// Bot "trapaceiro" (lê o estado real) só para os frames terem cortes, metades e
// respingos, que funcionam como distratores. Não é a IA final.
function bot(game) {
  if (Math.random() > 0.04) return;
  const fruits = game.entities.filter(e => !e.bomb && e.wait <= 0 && e.y < H);
  const f = fruits[Math.floor(Math.random() * fruits.length)];
  if (!f) return;
  const a = Math.random() * Math.PI, d = f.r * 1.5;
  game.slice(
    { x: f.x - Math.cos(a) * d, y: f.y - Math.sin(a) * d },
    { x: f.x + Math.cos(a) * d, y: f.y + Math.sin(a) * d },
  );
}
