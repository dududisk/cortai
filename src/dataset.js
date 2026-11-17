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
  const capture = () => {
    busy = true;
    const labels = game.labels()
      .map(l => `${l.cls} ${l.cx.toFixed(6)} ${l.cy.toFixed(6)} ${l.w.toFixed(6)} ${l.h.toFixed(6)}`)
      .join('\n');
    const name = String(n).padStart(6, '0');
    const split = n % 10 === 0 ? 'val' : 'train'; // 10% validação
    n++;
    // toBlob copia o canvas na hora da chamada; a codificação do JPEG é que é assíncrona
    canvas.toBlob(async blob => {
      // .txt antes do .jpg: se a gravação for interrompida, sobra um .txt sem imagem (ignorado
      // no treino) e não uma imagem sem rótulo (que o treino leria como "frame vazio")
      await write(dirs[`labels/${split}`], `${name}.txt`, labels);
      await write(dirs[`images/${split}`], `${name}.jpg`, blob);
      status(`${n} frames salvos`);
      busy = false;
    }, 'image/jpeg', 0.9);
  };

  game.reset();
  return () => {
    // O main.js já desenhou este frame. Os rótulos têm de sair desse MESMO estado, então tudo o
    // que muda o jogo (reset, corte do bot) vem depois da captura. Antes, o bot cortava primeiro e
    // a fruta aparecia inteira na imagem, mas sem caixa no .txt.
    if (game.over) { game.reset(); return; } // frame de "Fim de jogo": não grava
    if (++frame % EVERY_N_FRAMES === 0 && !busy) capture();
    bot(game);
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
