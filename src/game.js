// Núcleo do jogo: física, corte, pontuação e rótulos YOLO.
// Não depende do DOM além do canvas, então a IA usa a mesma API que o mouse.
export const W = 960, H = 540;
const GRAVITY = 900; // px/s²

// raio = raio visível do desenho (o SVG ocupa 90% do quadro)
export const FRUITS = {
  acai:    { juice: '#5b1460', r: 36 },
  cupuacu: { juice: '#f1e6c4', r: 44 },
  pupunha: { juice: '#f07c1e', r: 34 },
  guarana: { juice: '#d62f2f', r: 34 },
  bacuri:  { juice: '#f3e7a8', r: 38 },
  tucuma:  { juice: '#f5a623', r: 32 },
};
const BOMB_R = 34;
export const CLASSES = ['fruta', 'bomba'];

// mulberry32: mesma seed = mesma sequência de lançamentos (partidas reproduzíveis)
export function rng(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function segHitsCircle(a, b, c) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((c.x - a.x) * dx + (c.y - a.y) * dy) / len2));
  const px = a.x + t * dx - c.x, py = a.y + t * dy - c.y;
  return px * px + py * py <= c.r * c.r;
}

const images = {};
export function loadAssets() {
  return Promise.all([...Object.keys(FRUITS), 'bomba'].map(name => new Promise((ok, fail) => {
    const img = new Image();
    img.onload = () => ok(images[name] = img);
    img.onerror = () => fail(new Error(`asset ${name} não carregou`));
    img.src = `assets/${name}.svg`;
  })));
}

function move(o, dt) {
  o.vy += GRAVITY * dt;
  o.x += o.vx * dt;
  o.y += o.vy * dt;
  o.rot += o.vr * dt;
}

export class Game {
  constructor(canvas, seed) {
    this.canvas = canvas;
    this.ctx = canvas?.getContext('2d');
    this.reset(seed);
  }

  reset(seed = Date.now()) {
    this.seed = seed;
    this.rand = rng(seed);
    this.entities = []; // frutas e bombas inteiras (o que a IA deve detectar)
    this.pieces = [];   // metades cortadas
    this.drops = [];    // respingos de suco
    this.trail = [];    // rastro da lâmina
    this.score = 0;
    this.lives = 3;
    this.over = false;
    this.time = 0;
    this.nextWave = 0.5;
  }

  spawnWave() {
    const kinds = Object.keys(FRUITS);
    const n = 1 + Math.floor(this.rand() * Math.min(5, 2 + this.time / 20));
    for (let i = 0; i < n; i++) {
      const bomb = this.rand() < Math.min(0.35, 0.12 + this.time / 300);
      const kind = bomb ? 'bomba' : kinds[Math.floor(this.rand() * kinds.length)];
      const x = 120 + this.rand() * (W - 240);
      const apex = H * (0.08 + this.rand() * 0.4); // altura máxima do arremesso
      this.entities.push({
        kind, bomb,
        r: bomb ? BOMB_R : FRUITS[kind].r,
        x, y: H + 50,
        vx: (W / 2 - x) * (0.2 + this.rand() * 0.4) + (this.rand() - 0.5) * 120,
        vy: -Math.sqrt(2 * GRAVITY * (H + 50 - apex)),
        rot: this.rand() * Math.PI * 2,
        vr: (this.rand() - 0.5) * 4,
        wait: i * 0.15,
      });
    }
  }

  update(dt) {
    if (this.over) return;
    this.time += dt;
    if ((this.nextWave -= dt) <= 0) {
      this.spawnWave();
      this.nextWave = Math.max(0.7, 1.8 - this.time / 60);
    }
    for (const e of this.entities) if ((e.wait -= dt) <= 0) move(e, dt);
    for (const p of this.pieces) move(p, dt);
    for (const d of this.drops) { move(d, dt); d.life -= dt; }

    for (const e of this.entities) {
      if (e.vy > 0 && e.y - e.r > H) {
        e.gone = true;
        if (!e.bomb && --this.lives <= 0) this.over = true;
      }
    }
    this.entities = this.entities.filter(e => !e.gone);
    this.pieces = this.pieces.filter(p => p.y - 80 < H);
    this.drops = this.drops.filter(d => d.life > 0);
    this.trail = this.trail.filter(s => this.time - s.t < 0.15);
  }

  // Ação única do jogo: um segmento de corte de a até b (coordenadas do canvas).
  // Retorna o que foi atingido.
  slice(a, b) {
    if (this.over || (b.x - a.x) ** 2 + (b.y - a.y) ** 2 < 4) return [];
    this.trail.push({ a, b, t: this.time });
    const hit = [];
    for (const e of this.entities) {
      if (e.wait > 0 || e.gone || !segHitsCircle(a, b, e)) continue;
      e.gone = true;
      hit.push(e);
      if (e.bomb) {
        this.splash(e, ['#ffb000', '#ff5a00', '#555'], 40);
        this.over = true;
        break;
      }
      this.score++;
      this.split(e, Math.atan2(b.y - a.y, b.x - a.x));
    }
    this.entities = this.entities.filter(e => !e.gone);
    return hit;
  }

  split(e, angle) {
    const nx = -Math.sin(angle), ny = Math.cos(angle);
    for (const side of [-1, 1]) {
      this.pieces.push({
        kind: e.kind, r: e.r, side,
        x: e.x, y: e.y,
        vx: e.vx + side * nx * 140, vy: Math.min(e.vy, 0) + side * ny * 140,
        rot: angle, vr: side * 3,
        imgRot: e.rot - angle,
      });
    }
    this.splash(e, [FRUITS[e.kind].juice], 14);
  }

  // efeito visual: usa Math.random para não alterar a sequência da seed
  splash(e, colors, n) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2, s = 80 + Math.random() * 260;
      this.drops.push({
        x: e.x, y: e.y, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 120,
        rot: 0, vr: 0, r: 2 + Math.random() * 5, life: 0.7,
        color: colors[i % colors.length],
      });
    }
  }

  // Ground truth para o dataset: caixas YOLO normalizadas (cls cx cy w h)
  labels() {
    return this.entities.flatMap(e => {
      if (e.wait > 0) return [];
      const x1 = Math.max(0, e.x - e.r), x2 = Math.min(W, e.x + e.r);
      const y1 = Math.max(0, e.y - e.r), y2 = Math.min(H, e.y + e.r);
      if (x2 - x1 < 6 || y2 - y1 < 6) return [];
      return [{ cls: e.bomb ? 1 : 0, cx: (x1 + x2) / 2 / W, cy: (y1 + y2) / 2 / H, w: (x2 - x1) / W, h: (y2 - y1) / H }];
    });
  }

  draw() {
    const c = this.ctx;
    c.drawImage(background(), 0, 0);

    for (const p of this.pieces) {
      c.save();
      c.translate(p.x, p.y);
      c.rotate(p.rot);
      c.beginPath();
      c.rect(-p.r * 1.3, p.side < 0 ? -p.r * 1.3 : 0, p.r * 2.6, p.r * 1.3);
      c.clip();
      c.rotate(p.imgRot);
      sprite(c, p.kind, p.r);
      c.restore();
    }

    for (const e of this.entities) {
      if (e.wait > 0) continue;
      c.save();
      c.translate(e.x, e.y);
      c.rotate(e.rot);
      sprite(c, e.kind, e.r);
      c.restore();
    }

    for (const d of this.drops) {
      c.globalAlpha = Math.max(0, d.life / 0.7);
      c.fillStyle = d.color;
      c.beginPath();
      c.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      c.fill();
    }
    c.globalAlpha = 1;

    c.lineCap = 'round';
    for (const s of this.trail) {
      const k = 1 - (this.time - s.t) / 0.15;
      c.strokeStyle = `rgba(255,255,255,${k})`;
      c.lineWidth = 2 + 6 * k;
      c.beginPath();
      c.moveTo(s.a.x, s.a.y);
      c.lineTo(s.b.x, s.b.y);
      c.stroke();
    }

    c.fillStyle = '#fff';
    c.font = 'bold 28px system-ui, sans-serif';
    c.textAlign = 'left';
    c.fillText(`${this.score}`, 20, 40);
    c.textAlign = 'right';
    c.fillText('●'.repeat(this.lives) + '○'.repeat(3 - this.lives), W - 20, 40);
    if (this.over) {
      c.textAlign = 'center';
      c.font = 'bold 56px system-ui, sans-serif';
      c.fillText('Fim de jogo', W / 2, H / 2);
      c.font = '22px system-ui, sans-serif';
      c.fillText('clique para jogar de novo', W / 2, H / 2 + 40);
    }
  }
}

function sprite(c, kind, r) {
  const s = (r * 2) / 0.9;
  c.drawImage(images[kind], -s / 2, -s / 2, s, s);
}

let bg = null;
function background() {
  if (bg) return bg;
  bg = document.createElement('canvas');
  bg.width = W; bg.height = H;
  const c = bg.getContext('2d');
  const g = c.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W * 0.7);
  g.addColorStop(0, '#1d4a2c');
  g.addColorStop(1, '#06140c');
  c.fillStyle = g;
  c.fillRect(0, 0, W, H);
  // folhagem de fundo fixa (seed constante)
  const r = rng(7);
  for (let i = 0; i < 28; i++) {
    const edge = i % 2 ? W * r() : (r() < 0.5 ? r() * 160 : W - r() * 160);
    c.save();
    c.translate(edge, H * r());
    c.rotate(r() * Math.PI);
    c.fillStyle = `rgba(${40 + r() * 40},${100 + r() * 60},${50 + r() * 30},0.12)`;
    c.beginPath();
    c.ellipse(0, 0, 90 + r() * 80, 18 + r() * 14, 0, 0, Math.PI * 2);
    c.fill();
    c.restore();
  }
  return bg;
}
