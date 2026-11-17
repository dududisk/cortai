"""Refaz a divisão train/val do dataset do Cortaí por BLOCOS contíguos de frames.

Por quê: o `src/dataset.js` manda para `val` todo frame com `n % 10 == 0`. Os frames são capturados a
~133 ms um do outro, então cada frame de val tem vizinhos quase idênticos em train e o mAP de val fica
inflado. Aqui juntamos os dois splits originais pelo número do arquivo, cortamos a sequência em blocos
contíguos e mandamos 1 de cada `--val-every` blocos inteiros para val.

Faixa de guarda: um objeto fica no ar no máximo ~2,3 s (≈ 18 frames capturados). Os `--guard` frames
de cada lado de um bloco de val NÃO entram em train (nem em val), para que nenhum objeto apareça nos
dois splits.

Nada em `data/` é alterado: só escrevemos listas de caminhos absolutos em `training/splits/`.
A Ultralytics acha o rótulo trocando `/images/` por `/labels/` no caminho da imagem.

Uso:
    python training/make_splits.py                       # padrões abaixo
    python training/make_splits.py --block 100 --val-every 10 --val-offset 5 --guard 20
    python training/make_splits.py --train-stride 2      # metade do train (treino em CPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # raiz do repositório
TRAINING = ROOT / "training"
NAMES = {0: "fruta", 1: "bomba"}
CANVAS_W, CANVAS_H = 960, 540
SMALL_PX = 8


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]] | str:
    """Lê um rótulo YOLO. Devolve a lista de caixas ou uma string com o erro."""
    boxes = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            return f"linha {i}: {len(parts)} campos"
        try:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            return f"linha {i}: valor inválido"
        if cls not in NAMES:
            return f"linha {i}: classe {cls}"
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            return f"linha {i}: coordenada fora de [0, 1]"
        boxes.append((cls, cx, cy, w, h))
    return boxes


def collect(data: Path) -> tuple[dict[int, dict], list[str]]:
    """Junta images/{train,val} pelo número do nome. Devolve {idx: info} e a lista de problemas."""
    frames: dict[int, dict] = {}
    problems: list[str] = []
    for split in ("train", "val"):
        for img in sorted((data / "images" / split).glob("*.jpg")):
            try:
                idx = int(img.stem)
            except ValueError:
                problems.append(f"nome não numérico, ignorado: {img}")
                continue
            if idx in frames:
                problems.append(f"índice duplicado {idx}: {img} e {frames[idx]['img']}")
                continue
            lbl = data / "labels" / split / f"{img.stem}.txt"
            if not lbl.exists():
                problems.append(f"sem rótulo, excluído: {img}")
                continue
            boxes = parse_label(lbl)
            if isinstance(boxes, str):
                problems.append(f"rótulo inválido, excluído: {lbl} ({boxes})")
                continue
            frames[idx] = {"img": img.resolve(), "boxes": boxes}
    return frames, problems


def split_frames(frames: dict[int, dict], block: int, val_every: int, val_offset: int, guard: int,
                 train_stride: int = 1):
    """Atribui cada índice a 'train', 'val' ou 'guard'."""
    last = max(frames)
    n_blocks = last // block + 1
    val_blocks = [b for b in range(n_blocks) if b % val_every == val_offset]
    val_ranges = [(b * block, b * block + block - 1) for b in val_blocks]

    def where(idx: int) -> str:
        for s, e in val_ranges:
            if s <= idx <= e:
                return "val"
        for s, e in val_ranges:
            if s - guard <= idx < s or e < idx <= e + guard:
                return "guard"
        return "train"

    out = {"train": [], "val": [], "guard": []}
    for idx in sorted(frames):
        out[where(idx)].append(idx)
    if train_stride > 1:  # frames vizinhos são quase iguais: 1 a cada k mantém a variedade
        out["train"] = out["train"][::train_stride]
    return out, val_ranges


def stats(frames: dict[int, dict], idxs: list[int]) -> dict:
    cls = Counter()
    empty = small = 0
    for i in idxs:
        boxes = frames[i]["boxes"]
        if not boxes:
            empty += 1
        for c, _, _, w, h in boxes:
            cls[NAMES[c]] += 1
            if w * CANVAS_W < SMALL_PX or h * CANVAS_H < SMALL_PX:
                small += 1
    n = len(idxs)
    return {
        "images": n,
        "empty": empty,
        "empty_pct": round(100 * empty / n, 1) if n else 0.0,
        "boxes": dict(cls),
        f"boxes_lt_{SMALL_PX}px": small,
    }


def make_splits(data: Path = ROOT / "data", out: Path = TRAINING / "splits", block: int = 100,
                val_every: int = 10, val_offset: int = 5, guard: int = 20, train_stride: int = 1,
                verbose: bool = True) -> dict:
    data, out = Path(data).resolve(), Path(out).resolve()
    if not (data / "images").is_dir():
        raise SystemExit(f"Dataset não encontrado em {data}. Gere com o botão 'Gerar dataset' do jogo.")
    frames, problems = collect(data)
    parts, val_ranges = split_frames(frames, block, val_every, val_offset, guard, train_stride)

    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val"):
        lines = [frames[i]["img"].as_posix() for i in parts[name]]
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "params": {"data": data.as_posix(), "block": block, "val_every": val_every,
                   "val_offset": val_offset, "guard": guard, "train_stride": train_stride},
        "val_ranges": val_ranges,
        "train": stats(frames, parts["train"]),
        "val": stats(frames, parts["val"]),
        "guard_excluded": len(parts["guard"]),
        "problems": problems,
    }
    total = report["train"]["images"] + report["val"]["images"]
    report["val_pct"] = round(100 * report["val"]["images"] / total, 1)
    (out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"Frames válidos: {len(frames)}  (índices {min(frames)}..{max(frames)})")
        print(f"Blocos de val (índices): {val_ranges}")
        for name in ("train", "val"):
            s = report[name]
            print(f"  {name:5s}: {s['images']:5d} imagens | vazias {s['empty']} ({s['empty_pct']}%) | "
                  f"caixas {s['boxes']} | <{SMALL_PX}px {s[f'boxes_lt_{SMALL_PX}px']}")
        print(f"  guarda: {len(parts['guard'])} frames fora dos dois splits | val = {report['val_pct']}%")
        for p in problems:
            print(f"  AVISO: {p}")
        print(f"Listas em {out}")
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos no console do Windows
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=TRAINING / "splits")
    ap.add_argument("--block", type=int, default=100, help="frames por bloco (100 ≈ 13 s de jogo)")
    ap.add_argument("--val-every", type=int, default=10, help="1 bloco de val a cada N blocos")
    ap.add_argument("--val-offset", type=int, default=5, help="qual bloco (0..N-1) de cada grupo vai para val")
    ap.add_argument("--guard", type=int, default=20, help="frames descartados de cada lado de um bloco de val")
    ap.add_argument("--train-stride", type=int, default=1, help="usa 1 a cada k frames de train (val fica inteiro)")
    a = ap.parse_args()
    make_splits(a.data, a.out, a.block, a.val_every, a.val_offset, a.guard, a.train_stride)


if __name__ == "__main__":
    main()
