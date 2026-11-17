"""Exporta o YOLO11n treinado para TF.js (graph model) em `models/yolo11n_web/`.

ATENÇÃO: desde a Ultralytics 8.4.83, `format="tfjs"` virou apelido de LiteRT (gera .tflite, só em
Linux/macOS) e NÃO produz `model.json`. Por isso o export é feito em dois passos:
  1. Ultralytics: best.pt -> ONNX -> TF SavedModel   (format="saved_model", usa onnx2tf)
  2. tensorflowjs_converter: SavedModel -> TF.js graph model (model.json + shards .bin)

Rode num ambiente SEPARADO do treino (TF <= 2.19 exige numpy < 2). Caminho recomendado: Colab ou
Linux x86-64 com `training/requirements-export.txt` (instruções lá). Não roda no venv de treino.

Exemplos:
    python training/export.py                                       # best.pt mais recente em training/runs/
    python training/export.py --weights best.pt --imgsz 640
    python training/export.py --saved-model best_saved_model        # só o passo 2

Contrato da saída (sem NMS; o worker web aplica tf.image.nonMaxSuppressionAsync):
    entrada  [1, imgsz, imgsz, 3]  float32 RGB em [0, 1] (NHWC), letterbox com cinza 114
    saída    [1, 6, N]  N = (s/8)² + (s/16)² + (s/32)²  (8400 para 640, 4725 para 480)
             linhas 0..3 = cx, cy, w, h em PIXELS da entrada (0..imgsz)
             linhas 4..5 = score de fruta e de bomba (sigmoid já aplicada)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
ROOT = TRAINING.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(TRAINING / ".ultralytics"))

OUT = ROOT / "models" / "yolo11n_web"
NAMES = {0: "fruta", 1: "bomba"}


def latest_best() -> Path:
    cands = [p for p in (TRAINING / "runs").glob("*/weights/best.pt") if not p.parts[-3].startswith("estimate")]
    if not cands:
        raise SystemExit("Nenhum best.pt em training/runs/. Rode training/train.py antes ou passe --weights.")
    return max(cands, key=lambda p: p.stat().st_mtime)


def n_anchors(h: int, w: int) -> int:
    return sum((h // s) * (w // s) for s in (8, 16, 32))


def to_saved_model(weights: Path, imgsz: int | None) -> tuple[Path, int]:
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}
    if names != NAMES:
        raise SystemExit(f"Classes inesperadas no modelo: {names} (esperado {NAMES})")
    imgsz = imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))
    print(f"[1/2] {weights} -> SavedModel (imgsz {imgsz}, sem NMS)")
    out = model.export(format="saved_model", imgsz=imgsz, nms=False)
    return Path(out), imgsz


def to_tfjs(saved_model: Path, out: Path) -> None:
    exe = shutil.which("tensorflowjs_converter")
    cmd = [exe] if exe else [sys.executable, "-m", "tensorflowjs.converters.converter"]
    cmd += [
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        str(saved_model),
        str(out),
    ]
    print("[2/2] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def signature_shapes(section: dict) -> dict:
    res = {}
    for k, v in section.items():
        dims = v.get("tensorShape", {}).get("dim", [])
        res[k] = [int(d.get("size", -1)) for d in dims]
    return res


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=None, help="padrão: best.pt mais recente em training/runs/")
    ap.add_argument("--imgsz", type=int, default=None, help="padrão: o imgsz usado no treino")
    ap.add_argument("--saved-model", type=Path, default=None, help="pula o passo 1 e usa este SavedModel")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    if a.saved_model:
        saved, imgsz = a.saved_model.resolve(), a.imgsz
        weights = None
    else:
        weights = (a.weights or latest_best()).resolve()
        saved, imgsz = to_saved_model(weights, a.imgsz)

    out = a.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    to_tfjs(saved, out)
    if (saved / "metadata.yaml").exists():
        shutil.copy2(saved / "metadata.yaml", out / "metadata.yaml")  # names, imgsz, stride (Ultralytics)

    sig = json.loads((out / "model.json").read_text(encoding="utf-8")).get("signature", {})
    inputs, outputs = signature_shapes(sig.get("inputs", {})), signature_shapes(sig.get("outputs", {}))
    if imgsz is None:  # --saved-model sem --imgsz: lê da entrada [1, H, W, 3]
        imgsz = next(iter(inputs.values()))[1]
    expected = [1, 4 + len(NAMES), n_anchors(imgsz, imgsz)]
    info = {
        "source_weights": weights.as_posix() if weights else None,
        "imgsz": imgsz,
        "names": NAMES,
        "input": inputs,
        "output": outputs,
        "expected_output": expected,
        "layout": "saída [1, 6, N]: cx, cy, w, h (pixels da entrada), score_fruta, score_bomba; sem NMS",
        "preprocess": "RGB, float32 /255, letterbox para imgsz x imgsz com cinza 114, NHWC",
    }
    (out / "cortai_model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if expected not in outputs.values():
        print(f"AVISO: saída {outputs} difere do esperado {expected}. Avise o agente percepcao-web.")
    print(f"\nOK: {out / 'model.json'}")


if __name__ == "__main__":
    main()
