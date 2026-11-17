"""Treina o YOLO11n do Cortaí (classes 0: fruta, 1: bomba).

Passos:
  1. (re)gera `training/splits/{train,val}.txt` por blocos contíguos (make_splits.py) a cada execução;
  2. escreve `training/data.yaml` com caminhos absolutos;
  3. detecta o device (CUDA se houver, senão CPU) e treina a partir de `yolo11n.pt`.

Tudo o que a Ultralytics grava (pesos baixados, settings, fontes, cache de rótulos, runs) fica em
`training/`. Nada é escrito em `data/`.

Exemplos (a partir da raiz do repositório):
    training/.venv/Scripts/python training/train.py --epochs 50 --imgsz 640
    training/.venv/Scripts/python training/train.py --estimate --imgsz 480    # mede o tempo por época
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
ROOT = TRAINING.parent
# Settings, fontes e afins da Ultralytics ficam dentro de training/ (precisa vir antes do import).
os.environ.setdefault("YOLO_CONFIG_DIR", str(TRAINING / ".ultralytics"))

sys.path.insert(0, str(TRAINING))
from make_splits import make_splits  # noqa: E402

NAMES = {0: "fruta", 1: "bomba"}
SPLITS = TRAINING / "splits"
DATA_YAML = TRAINING / "data.yaml"
RUNS = TRAINING / "runs"
WEIGHTS = TRAINING / "weights" / "yolo11n.pt"


def write_data_yaml() -> Path:
    lines = [
        "# Gerado por training/train.py. Listas de imagens geradas por training/make_splits.py.",
        f"path: {SPLITS.as_posix()}",
        f"train: {(SPLITS / 'train.txt').as_posix()}",
        f"val: {(SPLITS / 'val.txt').as_posix()}",
        "names:",
        *[f"  {k}: {v}" for k, v in NAMES.items()],
        "",
    ]
    DATA_YAML.write_text("\n".join(lines), encoding="utf-8")
    return DATA_YAML


def detect_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def describe_device(device: str) -> str:
    import torch

    if device != "cpu" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(int(device.split(",")[0]))
        return f"GPU {p.name} ({p.total_memory / 2**30:.1f} GB)"
    return f"CPU {platform.processor() or platform.machine()} | {os.cpu_count()} núcleos | torch threads {torch.get_num_threads()}"


def redirect_label_cache() -> None:
    """A Ultralytics salva o cache de rótulos ao lado da pasta labels (ou seja, em data/).

    Aqui o cache vai para training/cache/<lista>.cache (train.cache, val.cache). O conteúdo é validado
    por hash pela própria Ultralytics, então um cache velho é refeito automaticamente.
    """
    import ultralytics.data.dataset as ds

    cache_dir = TRAINING / "cache"
    cache_dir.mkdir(exist_ok=True)
    orig = getattr(ds.YOLODataset, "_load_or_scan_cache", None)
    if orig is None:
        # Outra versão da Ultralytics: só impede a gravação em data/ (rótulos relidos a cada execução).
        ds.save_dataset_cache_file = lambda *args, **kwargs: None
        return

    def _load_or_scan_cache(self, cache_path, cache_hash):
        lista = self.img_path if isinstance(self.img_path, (str, Path)) else cache_path
        return orig(self, cache_dir / f"{Path(str(lista)).stem}.cache", cache_hash)

    ds.YOLODataset._load_or_scan_cache = _load_or_scan_cache


def add_timing(model, epochs_planned: int, n_train_full: int, log_path_holder: dict, estimate: bool) -> dict:
    """Mede tempo de treino e de validação por época e projeta o total."""
    t: dict = {"rows": [], "train_end": None}

    def on_train_epoch_start(trainer):
        t["start"] = time.perf_counter()
        t["first_batch"] = None
        t["batches"] = 0

    def on_train_batch_end(trainer):
        t["batches"] += 1
        if t["first_batch"] is None:
            t["first_batch"] = time.perf_counter()

    def on_train_epoch_end(trainer):
        t["train_end"] = time.perf_counter()

    def on_fit_epoch_end(trainer):
        now = time.perf_counter()
        if t["train_end"] is None:  # chamada extra da validação final (final_eval): não é uma época
            return
        n_imgs = len(trainer.train_loader.dataset)
        train_s = t["train_end"] - t["start"]
        steady = (t["train_end"] - t["first_batch"]) / max(1, t["batches"] - 1) if t["batches"] > 1 else None
        val_s = now - t["train_end"]
        s_per_img = train_s / n_imgs
        full_epoch = s_per_img * n_train_full + val_s
        row = {
            "epoch": trainer.epoch + 1, "imgsz": trainer.args.imgsz, "batch": trainer.batch_size,
            "train_images": n_imgs, "train_batches": t["batches"], "train_s": round(train_s, 1),
            "s_per_batch_steady": round(steady, 2) if steady else None, "s_per_img": round(s_per_img, 4),
            "val_s": round(val_s, 1), "full_epoch_est_min": round(full_epoch / 60, 1),
        }
        try:  # memória do processo (no Windows, pico do working set): ajuda a detectar paginação
            import psutil

            mem = psutil.Process().memory_info()
            row["rss_gb"] = round(mem.rss / 2**30, 2)
            if hasattr(mem, "peak_wset"):
                row["peak_gb"] = round(mem.peak_wset / 2**30, 2)
        except Exception:
            pass
        remaining = epochs_planned - (trainer.epoch + 1)
        if remaining > 0:
            row["eta_h"] = round(remaining * full_epoch / 3600, 1)
        t["rows"].append(row)
        print(f"\n[tempo] {json.dumps(row, ensure_ascii=False)}", flush=True)
        log = Path(trainer.save_dir) / "timing.jsonl"
        log_path_holder["path"] = log
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        t["train_end"] = None
        if estimate:  # sem best.pt a Ultralytics pula a validação final repetida (só queremos o tempo)
            Path(trainer.best).unlink(missing_ok=True)

    for name, fn in [("on_train_epoch_start", on_train_epoch_start), ("on_train_batch_end", on_train_batch_end),
                     ("on_train_epoch_end", on_train_epoch_end), ("on_fit_epoch_end", on_fit_epoch_end)]:
        model.add_callback(name, fn)
    return t


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2, help="workers do DataLoader (a Ultralytics usa 0 em CPU)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1 ...")
    ap.add_argument("--patience", type=int, default=15, help="early stopping (épocas sem melhora)")
    ap.add_argument("--fraction", type=float, default=None, help="fração do train usada (padrão 1.0; 0.1 no --estimate)")
    ap.add_argument("--hours", type=float, default=None, help="limite de horas de treino (arg `time` da Ultralytics)")
    ap.add_argument("--name", default=None, help="nome da pasta em training/runs/")
    ap.add_argument("--model", default=str(WEIGHTS), help="pesos iniciais (baixa yolo11n.pt se faltar)")
    ap.add_argument("--train-stride", type=int, default=1,
                    help="usa 1 a cada k frames de train (ex.: 2 = metade, para CPU; val inalterado)")
    ap.add_argument("--resume", action="store_true", help="retoma training/runs/<name>/weights/last.pt")
    ap.add_argument("--estimate", action="store_true", help="1 época curta só para medir tempo e projetar o total")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--close-mosaic", type=int, default=None, help="desliga o mosaic nas N últimas épocas (padrão 10)")
    ap.add_argument("--freeze", type=int, default=None,
                    help="congela as N primeiras camadas (10 = backbone do YOLO11n); bem mais rápido em CPU")
    ap.add_argument("--cache-ram", action="store_true",
                    help="guarda as imagens decodificadas na RAM (~2,6 GB em 640); útil em GPU/Colab")
    a = ap.parse_args()

    # Refaz as listas a cada execução (determinístico, ~2 s): se data/ for regerado, nada fica velho.
    if not a.resume or not (SPLITS / "train.txt").exists():
        make_splits(train_stride=a.train_stride)
    data_yaml = write_data_yaml()
    n_train_full = sum(1 for line in (SPLITS / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    n_val = sum(1 for line in (SPLITS / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip())

    device = detect_device(a.device)
    print(f"Device: {describe_device(device)}")
    print(f"Dataset: {n_train_full} train / {n_val} val  ->  {data_yaml}")
    if device == "cpu":
        print("Sem GPU CUDA: o treino roda em CPU (lento). Use --estimate para medir o tempo por época antes.")

    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # sem telemetria (settings ficam em training/.ultralytics)
    redirect_label_cache()

    planned_epochs = a.epochs
    fraction = a.fraction if a.fraction is not None else (0.1 if a.estimate else 1.0)
    name = a.name or (f"estimate_{a.imgsz}" if a.estimate else f"yolo11n_{a.imgsz}")
    epochs = 1 if a.estimate else a.epochs

    if a.resume:
        last = RUNS / name / "weights" / "last.pt"
        model = YOLO(str(last))
    else:
        Path(a.model).parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(a.model)
    holder: dict = {}  # recebe o caminho do timing.jsonl
    timing = add_timing(model, planned_epochs, n_train_full, holder, a.estimate)

    t0 = time.perf_counter()
    if a.resume:
        model.train(resume=True)
    else:
        kw = dict(
            data=str(data_yaml), epochs=epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
            device=device, patience=a.patience, fraction=fraction, project=str(RUNS), name=name,
            exist_ok=a.estimate, seed=a.seed, plots=not a.estimate, verbose=True,
        )
        if a.hours:
            kw["time"] = a.hours
        if a.estimate:
            kw["close_mosaic"] = 0  # com epochs=1 a Ultralytics fecharia o mosaic; queremos uma época típica
        elif a.close_mosaic is not None:
            kw["close_mosaic"] = a.close_mosaic
        if a.freeze:
            kw["freeze"] = a.freeze
        if a.cache_ram:
            kw["cache"] = "ram"  # nunca "disk": gravaria .npy dentro de data/
        model.train(**kw)
    total_s = time.perf_counter() - t0
    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else RUNS / name

    if a.estimate and timing["rows"]:
        r = timing["rows"][-1]
        per_epoch_min = r["full_epoch_est_min"]
        print("\n=== Estimativa (CPU/GPU atual, este imgsz/batch) ===")
        print(f"imgsz {a.imgsz} batch {a.batch}: {r['s_per_img']} s/img no treino, val {r['val_s']} s "
              f"-> ~{per_epoch_min} min por época completa ({n_train_full} imgs)")
        for e in sorted({30, 50, planned_epochs}):
            print(f"  {e:3d} épocas ~ {per_epoch_min * e / 60:.1f} h")
        print(f"(rodada de medição levou {total_s / 60:.1f} min no total)")
    else:
        print(f"\nTreino terminou em {total_s / 3600:.2f} h. Pesos: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
