#!/usr/bin/env python3
"""
v1.4.14 crack models trained from the reviewed feedback store (training/feedback/feedback.json + images/).

  --mode cls  (option A) pretrained Ultralytics classifier (yolo11n-cls by default) on one crop per cell.
  --mode det  (option B) pretrained YOLO detector on native-resolution tiles (module-level shadow model).

Both modes share one state-file format, one image-level split and one atomic activation step, so the two jobs can run
at the same time as independent processes. The split is ALWAYS by whole image (never random cells) so the reported
numbers come from images the model never trained on.
"""
import argparse, json, math, os, random, shutil, sys, time, traceback
from pathlib import Path

STATE = None  # path of the job-state JSON; set in main()


def configure_windows_worker_executable(executable=None, windows=None):
    """Use the matching pythonw.exe for PyTorch DataLoader children on Windows.

    Worker count and training parallelism stay unchanged; pythonw prevents each
    spawned loader process from opening a separate console window. If pythonw
    is unavailable, leave Python's normal executable untouched.
    """
    is_windows = (os.name == "nt") if windows is None else bool(windows)
    if not is_windows:
        return None
    current = Path(executable or sys.executable)
    hidden_python = current.with_name("pythonw.exe")
    if not hidden_python.is_file():
        return None
    try:
        import multiprocessing
        multiprocessing.set_executable(os.fsencode(str(hidden_python)))
        return str(hidden_python)
    except Exception:
        return None


WINDOWS_WORKER_EXECUTABLE = configure_windows_worker_executable()


def resolve_workers(windows=None, cpu_count=None):
    """Keep up to four DataLoader subprocesses on every platform, as before."""
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(0, min(4, int(cpus)))


# ----------------------------------------------------------------------------- state / logging
def _read_state():
    try:
        return json.loads(Path(STATE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(**patch):
    """Merge `patch` into the state file atomically (Node polls this file; a torn write would look like a crash)."""
    state = _read_state()
    log = patch.pop("log_line", None)
    state.update(patch)
    if log:
        state["log"] = (state.get("log") or [])[-499:] + [f"[{time.strftime('%H:%M:%S')}] {log}"]
        print(log, flush=True)
    state["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = f"{STATE}.tmp"
    Path(tmp).write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def scalar_loss(value):
    """Total of an Ultralytics loss container as a float, or None when it cannot be read.

    FIX (v1.4.14.4): `trainer.tloss` is a torch tensor in older Ultralytics releases but a dict of named loss
    components ({"loss": tensor, ...}) in newer ones, so the old `trainer.tloss.sum().item()` crashed the whole
    run with "'dict' object has no attribute 'sum'". Tensors, numpy arrays, numbers, dicts and sequences are all
    accepted here; anything unreadable yields None instead of an exception.
    """
    try:
        if value is None:
            return None
        if isinstance(value, dict):
            parts = [scalar_loss(v) for v in value.values()]
            parts = [p for p in parts if p is not None]
            return float(sum(parts)) if parts else None
        if isinstance(value, (list, tuple)):
            parts = [scalar_loss(v) for v in value]
            parts = [p for p in parts if p is not None]
            return float(sum(parts)) if parts else None
        if hasattr(value, "detach"):  # torch.Tensor (any shape, any device)
            value = value.detach().float().sum().item()
        elif hasattr(value, "sum") and hasattr(value, "dtype"):  # numpy array / scalar
            value = value.sum().item()
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def trainer_save_dir(model, result):
    """Run directory of the training that just finished.

    FIX (v1.4.14.4): `model.train()` returns a metrics object. DetMetrics carries `save_dir`, but the
    classifier's ClassifyMetrics does not, so `res.save_dir` could raise AttributeError after a successful
    classifier run. The trainer itself always knows its directory, so use it first.
    """
    for holder in (getattr(model, "trainer", None), result):
        save_dir = getattr(holder, "save_dir", None)
        if save_dir:
            return Path(save_dir)
    raise RuntimeError("Training finished but Ultralytics did not report its run directory.")


def epoch_callback(handler):
    """Wrap a per-epoch progress callback so a telemetry problem can never abort a training run that is otherwise healthy."""
    def wrapped(trainer):
        try:
            handler(trainer)
        except Exception as exc:  # progress reporting is best-effort; the model keeps training
            try:
                write_state(log_line=f"Progress update skipped: {type(exc).__name__}: {exc}")
            except Exception:
                pass
    return wrapped


def atomic_copy(src, dst):
    """Copy via a temp file so a half-written model is never picked up by the app."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{dst}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


# ----------------------------------------------------------------------------- data
def load_feedback(feedback_dir):
    """Cell-level labels (category spider_web_crack) grouped by source image."""
    items = json.loads((Path(feedback_dir) / "feedback.json").read_text(encoding="utf-8"))
    grouped = {}
    for it in items:
        if it.get("category") == "spider_web_crack" and it.get("label") in (0, 1):
            grouped.setdefault(it["imageFile"], []).append(it)
    return grouped


def split_by_image(grouped, val_fraction, seed):
    """Whole-image split; both classes are guaranteed in validation when any image can supply them."""
    names = sorted(grouped)
    random.Random(seed).shuffle(names)
    n_val = max(1, round(len(names) * val_fraction))
    val, train = names[:n_val], names[n_val:]
    for label in (1, 0):  # make sure validation sees cracked AND clean cells
        if not any(i["label"] == label for n in val for i in grouped[n]):
            donor = next((n for n in train if any(i["label"] == label for i in grouped[n])), None)
            if donor and len(train) > 1:
                train.remove(donor)
                val.append(donor)
    return train, val


def open_image(path):
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None  # orthomosaic tiles can exceed Pillow's decompression-bomb guard
    # Close the source file immediately after EXIF normalization and RGB conversion.
    # The returned image is a detached copy, so large batch runs do not retain file handles.
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")  # Node side uses sharp().rotate(): same orientation


def buffered_crop(w, h, it, buffer):
    """Mirror of bufferedCellCrop() in server/crackModel.ts so training and inference crop identically."""
    b = max(1, round(min(it["width"], it["height"]) * buffer)) if buffer > 0 else 0
    left, top = max(0, int(it["left"] - b)), max(0, int(it["top"] - b))
    right, bottom = min(w, math.ceil(it["left"] + it["width"] + b)), min(h, math.ceil(it["top"] + it["height"] + b))
    return left, top, right, bottom


def is_outer_cell(it):
    """Cell centre lies within ~1.2 cell sizes of the module outline (the module's outer ring)."""
    if not it.get("moduleWidth"):
        return False
    cx, cy = it["left"] + it["width"] / 2, it["top"] + it["height"] / 2
    reach = max(it["width"], it["height"]) * 1.2
    ml, mt = it.get("moduleLeft", 0), it.get("moduleTop", 0)
    return min(cx - ml, ml + it["moduleWidth"] - cx, cy - mt, mt + it["moduleHeight"] - cy) <= reach


# ----------------------------------------------------------------------------- option A: crop classifier
def build_cls_dataset(grouped, train_imgs, val_imgs, feedback_dir, out, imgsz, buffer, a):
    """ImageFolder layout: {train,val}/{clean,cracked}/*.png. Crops are squashed to imgsz x imgsz exactly like the Node path."""
    from PIL import Image
    stats = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}}
    pending = {0: [], 1: []}  # train crops with their weight multiplier, balanced after the loop
    for split, names in (("train", train_imgs), ("val", val_imgs)):
        for cls in ("clean", "cracked"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)
        for n_img, name in enumerate(names):
            img = open_image(Path(feedback_dir) / "images" / name)
            w, h = img.size
            for k, it in enumerate(grouped[name]):
                l, t, r, b = buffered_crop(w, h, it, buffer)
                if r - l < 8 or b - t < 8:
                    continue
                crop = img.crop((l, t, r, b)).resize((imgsz, imgsz), Image.LANCZOS)
                label = it["label"]
                stem = f"{name[:12]}_{k}"
                path = out / split / ("cracked" if label else "clean") / f"{stem}.png"
                crop.save(path, compress_level=1)  # lossless: JPEG would smear hairline cracks
                stats[split][label] += 1
                if split == "train":
                    mult = (a.edge_weight if is_outer_cell(it) else 1) * \
                           (a.hard_negative_weight if it.get("source") == "rejected-detection" and label == 0 else 1) * \
                           (a.missed_crack_weight if it.get("source") == "missed-crack" and label == 1 else 1)
                    pending[label].append((path, mult))
            write_state(progress=5 + 20 * (n_img + 1) / max(1, len(names)) * (0.6 if split == "train" else 0.4), message=f"Cropping cells: {split} image {n_img + 1}/{len(names)}")
    # Ultralytics' classifier has no class weights, so weights are applied as extra copies (augmentation makes each copy different).
    totals = {c: sum(m for _, m in pending[c]) for c in (0, 1)}
    minority = min(totals, key=totals.get)
    boost = min(8.0, totals[1 - minority] / max(1.0, totals[minority]))
    rng = random.Random(a.seed)
    for label, rows in pending.items():
        for path, mult in rows:
            want = mult * (boost if label == minority else 1.0)
            copies = int(want) + (1 if rng.random() < want - int(want) else 0)
            for c in range(1, max(1, copies)):
                shutil.copyfile(path, path.with_name(f"{path.stem}_x{c}.png"))
    return stats, {"minorityClass": "cracked" if minority == 1 else "clean", "boost": round(boost, 2)}


def softmax_if_needed(scores):
    import numpy as np
    s = np.asarray(scores, dtype="float32")
    if np.all(s >= 0) and np.allclose(s.sum(axis=1), 1.0, atol=1e-3):
        return s  # Ultralytics' exported classifier already applies softmax
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def evaluate_onnx(onnx_path, val_dir, imgsz, cracked_idx, target_recall):
    """Score every held-out crop with the EXPORTED model (the file the app will run) and pick the decision threshold."""
    import numpy as np, onnxruntime as ort
    from PIL import Image
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    files, ys = [], []
    for cls, y in (("clean", 0), ("cracked", 1)):
        for f in sorted((Path(val_dir) / cls).glob("*.png")):
            files.append(f); ys.append(y)
    if not files:
        raise RuntimeError("No held-out crops to evaluate.")
    probs = []
    start = time.perf_counter()
    for i in range(0, len(files), 32):
        batch_images = []
        for f in files[i:i + 32]:
            with Image.open(f) as source:
                batch_images.append(np.asarray(source.convert("RGB"), dtype="float32").transpose(2, 0, 1) / 255.0)
        batch = np.stack(batch_images)
        probs.extend(softmax_if_needed(sess.run(None, {inp: batch})[0])[:, cracked_idx].tolist())
    ms_per_cell = (time.perf_counter() - start) * 1000 / len(files)
    return np.asarray(ys), np.asarray(probs), ms_per_cell


def metrics_at(y, p, t):
    pred = p >= t
    tp, fp = int(((pred) & (y == 1)).sum()), int(((pred) & (y == 0)).sum())
    fn, tn = int(((~pred) & (y == 1)).sum()), int(((~pred) & (y == 0)).sum())
    rec, spec, prec = tp / max(1, tp + fn), tn / max(1, tn + fp), tp / max(1, tp + fp)
    return {"threshold": round(float(t), 4), "precision": round(prec, 4), "recall": round(rec, 4), "specificity": round(spec, 4),
            "balancedAccuracy": round((rec + spec) / 2, 4), "f1": round(2 * prec * rec / max(1e-12, prec + rec), 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "positives": tp + fn, "negatives": tn + fp}


def pick_threshold(y, p, target_recall):
    """Highest threshold that still reaches target recall (= best precision at that recall); best-F1 fallback."""
    import numpy as np
    candidates = np.unique(np.round(np.concatenate([p, np.linspace(0.05, 0.95, 91)]), 4))
    ok = [t for t in candidates if metrics_at(y, p, t)["recall"] >= target_recall]
    if ok:
        return float(max(ok)), True
    return float(max(candidates, key=lambda t: metrics_at(y, p, t)["f1"])), False


def run_cls(a, grouped, train_imgs, val_imgs):
    from ultralytics import YOLO
    ds = Path(a.work) / "cls_dataset"
    shutil.rmtree(ds, ignore_errors=True)
    stats, balance = build_cls_dataset(grouped, train_imgs, val_imgs, a.feedback_dir, ds, a.imgsz, a.cell_buffer, a)
    write_state(stage="training", status="training", progress=28, message=f"Training {a.base_model} on {stats['train'][0] + stats['train'][1]} cell crops.",
                log_line=f"Crops train={stats['train']} val={stats['val']}; class boost {balance}")
    model = YOLO(a.base_model)  # pretrained ImageNet weights: this is the whole point of option A

    def on_epoch(trainer):
        e, total = trainer.epoch + 1, trainer.epochs
        m = {k: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
        state = _read_state()
        # scalar_loss() replaces trainer.tloss.sum().item(), which crashed when tloss is a dict (see scalar_loss).
        write_state(epoch=e, epochs=total, progress=28 + 55 * e / total, message=f"Epoch {e}/{total}",
                    metrics=(state.get("metrics") or []) + [{"epoch": e, "trainLoss": scalar_loss(getattr(trainer, "tloss", None)),
                                                             "validationLoss": m.get("val/loss"), "top1": m.get("metrics/accuracy_top1")}], log_line=f"epoch {e}/{total} train_loss={scalar_loss(getattr(trainer, 'tloss', None))} val_loss={m.get('val/loss')} top1={m.get('metrics/accuracy_top1')}")

    model.add_callback("on_fit_epoch_end", epoch_callback(on_epoch))
    res = model.train(data=str(ds), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience, seed=a.seed, project=str(Path(a.work) / "runs"),
                      name="cls", exist_ok=True, pretrained=True, optimizer="AdamW", lr0=1e-3, cos_lr=True, fliplr=0.5, flipud=0.5,
                      scale=0.15, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, erasing=0.0,  # erasing=0: a random erase can delete the very crack that defines the label
                      workers=resolve_workers(), plots=False, verbose=False, deterministic=False)
    best = trainer_save_dir(model, res) / "weights" / "best.pt"
    write_state(stage="export", status="validating", progress=85, message="Exporting ONNX and validating the exported file on held-out images.")
    best_model = YOLO(str(best))  # loaded once: names and export come from the same object
    names = best_model.names
    cracked_idx = next((i for i, n in names.items() if str(n).lower() == "cracked"), None)
    if cracked_idx is None:
        raise RuntimeError(f"The trained classifier has no 'cracked' class (found {list(names.values())}).")
    onnx = Path(best_model.export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=True, batch=1))
    y, p, ms = evaluate_onnx(onnx, ds / "val", a.imgsz, cracked_idx, a.target_recall)
    thr, met_target = pick_threshold(y, p, a.target_recall)
    final = metrics_at(y, p, thr)
    accepted = final["balancedAccuracy"] >= a.min_balanced_accuracy and final["positives"] > 0 and final["negatives"] > 0
    version = f"cell-cls-{Path(a.base_model).stem}-{time.strftime('%Y%m%d%H%M')}"
    report = {"accepted": accepted, "version": version, "validation": final, "metTargetRecall": met_target, "targetRecall": a.target_recall,
              "msPerCell": round(ms, 2), "trainCrops": stats["train"], "valCrops": stats["val"], "valImages": len(val_imgs), "trainImages": len(train_imgs),
              "evaluationNote": "Held-out whole images, but the best epoch was picked on the same images: figures are slightly optimistic."}
    if accepted:
        atomic_copy(onnx, a.activate)
        meta = {"schema": "cell-classifier-v1", "version": version, "imgsz": a.imgsz, "classNames": [str(names[i]) for i in sorted(names)], "crackedIndex": cracked_idx,
                "threshold": thr, "cellBuffer": a.cell_buffer, "baseModel": a.base_model, "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"), "validation": final, "msPerCell": report["msPerCell"]}
        Path(f"{a.meta_activate}.tmp").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(f"{a.meta_activate}.tmp", a.meta_activate)  # meta last: the app treats a model without meta as absent
    return report


# ----------------------------------------------------------------------------- option B: tiled detector
def tile_origins(size, tile, stride):
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    return xs if xs[-1] == size - tile else xs + [size - tile]  # last tile is flush with the image edge, nothing is left uncovered


def build_det_dataset(grouped, train_imgs, val_imgs, feedback_dir, out, a):
    """Tiles at NATIVE resolution (no downscale: hairline cracks vanish when shrunk).
    Positive tile = contains a cracked cell. Negative tile = >= min_clean_coverage of its area is covered by cells reviewed CLEAN and
    it has no crack. Everything else is skipped because unlabeled cells may be cracked (module-level negatives are not complete)."""
    rng = random.Random(a.seed)
    counts = {"train": [0, 0], "val": [0, 0]}
    tile, stride = a.imgsz, max(1, int(a.imgsz * (1 - a.tile_overlap)))
    for split, names in (("train", train_imgs), ("val", val_imgs)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        pos, neg = [], []
        for name in names:
            img = open_image(Path(feedback_dir) / "images" / name)
            w, h = img.size
            for y0 in tile_origins(h, tile, stride):
                for x0 in tile_origins(w, tile, stride):
                    x1, y1 = min(w, x0 + tile), min(h, y0 + tile)
                    boxes, clean_area = [], 0.0
                    for it in grouped[name]:
                        ix0, iy0 = max(x0, it["left"]), max(y0, it["top"])
                        ix1, iy1 = min(x1, it["left"] + it["width"]), min(y1, it["top"] + it["height"])
                        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                        if inter <= 0:
                            continue
                        if it["label"] == 1 and inter >= 0.5 * it["width"] * it["height"]:
                            boxes.append(((ix0 + ix1) / 2 - x0, (iy0 + iy1) / 2 - y0, ix1 - ix0, iy1 - iy0))
                        elif it["label"] == 0:
                            clean_area += inter
                    if boxes:
                        pos.append((name, x0, y0, x1, y1, boxes))
                    elif clean_area >= a.min_clean_coverage * (x1 - x0) * (y1 - y0):
                        neg.append((name, x0, y0, x1, y1, []))
            del img
        rng.shuffle(neg)
        keep = pos + neg[: max(1, int(len(pos) * a.neg_ratio))]  # cap background tiles so they cannot drown the positives
        counts[split] = [len(pos), len(keep) - len(pos)]
        keep.sort(key=lambda row: row[0])  # one decode per source image: orthomosaics are large
        current, img = None, None
        for n, (name, x0, y0, x1, y1, boxes) in enumerate(keep):
            if name != current:
                current, img = name, open_image(Path(feedback_dir) / "images" / name)
            stem = f"{name[:10]}_{x0}_{y0}"
            img.crop((x0, y0, x1, y1)).save(out / "images" / split / f"{stem}.png", compress_level=1)
            tw, th = x1 - x0, y1 - y0
            (out / "labels" / split / f"{stem}.txt").write_text("".join(f"0 {cx / tw:.6f} {cy / th:.6f} {bw / tw:.6f} {bh / th:.6f}\n" for cx, cy, bw, bh in boxes))
            if n % 25 == 0:
                write_state(progress=5 + 20 * n / max(1, len(keep)), message=f"Writing {split} tiles {n}/{len(keep)}")
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: crack\n", encoding="utf-8")
    return counts


def run_det(a, grouped, train_imgs, val_imgs):
    from ultralytics import YOLO
    ds = Path(a.work) / "det_dataset"
    shutil.rmtree(ds, ignore_errors=True)
    counts = build_det_dataset(grouped, train_imgs, val_imgs, a.feedback_dir, ds, a)
    if counts["train"][0] < 4 or counts["val"][0] < 1:
        raise RuntimeError(f"Too few cracked tiles (train {counts['train'][0]}, val {counts['val'][0]}); label more cracked cells across more images.")
    write_state(stage="training", status="training", progress=28, message="Training tiled crack detector (shadow model).", log_line=f"Tiles [positive, background] train={counts['train']} val={counts['val']}")
    model = YOLO(a.base_model)

    def on_epoch(trainer):
        e, total = trainer.epoch + 1, trainer.epochs
        m = {k: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
        row = {"epoch": e, "trainLoss": scalar_loss(getattr(trainer, "tloss", None)), "precision": m.get("metrics/precision(B)"), "recall": m.get("metrics/recall(B)"), "map50": m.get("metrics/mAP50(B)")}
        write_state(epoch=e, epochs=total, progress=28 + 55 * e / total, message=f"Epoch {e}/{total}", metrics=(_read_state().get("metrics") or []) + [row],
                    log_line=f"epoch {e}/{total} P={row['precision']} R={row['recall']} mAP50={row['map50']}")

    model.add_callback("on_fit_epoch_end", epoch_callback(on_epoch))
    res = model.train(data=str(ds / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience, seed=a.seed, project=str(Path(a.work) / "runs"),
                      name="det", exist_ok=True, pretrained=True, cos_lr=True, flipud=0.5, fliplr=0.5, scale=0.25, mosaic=0.5, close_mosaic=10,
                      workers=resolve_workers(), plots=False, verbose=False, max_det=1000)
    rd = getattr(res, "results_dict", {}) or {}
    best = trainer_save_dir(model, res) / "weights" / "best.pt"
    write_state(stage="export", status="validating", progress=88, message="Exporting ONNX.")
    onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False))
    precision, recall, map50 = (float(rd.get(k, 0) or 0) for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"))
    version = f"module-det-{Path(a.base_model).stem}-{time.strftime('%Y%m%d%H%M')}"
    report = {"accepted": map50 > 0, "version": version, "validation": {"precision": precision, "recall": recall, "mAP50": map50}, "tiles": counts,
              "valImages": len(val_imgs), "trainImages": len(train_imgs), "evaluationNote": "Shadow model: compared with the classifier in the app, never shown as a result."}
    if report["accepted"]:
        atomic_copy(onnx, a.activate)
        meta = {"schema": "module-detector-v1", "version": version, "imgsz": a.imgsz, "tileOverlap": a.tile_overlap, "confidence": 0.25, "iou": 0.5,
                "baseModel": a.base_model, "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"), "validation": report["validation"]}
        Path(f"{a.meta_activate}.tmp").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(f"{a.meta_activate}.tmp", a.meta_activate)
    return report


# ----------------------------------------------------------------------------- main
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("cls", "det"), required=True)
    p.add_argument("--feedback-dir", required=True)
    p.add_argument("--work", required=True, help="scratch/run directory for this job")
    p.add_argument("--state-file", required=True)
    p.add_argument("--activate", required=True, help="destination .onnx")
    p.add_argument("--meta-activate", required=True, help="destination .json (written after the ONNX)")
    p.add_argument("--base-model", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--cell-buffer", type=float, default=0.03)
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--min-balanced-accuracy", type=float, default=0.8)
    p.add_argument("--edge-weight", type=float, default=2.0)
    p.add_argument("--hard-negative-weight", type=float, default=2.0)
    p.add_argument("--missed-crack-weight", type=float, default=2.0)
    p.add_argument("--tile-overlap", type=float, default=0.25)
    p.add_argument("--min-clean-coverage", type=float, default=0.5)
    p.add_argument("--neg-ratio", type=float, default=2.0)
    return p.parse_args()


def main():
    global STATE
    a = parse()
    STATE = a.state_file
    Path(a.work).mkdir(parents=True, exist_ok=True)
    try:
        write_state(status="preparing", stage="preparing", mode=a.mode, pid=os.getpid(), progress=1, message="Reading reviewed labels.", metrics=[], log=[])
        grouped = load_feedback(a.feedback_dir)
        if len(grouped) < 4:
            raise RuntimeError(f"Need at least 4 reviewed images, found {len(grouped)}.")
        train_imgs, val_imgs = split_by_image(grouped, a.val_fraction, a.seed)
        worker_executable = Path(WINDOWS_WORKER_EXECUTABLE).name if WINDOWS_WORKER_EXECUTABLE else 'default Python executable'
        write_state(log_line=f"Loader workers: {resolve_workers()}; worker executable: {worker_executable}")
        write_state(log_line=f"{len(grouped)} images: {len(train_imgs)} train / {len(val_imgs)} held-out (whole images).")
        report = (run_cls if a.mode == "cls" else run_det)(a, grouped, train_imgs, val_imgs)
        done = "Held-out validation passed; model activated." if report["accepted"] else "Held-out validation did not meet the gate; the previous model was kept."
        write_state(status="completed", stage="completed", progress=100, message=done, report=report, finishedAt=time.strftime("%Y-%m-%dT%H:%M:%S"), log_line=done)
        return 0
    except BaseException as exc:  # includes SystemExit/KeyboardInterrupt so the state never stays "training" after a crash
        write_state(status="failed", stage="failed", message=str(exc), error=str(exc), errorTraceback=traceback.format_exc(), finishedAt=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
