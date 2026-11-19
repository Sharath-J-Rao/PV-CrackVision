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
    if not STATE:  # no job state file configured (unit tests / library use): keep the update in memory only, never write "None.tmp"
        return
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
    """Create an image-grouped split with both classes in train and validation when feasible.

    A class represented in fewer than two source images cannot be independently validated
    without leaking crops from the same source image. Fail explicitly instead of publishing
    precision/recall metrics from a validation set missing that class.
    """
    names = sorted(grouped)
    if len(names) < 2:
        raise ValueError("At least two distinct source images are required for an image-level train/validation split.")

    class_groups = {
        label: [name for name in names if any(item.get("label") == label for item in grouped[name])]
        for label in (0, 1)
    }
    missing = [label for label in (0, 1) if not class_groups[label]]
    if missing:
        label_name = "clean" if missing[0] == 0 else "cracked"
        raise ValueError(f"Cannot validate the crack classifier: the dataset has no {label_name} source images.")
    insufficient = [label for label in (0, 1) if len(class_groups[label]) < 2]
    if insufficient:
        label_name = "clean" if insufficient[0] == 0 else "cracked"
        raise ValueError(
            f"Cannot validate {label_name} recall/precision without source-image leakage: "
            f"that class appears in only {len(class_groups[insufficient[0]])} source image. "
            "Add reviewed examples from at least two distinct source images."
        )

    rng = random.Random(seed)
    tie_order = list(names)
    rng.shuffle(tie_order)
    tie_rank = {name: index for index, name in enumerate(tie_order)}
    counts = {
        name: {label: sum(1 for item in grouped[name] if item.get("label") == label) for label in (0, 1)}
        for name in names
    }
    total = {label: sum(counts[name][label] for name in names) for label in (0, 1)}
    ratio = min(0.5, max(0.05, float(val_fraction)))
    target_count = max(1, min(len(names) - 1, round(len(names) * ratio)))
    target = {label: total[label] * ratio for label in (0, 1)}

    validation = []
    validation_set = set()
    validation_counts = {0: 0, 1: 0}

    def eligible(name):
        # Leave at least one distinct source image for every class in the training partition.
        return all(
            not counts[name][label]
            or sum(1 for source in class_groups[label] if source not in validation_set and source != name) >= 1
            for label in (0, 1)
        )

    def add(name):
        validation.append(name)
        validation_set.add(name)
        for label in (0, 1):
            validation_counts[label] += counts[name][label]

    # First satisfy class coverage in validation, preferring a mixed image when possible.
    while any(validation_counts[label] == 0 for label in (0, 1)):
        missing_labels = [label for label in (0, 1) if validation_counts[label] == 0]
        candidates = [name for name in names if name not in validation_set and eligible(name)
                      and any(counts[name][label] > 0 for label in missing_labels)]
        if not candidates:
            raise ValueError("No leakage-free image-level split can represent both clean and cracked classes in train and validation.")
        chosen = min(candidates, key=lambda name: (
            -sum(1 for label in missing_labels if counts[name][label] > 0),
            sum(((validation_counts[label] + counts[name][label] - target[label]) / max(1.0, target[label])) ** 2 for label in (0, 1)),
            tie_rank[name],
        ))
        add(chosen)

    # Add groups until the requested validation size is reached, never removing the last train image for a class.
    while len(validation) < target_count:
        candidates = [name for name in names if name not in validation_set and eligible(name)]
        if not candidates:
            break
        chosen = min(candidates, key=lambda name: (
            sum(((validation_counts[label] + counts[name][label] - target[label]) / max(1.0, target[label])) ** 2 for label in (0, 1)),
            abs((len(validation) + 1) - target_count),
            tie_rank[name],
        ))
        add(chosen)

    train = [name for name in names if name not in validation_set]
    if not train or not validation:
        raise ValueError("The available source images cannot form non-empty train and validation partitions.")
    for label, label_name in ((0, "clean"), (1, "cracked")):
        if not any(counts[name][label] for name in validation):
            raise ValueError(f"The validation split contains no {label_name} cells; add more independent source images.")
        if not any(counts[name][label] for name in train):
            raise ValueError(f"The training split contains no {label_name} cells; add more independent source images.")
    return train, validation


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


def augment_crack_crop(source, destination, seed):
    """Mild crack-only augmentation; tiny geometric/tonal changes preserve hairline crack evidence."""
    from PIL import Image, ImageEnhance, ImageOps
    rng = random.Random(seed)
    with Image.open(source) as image:
        out = image.convert("RGB")
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    angle = rng.uniform(-2.0, 2.0)
    if abs(angle) > 0.25:
        out = out.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.97, 1.03))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.98, 1.04))
    out.save(destination, compress_level=1)


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
                           (a.hard_negative_weight if it.get("source") in ("rejected-detection", "bulk-negative") and label == 0 else 1) * \
                           (a.missed_crack_weight if it.get("source") == "missed-crack" and label == 1 else 1)
                    pending[label].append((path, mult))
            write_state(progress=5 + 20 * (n_img + 1) / max(1, len(names)) * (0.6 if split == "train" else 0.4), message=f"Cropping cells: {split} image {n_img + 1}/{len(names)}")
    # Ultralytics' classifier has no class weights, so weights are applied as extra copies (augmentation makes each copy different).
    totals = {c: sum(m for _, m in pending[c]) for c in (0, 1)}
    minority = min(totals, key=totals.get)
    boost = min(8.0, totals[1 - minority] / max(1.0, totals[minority]))
    rng = random.Random(a.seed)
    crack_augmented = 0
    for label, rows in pending.items():
        for path, mult in rows:
            want = mult * (boost if label == minority else 1.0)
            copies = int(want) + (1 if rng.random() < want - int(want) else 0)
            # Every positive crop gets one additional mild variant. Clean crops are never augmented.
            if label == 1:
                copies = max(copies, 2)
            for c in range(1, max(1, copies)):
                destination = path.with_name(f"{path.stem}_x{c}.png")
                if label == 1:
                    augment_crack_crop(path, destination, a.seed + c + crack_augmented)
                    crack_augmented += 1
                else:
                    shutil.copyfile(path, destination)
    return stats, {"minorityClass": "cracked" if minority == 1 else "clean", "boost": round(boost, 2), "crackOnlyAugmentedCopies": crack_augmented}


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


def metrics_at(y, p, t, clean_to_cracked_ratio=None):
    pred = p >= t
    tp, fp = int(((pred) & (y == 1)).sum()), int(((pred) & (y == 0)).sum())
    fn, tn = int(((~pred) & (y == 1)).sum()), int(((~pred) & (y == 0)).sum())
    rec, spec, prec = tp / max(1, tp + fn), tn / max(1, tn + fp), tp / max(1, tp + fp)
    result = {"threshold": round(float(t), 4), "precision": round(prec, 4), "recall": round(rec, 4), "specificity": round(spec, 4),
              "balancedAccuracy": round((rec + spec) / 2, 4), "f1": round(2 * prec * rec / max(1e-12, prec + rec), 4),
              "tp": tp, "fp": fp, "tn": tn, "fn": fn, "positives": tp + fn, "negatives": tn + fp}
    if clean_to_cracked_ratio is not None:
        ratio = max(1.0, float(clean_to_cracked_ratio))
        cracked_prior, clean_prior = 1.0 / (1.0 + ratio), ratio / (1.0 + ratio)
        weighted_precision = (rec * cracked_prior) / max(1e-12, rec * cracked_prior + (1.0 - spec) * clean_prior)
        result["precisionAtRealisticRatio"] = round(weighted_precision, 4)
        result["falsePositivesPer1000"] = round((1.0 - spec) * clean_prior * 1000.0, 2)
        result["cleanToCrackedRatio"] = ratio
    return result


def precision_recall_curve(y, p, clean_to_cracked_ratio):
    import numpy as np
    candidates = np.unique(np.round(np.concatenate([np.linspace(0.0, 1.0, 51), np.asarray(p)]), 4))
    points = [metrics_at(y, p, threshold, clean_to_cracked_ratio) for threshold in candidates]
    return [{"threshold": row["threshold"], "precision": row["precisionAtRealisticRatio"], "recall": row["recall"],
             "falsePositivesPer1000": row["falsePositivesPer1000"]} for row in points]


def pick_threshold_by_precision(y, p, target_precision, clean_to_cracked_ratio):
    """Lowest threshold meeting reweighted operational precision; maximizes recall under the precision target."""
    import numpy as np
    candidates = np.unique(np.round(np.concatenate([np.asarray(p), np.linspace(0.01, 0.99, 99)]), 4))
    eligible = [threshold for threshold in candidates if metrics_at(y, p, threshold, clean_to_cracked_ratio)["precisionAtRealisticRatio"] >= target_precision]
    if eligible:
        return float(min(eligible)), True
    best = max(candidates, key=lambda threshold: (metrics_at(y, p, threshold, clean_to_cracked_ratio)["precisionAtRealisticRatio"], metrics_at(y, p, threshold, clean_to_cracked_ratio)["recall"]))
    return float(best), False


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
                      name="cls", exist_ok=True, pretrained=True, optimizer="AdamW", lr0=1e-3, cos_lr=True, fliplr=0.0, flipud=0.0,
                      degrees=0.0, translate=0.02, scale=0.03, hsv_h=0.0, hsv_s=0.0, hsv_v=0.02, erasing=0.0,  # strong transforms can erase hairline cracks; crack-only mild variants are generated above
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
    thr, met_target = pick_threshold_by_precision(y, p, a.target_precision, a.clean_to_cracked_ratio)
    final = metrics_at(y, p, thr, a.clean_to_cracked_ratio)
    pr_curve = precision_recall_curve(y, p, a.clean_to_cracked_ratio)
    accepted = final["balancedAccuracy"] >= a.min_balanced_accuracy and final["positives"] > 0 and final["negatives"] > 0
    # Compare against the currently installed user-trained classifier on this exact held-out
    # image split, using each model's own deployed threshold and the same field prevalence.
    # This is a same-split comparison, not an independent external benchmark.
    previous_comparison = None
    previous_comparison_error = None
    try:
        old_meta_path = Path(a.meta_activate)
        old_model_path = Path(a.activate)
        if old_model_path.is_file() and old_meta_path.is_file():
            old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
            old_cracked_idx = int(old_meta.get("crackedIndex", cracked_idx))
            old_y, old_p, _ = evaluate_onnx(old_model_path, ds / "val", a.imgsz, old_cracked_idx, a.target_recall)
            old_threshold = float(old_meta.get("threshold", 0.5))
            old_metrics = metrics_at(old_y, old_p, old_threshold, a.clean_to_cracked_ratio)
            previous_comparison = {
                "previousVersion": old_meta.get("version", "previous installed model"),
                "previousThreshold": old_threshold,
                "previousFalsePositivesPer1000": old_metrics["falsePositivesPer1000"],
                "newFalsePositivesPer1000": final["falsePositivesPer1000"],
                "changeFalsePositivesPer1000": round(final["falsePositivesPer1000"] - old_metrics["falsePositivesPer1000"], 2),
                "reduced": final["falsePositivesPer1000"] < old_metrics["falsePositivesPer1000"],
            }
    except Exception as comparison_error:
        previous_comparison_error = str(comparison_error)
    version = f"cell-cls-{Path(a.base_model).stem}-{time.strftime('%Y%m%d%H%M')}"
    report = {"accepted": accepted, "version": version, "validation": final, "metTargetPrecision": met_target, "targetPrecision": a.target_precision, "cleanToCrackedRatio": a.clean_to_cracked_ratio,
              "precisionRecallCurve": pr_curve, "classCounts": {"cracked": sum(1 for rows in grouped.values() for item in rows if item["label"] == 1), "clean": sum(1 for rows in grouped.values() for item in rows if item["label"] == 0)},
              "msPerCell": round(ms, 2), "trainCrops": stats["train"], "valCrops": stats["val"], "valImages": len(val_imgs), "trainImages": len(train_imgs),
              "previousModelComparison": previous_comparison, "previousModelComparisonError": previous_comparison_error,
              "evaluationNote": ("Target operational precision met. " if met_target else "Target operational precision not demonstrated on this validation set; review the PR curve and collect more clean examples. ") + "Validation uses whole source images; no crop from one image is split between train and validation."}
    if accepted:
        atomic_copy(onnx, a.activate)
        meta = {"schema": "cell-classifier-v1", "version": version, "imgsz": a.imgsz, "classNames": [str(names[i]) for i in sorted(names)], "crackedIndex": cracked_idx,
                "threshold": thr, "targetPrecision": a.target_precision, "cleanToCrackedRatio": a.clean_to_cracked_ratio, "cellBuffer": a.cell_buffer, "baseModel": a.base_model, "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"), "validation": final, "precisionRecallCurve": pr_curve, "msPerCell": report["msPerCell"]}
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
    p.add_argument("--target-recall", type=float, default=0.95)  # legacy compatibility
    p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--clean-to-cracked-ratio", type=float, default=100.0)
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
