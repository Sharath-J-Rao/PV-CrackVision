#!/usr/bin/env python3
"""
v1.4.17 crack ANOMALY detector (PatchCore-style), trained from the reviewed feedback store.

Only CLEAN cells are fitted. A frozen pretrained backbone (yolo11-cls) turns every cell crop into a grid of patch vectors; a greedy coreset of
the clean patches becomes the memory bank. A cell's score is the mean distance of its K most unusual patches to their nearest clean patch, so
anything the clean cells never showed (a crack) scores high. Cracked labels are never fitted: they only validate the decision threshold.

The whole scorer (backbone + memory bank + distance + sigmoid) is exported as ONE ONNX graph with a [N,2] = [1-p, p] output, so the Node app scores
it with the same code path as the archived classifier. p = sigmoid((score - threshold) / scale): 0.5 is the trained threshold.
State file, whole-image split, crops and metrics are shared with train_crack_classifier.py so both trainers behave identically.
"""
import argparse, json, math, os, random, sys, time, traceback
from pathlib import Path

# The app-managed embedded Python (python312._pth) does NOT put this script's folder on sys.path, so the sibling import below failed before
# any state was written ("exited unexpectedly (code 1)"). Add the folder explicitly; harmless for normal Pythons and for the unit tests.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:  # shared: write_state/STATE, load_feedback, split_by_image, open_image, buffered_crop, metrics_at, atomic_copy
    import train_crack_classifier as base  # run as a script from training/
except ModuleNotFoundError:
    from training import train_crack_classifier as base  # imported as a package (unit tests)

FEATURE_LAYERS = (4, 6)  # yolo11-cls backbone: stride-8 and stride-16 feature maps (local texture + cell structure)
POOL_FACTOR = 4          # candidate patches sampled per kept memory-bank vector before the greedy coreset
ONNX_BATCH = 8           # batch hint for the Node scorer: each batch builds a (patches x memory bank) distance matrix


# ----------------------------------------------------------------------------- pure numpy helpers (no torch needed; unit-tested)
def pick_device():
    """v1.4.17.7: CUDA when this PyTorch build and machine have it, otherwise CPU (the crack detector trains on the GPU when available)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def greedy_coreset(features, size, seed, progress=None, device="cpu"):
    """k-center greedy subset: repeatedly keeps the patch farthest from everything kept so far, so rare-but-normal textures stay represented."""
    import numpy as np
    n = len(features)
    if n <= size:
        return features
    rng = np.random.default_rng(seed)
    z = features if features.shape[1] <= 128 else features @ (rng.standard_normal((features.shape[1], 128)).astype("float32") / math.sqrt(128))
    chosen = [int(rng.integers(n))]
    if device != "cpu":  # same k-center greedy, distances computed on the GPU
        import torch
        zt = torch.as_tensor(z, dtype=torch.float32, device=device)
        nearest_t = ((zt - zt[chosen[0]]) ** 2).sum(1)
        for step in range(1, size):
            chosen.append(int(nearest_t.argmax()))
            nearest_t = torch.minimum(nearest_t, ((zt - zt[chosen[-1]]) ** 2).sum(1))
            if progress and step % 500 == 0:
                progress(step / size)
        return features[chosen]
    nearest = ((z - z[chosen[0]]) ** 2).sum(1)
    for step in range(1, size):
        chosen.append(int(nearest.argmax()))
        np.minimum(nearest, ((z - z[chosen[-1]]) ** 2).sum(1), out=nearest)
        if progress and step % 500 == 0:
            progress(step / size)
    return features[chosen]


def auroc(y, p):
    """Probability that a random cracked cell scores above a random clean one (rank statistic); None when a class is missing."""
    import numpy as np
    y, p = np.asarray(y), np.asarray(p)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return None
    # Average ranks so tied scores (saturated sigmoid outputs, identical patches) count 0.5, not an arbitrary order.
    uniq, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def split_images(grouped, val_fraction, seed):
    """Whole-image split. Prefer the shared two-class split; with too few cracked images fall back to splitting the images that have clean cells."""
    try:
        return base.split_by_image(grouped, val_fraction, seed)
    except ValueError:
        clean = sorted(name for name, rows in grouped.items() if any(r["label"] == 0 for r in rows))
        if len(clean) < 2:
            raise ValueError("At least two source images with clean cells are required for an image-level train/validation split.")
        random.Random(seed).shuffle(clean)
        k = max(1, min(len(clean) - 1, round(len(clean) * min(0.5, max(0.05, val_fraction)))))
        return clean[k:], clean[:k]


def choose_threshold(clean_scores, crack_scores, quantile, target_precision, ratio):
    """(threshold, mode). With >= 3 held-out cracked cells: lowest threshold whose precision (reweighted to the field ratio) meets the target, else the
    best-precision one ('precision'). Otherwise the `quantile` of held-out CLEAN scores ('clean-quantile'): the allowed false-alarm rate, crack recall unverified."""
    import numpy as np
    clean_scores = np.asarray(clean_scores, dtype="float64")
    if len(crack_scores) < 3:
        return float(np.quantile(clean_scores, quantile)), "clean-quantile"
    p = np.concatenate([clean_scores, np.asarray(crack_scores, dtype="float64")])
    y = np.concatenate([np.zeros(len(clean_scores), int), np.ones(len(crack_scores), int)])
    rows = [(t, base.metrics_at(y, p, t, ratio)) for t in np.unique(p)]
    levels = np.unique(p)

    def inside(t):
        """Midpoint to the next lower score: a threshold sitting exactly on a validation score flips with 1e-6 torch/ONNX differences (non-reproducible recall)."""
        lower = levels[levels < t]
        return float((t + lower[-1]) / 2.0) if len(lower) else float(t)
    ok = [t for t, m in rows if m["precisionAtRealisticRatio"] >= target_precision]
    if ok:
        return inside(min(ok)), "precision"
    return inside(max(rows, key=lambda r: (r[1]["precisionAtRealisticRatio"], r[1]["recall"]))[0]), "precision-unmet"


# ----------------------------------------------------------------------------- crops
def count_cells(grouped, names, label):
    return sum(1 for name in names for it in grouped[name] if it["label"] == label)


def iter_crops(grouped, names, labels, feedback_dir, imgsz, buffer, batch):
    """Yield (labels, float32 NCHW 0..1) batches of buffered cell crops squashed to imgsz, exactly like the Node path (one decode per source image)."""
    import numpy as np
    from PIL import Image
    xs, ys = [], []
    for name in names:
        img = base.open_image(Path(feedback_dir) / "images" / name)
        w, h = img.size
        for it in grouped[name]:
            if it["label"] not in labels:
                continue
            l, t, r, b = base.buffered_crop(w, h, it, buffer)
            if r - l < 8 or b - t < 8:
                continue
            xs.append(np.asarray(img.crop((l, t, r, b)).resize((imgsz, imgsz), Image.LANCZOS), dtype="float32").transpose(2, 0, 1) / 255.0)
            ys.append(it["label"])
            if len(xs) == batch:
                yield np.stack(xs), np.asarray(ys)
                xs, ys = [], []
    if xs:
        yield np.stack(xs), np.asarray(ys)


# ----------------------------------------------------------------------------- scorer
def build_scorer(backbone, bank, top_k, threshold=0.0, scale=1.0):
    """Torch module: crops [B,3,S,S] -> raw anomaly score [B] (`raw`) or [1-p, p] (`forward`). `bank` is an [M, D] numpy array."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(list(backbone.model.model)[: max(FEATURE_LAYERS) + 1]).eval().requires_grad_(False)
            bank_t = torch.as_tensor(bank, dtype=torch.float32)
            self.register_buffer("bank", bank_t)
            self.register_buffer("bank_sq", (bank_t ** 2).sum(1))
            self.register_buffer("threshold", torch.tensor(float(threshold)))
            self.register_buffer("scale", torch.tensor(float(scale)))

        def patches(self, x):
            """[B, P, D]: stride-8 map pooled to stride 16, joined with the stride-16 map, 3x3 neighbourhood-averaged (PatchCore local aggregation)."""
            maps = {}
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in FEATURE_LAYERS:
                    maps[i] = x
            low, high = maps[FEATURE_LAYERS[0]], maps[FEATURE_LAYERS[1]]
            f = torch.cat([F.avg_pool2d(low, low.shape[-1] // high.shape[-1]), high], 1)
            return F.avg_pool2d(f, 3, 1, 1).flatten(2).transpose(1, 2)

        def raw(self, x):
            f = self.patches(x)
            d2 = (f * f).sum(-1, keepdim=True) + self.bank_sq - 2.0 * (f @ self.bank.T)  # |a|^2 + |b|^2 - 2ab: plain MatMul in ONNX
            nearest = d2.min(-1).values.clamp_min(0).sqrt()  # [B, P] distance of each patch to its closest clean patch
            return nearest.topk(min(top_k, nearest.shape[-1]), dim=-1).values.mean(-1)  # a small crop can have fewer patches than K

        def forward(self, x):
            p = torch.sigmoid((self.raw(x) - self.threshold) / self.scale)
            return torch.stack([1.0 - p, p], 1)

    return Scorer().eval()


def export_onnx(scorer, imgsz, path):
    import torch
    args = (scorer, torch.zeros(1, 3, imgsz, imgsz), str(path))
    kwargs = dict(input_names=["images"], output_names=["output0"], dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(*args, dynamo=False, **kwargs)  # TorchScript exporter: stable for MatMul/TopK graphs
    except TypeError:  # older torch has no `dynamo` argument and only the TorchScript exporter
        torch.onnx.export(*args, **kwargs)


# ----------------------------------------------------------------------------- training
def run(a):
    import numpy as np
    import torch
    from ultralytics import YOLO
    grouped = base.load_feedback(a.feedback_dir)
    if len(grouped) < 2:
        raise RuntimeError(f"Need at least 2 reviewed images, found {len(grouped)}.")
    train_imgs, val_imgs = split_images(grouped, a.val_fraction, a.seed)
    fit_cells = count_cells(grouped, train_imgs, 0)
    if fit_cells < 20:
        raise RuntimeError(f"Only {fit_cells} clean cells in the training images; label more clean cells (at least 50).")
    base.write_state(log_line=f"{len(grouped)} images: {len(train_imgs)} fit (clean cells only: {fit_cells}) / {len(val_imgs)} held-out (whole images).")

    torch.manual_seed(a.seed)
    device = pick_device()
    base.write_state(log_line=f"Compute device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no CUDA GPU available; training on the CPU)"))
    backbone = YOLO(a.backbone)  # pretrained ImageNet weights, frozen
    scorer = build_scorer(backbone, np.zeros((1, 1), "float32"), a.top_k).to(device)  # placeholder bank: only `patches` is used while fitting
    # 1) clean patches from the training images, sampled down to a candidate pool
    patches_per_cell = (a.imgsz // 16) ** 2
    keep = min(1.0, POOL_FACTOR * a.memory_bank / (fit_cells * patches_per_cell))
    rng, pool, done = np.random.default_rng(a.seed), [], 0
    with torch.no_grad():
        for x, _ in iter_crops(grouped, train_imgs, (0,), a.feedback_dir, a.imgsz, a.cell_buffer, 16):
            f = scorer.patches(torch.from_numpy(x).to(device)).flatten(0, 1).cpu().numpy()
            pool.append(f[rng.random(len(f)) < keep])
            done += len(x)
            base.write_state(progress=5 + 40 * done / fit_cells, message=f"Extracting clean-cell features {done}/{fit_cells}")
    pool = np.concatenate(pool).astype("float32")
    # 2) greedy coreset = memory bank
    base.write_state(progress=48, message=f"Building memory bank from {len(pool)} clean patches.", log_line=f"Candidate pool {len(pool)} patches x {pool.shape[1]} dims")
    bank = greedy_coreset(pool, a.memory_bank, a.seed, lambda r: base.write_state(progress=48 + 17 * r), device)
    del pool
    scorer = build_scorer(backbone, bank, a.top_k).to(device)
    # 3) held-out raw scores -> threshold
    raw, ys = [], []
    with torch.no_grad():
        for x, y in iter_crops(grouped, val_imgs, (0, 1), a.feedback_dir, a.imgsz, a.cell_buffer, ONNX_BATCH):
            raw.extend(scorer.raw(torch.from_numpy(x).to(device)).cpu().tolist()); ys.extend(y.tolist())
            base.write_state(progress=65 + 10 * min(1.0, len(ys) / max(1, sum(len(grouped[n]) for n in val_imgs))), message=f"Scoring held-out cells {len(ys)}")
    raw, ys = np.asarray(raw), np.asarray(ys)
    clean, crack = raw[ys == 0], raw[ys == 1]
    if len(clean) < 10:
        raise RuntimeError(f"Only {len(clean)} clean cells in the held-out images; label more clean cells across more images.")
    threshold, mode = choose_threshold(clean, crack, a.clean_quantile, a.target_precision, a.clean_to_cracked_ratio)
    scale = max(float(clean.std()), 0.05 * threshold, 1e-6)
    scorer = build_scorer(backbone, bank, a.top_k, threshold, scale).cpu()  # export on the CPU: the ONNX graph is device independent
    # 4) export, then validate the EXPORTED file (the one the app runs) against the torch scores
    base.write_state(stage="export", status="validating", progress=78, message="Exporting ONNX and validating the exported file on held-out images.", log_line=f"Threshold {threshold:.4f} ({mode}), scale {scale:.4f}, bank {len(bank)} patches")
    onnx = Path(a.work) / "cellAnomaly.onnx"
    export_onnx(scorer, a.imgsz, onnx)
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    name, probs, start = session.get_inputs()[0].name, [], time.perf_counter()
    for x, _ in iter_crops(grouped, val_imgs, (0, 1), a.feedback_dir, a.imgsz, a.cell_buffer, ONNX_BATCH):
        probs.extend(session.run(None, {name: x})[0][:, 1].tolist())
    ms = (time.perf_counter() - start) * 1000 / max(1, len(probs))
    probs = np.asarray(probs)
    expected = 1.0 / (1.0 + np.exp(-(raw - threshold) / scale))
    if len(probs) != len(raw) or float(np.abs(probs - expected).max()) > 1e-2:
        raise RuntimeError("The exported ONNX scores differ from the trained model; the previous model was kept.")

    cracks_verified = len(crack) >= 3
    final = base.metrics_at(ys, probs, 0.5, a.clean_to_cracked_ratio)
    accepted = final["balancedAccuracy"] >= a.min_balanced_accuracy if cracks_verified else True
    pr_curve = base.precision_recall_curve(ys, probs, a.clean_to_cracked_ratio) if cracks_verified else []
    version = f"cell-anomaly-{Path(a.backbone).stem}-{time.strftime('%Y%m%d%H%M')}"
    note = {"precision": "Threshold chosen on held-out cracked + clean cells at the target operational precision. ",
            "precision-unmet": "Target operational precision NOT reached on this validation set; the best-precision threshold was used. ",
            "clean-quantile": "No (or fewer than 3) held-out cracked cells: threshold set from held-out CLEAN cells only; crack recall is NOT verified. "}[mode]
    report = {"accepted": accepted, "version": version, "validation": final, "thresholdMode": mode, "cracksVerified": cracks_verified, "auroc": auroc(ys, raw),
              "rawThreshold": threshold, "scoreScale": scale, "memoryBank": len(bank), "targetPrecision": a.target_precision, "cleanToCrackedRatio": a.clean_to_cracked_ratio,
              "cleanQuantile": a.clean_quantile, "fitCells": fit_cells, "valCells": {"clean": int(len(clean)), "cracked": int(len(crack))}, "precisionRecallCurve": pr_curve,
              "msPerCell": round(ms, 2), "trainImages": len(train_imgs), "valImages": len(val_imgs),
              "evaluationNote": note + "Validation uses whole source images; no cell from a fitted image is held out."}
    if accepted:
        base.atomic_copy(onnx, a.activate)
        meta = {"schema": "cell-anomaly-v1", "version": version, "imgsz": a.imgsz, "classNames": ["clean", "anomalous"], "crackedIndex": 1, "threshold": 0.5, "rawThreshold": threshold,
                "scoreScale": scale, "topK": a.top_k, "memoryBank": len(bank), "thresholdMode": mode, "cracksVerified": cracks_verified, "cellBuffer": a.cell_buffer, "batchSize": ONNX_BATCH,
                "baseModel": a.backbone, "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"), "validation": final, "precisionRecallCurve": pr_curve, "msPerCell": report["msPerCell"]}
        Path(f"{a.meta_activate}.tmp").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(f"{a.meta_activate}.tmp", a.meta_activate)  # meta last: the app treats a model without meta as absent
    return report


def parse(argv=None):
    p = argparse.ArgumentParser()
    for flag, kind, default in (("--feedback-dir", str, None), ("--work", str, None), ("--state-file", str, None), ("--activate", str, None), ("--meta-activate", str, None),
                                ("--backbone", str, "yolo11n-cls.pt"), ("--imgsz", int, 224), ("--memory-bank", int, 8000), ("--top-k", int, 3), ("--val-fraction", float, 0.2),
                                ("--clean-quantile", float, 0.995), ("--target-precision", float, 0.95), ("--clean-to-cracked-ratio", float, 100.0),
                                ("--min-balanced-accuracy", float, 0.8), ("--seed", int, 42), ("--cell-buffer", float, 0.03)):
        p.add_argument(flag, type=kind, default=default, required=default is None)
    a = p.parse_args(argv)
    if a.imgsz % 32:
        p.error("--imgsz must be a multiple of 32")
    return a


def main():
    a = parse()
    base.STATE = a.state_file
    Path(a.work).mkdir(parents=True, exist_ok=True)
    try:
        base.write_state(status="preparing", stage="preparing", mode="anom", pid=os.getpid(), progress=1, message="Reading reviewed labels.", metrics=[], log=[])
        report = run(a)
        done = "Held-out validation passed; anomaly detector activated." if report["accepted"] else "Held-out validation did not meet the gate; the previous model was kept."
        base.write_state(status="completed", stage="completed", progress=100, message=done, report=report, finishedAt=time.strftime("%Y-%m-%dT%H:%M:%S"), log_line=done)
        return 0
    except BaseException as exc:  # includes SystemExit/KeyboardInterrupt so the state never stays "training" after a crash
        base.write_state(status="failed", stage="failed", message=str(exc), error=str(exc), errorTraceback=traceback.format_exc(), finishedAt=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
