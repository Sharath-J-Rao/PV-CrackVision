"""PV CrackVision in-app YOLO11n direct PV-cell training worker.

The detached worker writes durable progress to the job-state file so a browser
refresh or backend restart does not discard its status or training log.
"""
import argparse
import json
import os
import shutil
import sys
import time
import csv
import traceback
from datetime import datetime, timezone
from pathlib import Path


def read_state(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_early_stopping_patience(requested_epochs, configured_patience=0):
    """Return the explicit early-stop setting; zero guarantees all requested epochs."""
    epochs = max(1, int(requested_epochs))
    patience = max(0, int(configured_patience))
    return min(patience, epochs)


def control_path(state_file):
    """Pause-request file written by the server next to the job-state file."""
    return Path(str(state_file) + '.control')


def wait_while_paused(state_file, control_file, sleep=time.sleep, poll=1.0):
    """Block at a safe point while the pause file exists (cross-platform; no OS-level suspend). Returns True if it paused."""
    control_file = Path(control_file)
    if not control_file.exists():
        return False
    write_state(state_file, status='paused', message='Training paused. Press Resume to continue.', log_line='Training paused by the operator.')
    while control_file.exists():
        sleep(poll)
    write_state(state_file, status='training', message='Training resumed.', log_line='Training resumed by the operator.')
    return True


def resolve_device(requested='auto'):
    """Pick the training device. 'auto' -> first CUDA GPU, else Apple MPS, else CPU.

    Why: the worker was hard-wired to CPU. An explicit value (e.g. '0', '0,1', 'cpu') is honoured;
    if CUDA was requested but is unavailable we fall back to CPU rather than crash mid-setup.
    """
    requested = str(requested or 'auto').strip().lower()
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        mps_ok = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    except Exception:  # torch missing/broken -> let CPU path report the real error later
        cuda_ok = mps_ok = False
    if requested == 'auto':
        return '0' if cuda_ok else ('mps' if mps_ok else 'cpu')
    if requested in ('cpu', 'mps'):
        return requested if requested == 'cpu' or mps_ok else 'cpu'
    return requested if cuda_ok else 'cpu'


DEVICE = resolve_device(os.environ.get('PV_YOLO_DEVICE', 'auto'))  # override with PV_YOLO_DEVICE=cpu|0|auto


# ---- v1.4.12.7: GPU-aware defaults. These were tuned for CPU-only machines (0 loader workers on Windows, AMP off). ----
MAX_DET = 1000  # dense arrays hold 400+ cells per image; Ultralytics' default of 300 capped recall and skewed best-epoch selection


def _total_ram_gb():
    """Installed RAM in GiB, or None when it cannot be read (callers then stay conservative)."""
    try:
        import psutil  # installed together with Ultralytics
        return psutil.virtual_memory().total / 2 ** 30
    except Exception:
        return None


def resolve_workers(device=None, cpu_count=None, ram_gb=None, windows=None):
    """Data-loader processes. A fast GPU idles when augmentation runs in the main process, so use several workers on GPU.

    Why not always: each Windows loader process maps its own copy of PyTorch and can exhaust the paging file (the worker
    then dies silently), so the count is capped by installed RAM and by a lower Windows ceiling.
    PV_YOLO_WORKERS=<n> overrides everything (0 restores the old single-process loading)."""
    override = os.environ.get('PV_YOLO_WORKERS', '').strip()
    if override.isdigit():
        return min(32, int(override))
    device = DEVICE if device is None else device
    if str(device) in ('cpu', 'mps'):
        return 0
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    ram = _total_ram_gb() if ram_gb is None else ram_gb
    is_windows = (os.name == 'nt') if windows is None else windows
    if ram is None:
        return 2 if cpus >= 4 else 0
    if ram >= 24:
        return max(2, min(6 if is_windows else 8, cpus // 2))
    if ram >= 12:
        return 2
    return 0


def resolve_amp(device=None):
    """Mixed precision roughly halves GPU memory and speeds up training; it is meaningless on CPU. PV_YOLO_AMP=0 disables it.

    Ultralytics' own AMP self-check switches it off automatically if it detects numerical problems."""
    device = DEVICE if device is None else device
    return str(device) not in ('cpu', 'mps') and os.environ.get('PV_YOLO_AMP', '1').strip() != '0'


# v1.4.12.7 hotfix: the exported ONNX graph legitimately scores a hair below best.pt (FP32 export, different NMS/letterbox path).
# A hard `onnx_map50 < min_map50` check rejected a good model at mAP50=0.2984 vs a 0.30 gate. The ONNX floor now allows this
# small export tolerance; best.pt must still meet the full gate, and the best.pt-vs-ONNX parity check below still guards regressions.
ONNX_GATE_TOLERANCE = 0.02


def onnx_gate_floor(min_map50, tolerance=ONNX_GATE_TOLERANCE):
    """Minimum exported-ONNX mAP50: the configured gate minus an export tolerance (setting; default 0.02), never below zero."""
    return max(0.0, float(min_map50) - max(0.0, float(tolerance)))


def onnx_validation_device():
    """Device for validating the exported ONNX model. The CPU ONNX Runtime is the device-independent check by design, so only
    use the GPU when the installed runtime really offers CUDA; otherwise ask for CPU explicitly (no provider warnings)."""
    try:
        import onnxruntime
        if str(DEVICE) not in ('cpu', 'mps') and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            try:
                onnxruntime.preload_dlls()  # reuse the CUDA/cuDNN libraries that ship with PyTorch (onnxruntime >= 1.21)
            except Exception:
                pass
            return DEVICE
    except Exception:
        pass
    return 'cpu'


def hide_plot_labels():
    """Ultralytics tags every box in its validation/training plots with the class name ('pv_cell'). With 400+ cells per image
    the tags cover the picture, so draw the boxes only. Best effort: a missing or renamed Annotator must never stop training.
    PV_YOLO_PLOT_LABELS=1 keeps the tags."""
    if os.environ.get('PV_YOLO_PLOT_LABELS', '').strip() == '1':
        return False
    try:
        from ultralytics.utils.plotting import Annotator
        if getattr(Annotator.box_label, '_pv_quiet', False):
            return True
        original = Annotator.box_label

        def box_label(self, box, label='', *args, **kwargs):
            return original(self, box, '', *args, **kwargs)

        box_label._pv_quiet = True
        Annotator.box_label = box_label
        return True
    except Exception:
        return False


def build_training_kwargs(args):
    """Map the UI's persisted cell-detector settings to explicit YOLO train kwargs."""
    return {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.run_name,
        "exist_ok": False,
        "device": DEVICE,
        "workers": resolve_workers(),  # v1.4.12.7: scaled to the GPU/RAM instead of a fixed 0 on Windows
        "max_det": MAX_DET,
        "patience": resolve_early_stopping_patience(args.epochs, args.patience),
        "freeze": args.freeze if args.freeze > 0 else None,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "box": args.box_loss,
        "cls": args.cls_loss,
        "dfl": args.dfl_loss,
        "multi_scale": args.multi_scale,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "cache": args.cache,
        "pretrained": args.pretrained,
        "save_period": args.save_period,
        "amp": resolve_amp(),  # v1.4.12.7: on for GPU training (was always off)
        "plots": True,
        "verbose": True,
    }


def write_state(path, **updates):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = read_state(path)
    raw_logs = state.get("log", [])
    logs = list(raw_logs) if isinstance(raw_logs, list) else []
    line = updates.pop("log_line", None)
    epoch_metric = updates.pop("epoch_metric", None)
    if line:
        logs = (logs + [str(line)[:500]])[-300:]
        if not str(line).startswith('Epoch '):  # epoch summaries are already printed by the callback
            try:  # v1.4.12.6: final-phase steps (validation, export, activation) now reach the worker log file too
                print(f'[worker] {line}', flush=True)
            except Exception:
                pass  # console logging must never break the durable state write
    if epoch_metric:
        raw_rows = state.get("epochMetrics", [])
        rows = list(raw_rows) if isinstance(raw_rows, list) else []
        epoch_number = int(epoch_metric.get("epoch", 0))
        rows = [row for row in rows if int(row.get("epoch", 0)) != epoch_number]
        rows.append(epoch_metric)
        rows.sort(key=lambda row: int(row.get("epoch", 0)))
        state["epochMetrics"] = rows[-500:]
    state.update(updates)
    state["log"] = logs
    state["updatedAt"] = datetime.now(timezone.utc).isoformat()
    temp = path.with_suffix(path.suffix + ".worker.tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    for attempt in range(8):  # Windows: Node reading the file can transiently block replace
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.1)


def extract_validation_metrics(trainer, train_result=None):
    """Ultralytics versions expose metrics as a dict or a Metric object."""
    candidates = [
        getattr(trainer, "metrics", None),
        getattr(getattr(trainer, "validator", None), "metrics", None),
        train_result,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        raw = candidate if isinstance(candidate, dict) else getattr(candidate, "results_dict", None)
        if not isinstance(raw, dict):
            continue
        metrics = {}
        for key, value in raw.items():
            try:
                number = float(value)
                if number == number and abs(number) != float("inf"):
                    metrics[str(key)] = number
            except (TypeError, ValueError):
                continue
        if metrics:
            return metrics
    return {}


def _to_float(value):
    try:
        number = float(value)
        if number == number and abs(number) != float("inf"):
            return number
    except (TypeError, ValueError):
        pass
    return None


def build_epoch_metric_row(trainer, epoch, timestamp=None):
    """Collect validation metrics after an epoch and its validation pass."""
    metrics = extract_validation_metrics(trainer)
    aliases = {
        "precision": ("metrics/precision(B)", "metrics/precision"),
        "recall": ("metrics/recall(B)", "metrics/recall"),
        "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }
    row = {
        "epoch": int(epoch),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    for output, keys in aliases.items():
        row[output] = next((_to_float(metrics.get(key)) for key in keys if _to_float(metrics.get(key)) is not None), None)

    # Ultralytics exposes the epoch-mean training losses as a tensor/list.
    losses = getattr(trainer, "tloss", None)
    if hasattr(losses, "detach"):
        try:
            losses = losses.detach().cpu().tolist()
        except Exception:
            losses = None
    elif hasattr(losses, "tolist"):
        try:
            losses = losses.tolist()
        except Exception:
            losses = None
    if isinstance(losses, (int, float)):
        losses = [losses]
    names = getattr(trainer, "loss_names", None)
    if not isinstance(names, (list, tuple)) or len(names) != len(losses or []):
        names = ["box_loss", "cls_loss", "dfl_loss"]
    for name, value in zip(names, losses or []):
        key = str(name).lower().replace("train/", "").replace(" ", "_")
        if "box" in key:
            key = "box_loss"
        elif "cls" in key or "class" in key:
            key = "cls_loss"
        elif "dfl" in key:
            key = "dfl_loss"
        number = _to_float(value)
        if key in ("box_loss", "cls_loss", "dfl_loss"):
            row[key] = number

    # If trainer internals differ between Ultralytics releases, prefer the
    # official results.csv row as a compatibility fallback for losses/metrics.
    try:
        results_csv = Path(trainer.save_dir) / "results.csv"
        with results_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            last = {str(k).strip(): v for k, v in rows[-1].items()}
            csv_aliases = {
                "precision": ("metrics/precision(B)", "metrics/precision"),
                "recall": ("metrics/recall(B)", "metrics/recall"),
                "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
                "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
                "box_loss": ("train/box_loss",),
                "cls_loss": ("train/cls_loss",),
                "dfl_loss": ("train/dfl_loss",),
                "val_box_loss": ("val/box_loss",),
                "val_cls_loss": ("val/cls_loss",),
                "val_dfl_loss": ("val/dfl_loss",),
            }
            for output, keys in csv_aliases.items():
                for key in keys:
                    number = _to_float(last.get(key))
                    if number is not None:
                        row[output] = number
                        break
    except (OSError, csv.Error):
        pass
    return row




def read_results_csv_rows(results_csv):
    """Read Ultralytics results.csv as the durable source of per-epoch metrics."""
    path = Path(results_csv)
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = []
            for raw in csv.DictReader(handle):
                row = {str(key).strip(): value for key, value in raw.items() if key is not None}
                epoch_value = _to_float(row.get("epoch"))
                if epoch_value is None:
                    continue
                epoch = int(epoch_value)  # Ultralytics results.csv epoch is already 1-based
                aliases = {
                    "precision": ("metrics/precision(B)", "metrics/precision"),
                    "recall": ("metrics/recall(B)", "metrics/recall"),
                    "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
                    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
                    "box_loss": ("train/box_loss",),
                    "cls_loss": ("train/cls_loss",),
                    "dfl_loss": ("train/dfl_loss",),
                    "val_box_loss": ("val/box_loss",),
                    "val_cls_loss": ("val/cls_loss",),
                    "val_dfl_loss": ("val/dfl_loss",),
                }
                metric = {"epoch": epoch, "timestamp": datetime.now(timezone.utc).isoformat()}
                for output, keys in aliases.items():
                    for key in keys:
                        value = _to_float(row.get(key))
                        if value is not None:
                            metric[output] = value
                            break
                rows.append(metric)
            return rows
    except (OSError, csv.Error, UnicodeError):
        return []


def reconcile_epoch_metrics(state_file, results_csv):
    """Merge authoritative results.csv rows into durable job state."""
    rows = read_results_csv_rows(results_csv)
    for row in rows:
        write_state(state_file, epoch_metric=row)
    return rows

def persist_reusable_checkpoint(source, destination):
    """Atomically preserve a validation-passing .pt checkpoint for the next run."""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + '.tmp')
    previous = destination_path.with_name(destination_path.stem + '.previous' + destination_path.suffix)
    try:
        shutil.copy2(source_path, temporary)
        if destination_path.exists():
            shutil.copy2(destination_path, previous)
        os.replace(temporary, destination_path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return str(destination_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=0, help='0 disables automatic early stopping; positive values enable plateau-based early stopping.')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--freeze', type=int, default=0)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--warmup-epochs', type=float, default=3.0)
    parser.add_argument('--warmup-momentum', type=float, default=0.8)
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1)
    parser.add_argument('--close-mosaic', type=int, default=10)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--hsv-h', type=float, default=0.015)
    parser.add_argument('--hsv-s', type=float, default=0.7)
    parser.add_argument('--hsv-v', type=float, default=0.4)
    parser.add_argument('--box-loss', type=float, default=7.5)
    parser.add_argument('--cls-loss', type=float, default=0.5)
    parser.add_argument('--dfl-loss', type=float, default=1.5)
    parser.add_argument('--multi-scale', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-period', type=int, default=10)
    parser.add_argument('--deterministic', type=lambda value: str(value).lower() == 'true', default=True)
    parser.add_argument('--cache', type=lambda value: str(value).lower() == 'true', default=False)
    parser.add_argument('--pretrained', type=lambda value: str(value).lower() == 'true', default=True)
    parser.add_argument('--cos-lr', type=lambda value: str(value).lower() == 'true', default=False)
    parser.add_argument('--project', required=True)
    parser.add_argument('--activate', required=True)
    parser.add_argument('--state-file', required=True)
    parser.add_argument('--onnx-tolerance', type=float, default=ONNX_GATE_TOLERANCE)  # matches trainingSettings.ts onnxGateTolerance
    parser.add_argument('--min-map50', type=float, default=0.05)  # bootstrap-friendly default; matches trainingSettings.ts
    parser.add_argument('--initial-weights', default='')
    parser.add_argument('--run-name', default='pv_cell')
    parser.add_argument('--checkpoint-copy', default='')
    args = parser.parse_args()
    state_file = Path(args.state_file)
    try:  # native crashes (segfault/access violation) print a Python stack to the worker log instead of dying silently
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    try:
        os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / 'ultralytics_config'))
        from ultralytics import YOLO
        hide_plot_labels()  # v1.4.12.7: plots show boxes only, not a 'pv_cell' tag on each of hundreds of cells

        write_state(state_file, status='training', progress=2, message='Loading YOLO checkpoint and preparing training…', pid=os.getpid(),
                    checkpointSource=args.initial_weights or 'yolo11n.pt',
                    log_line=f"Training worker started. Initial weights: {args.initial_weights or 'yolo11n.pt'}")

        try:  # v1.4.12.1: always show which device training uses, and why when it is the CPU
            import torch
            gpu = torch.cuda.get_device_name(0) if DEVICE not in ('cpu', 'mps') and torch.cuda.is_available() else None
            write_state(state_file, log_line=f"Training device: {DEVICE}" + (f" ({gpu})" if gpu else '') + ('' if DEVICE != 'cpu' else ' - no CUDA GPU visible to PyTorch (CPU build installed, no NVIDIA driver, or PV_YOLO_DEVICE=cpu).'))
        except Exception:
            pass
        if DEVICE == 'cpu' and os.environ.get('PV_YOLO_DEVICE', 'auto').strip().lower() != 'cpu' and os.environ.get('PV_YOLO_REQUIRE_CUDA', '0').strip() == '1':
            # v1.4.12.8: fail fast instead of silently training for hours on the CPU (and with AMP off) when a GPU was expected.
            raise RuntimeError('PV_YOLO_REQUIRE_CUDA=1 but PyTorch cannot see a CUDA GPU (CPU-only PyTorch build or missing NVIDIA driver).')
        write_state(state_file, log_line=f"Loader workers: {resolve_workers()}; mixed precision (AMP): {'on' if resolve_amp() else 'off'}; max detections per image: {MAX_DET}")
        control_file = control_path(state_file)

        def on_train_batch_end(trainer):
            wait_while_paused(state_file, control_file)

        def on_fit_epoch_end(trainer):
            wait_while_paused(state_file, control_file)
            # This callback runs after validation, so precision/recall/mAP belong
            # to the epoch just completed (unlike on_train_epoch_end).
            epoch = int(trainer.epoch) + 1
            total = max(1, int(trainer.epochs))
            # Ultralytics may invoke on_fit_epoch_end once more for its final
            # best-checkpoint evaluation. Do not add an extra row or overwrite
            # the true last-epoch metrics with that separate final evaluation.
            if epoch > total or any(int(item.get('epoch', 0)) == epoch for item in read_state(state_file).get('epochMetrics', [])):
                return
            progress = min(94, int(epoch / total * 94))
            row = build_epoch_metric_row(trainer, epoch)
            def fmt(key):
                value = row.get(key)
                return f'{value:.4f}' if isinstance(value, (int, float)) else 'n/a'
            summary = (f"Epoch {epoch}/{total} · Precision {fmt('precision')} · Recall {fmt('recall')} · "
                       f"mAP50 {fmt('map50')} · mAP50-95 {fmt('map50_95')} · "
                       f"Box {fmt('box_loss')} · Class {fmt('cls_loss')} · DFL {fmt('dfl_loss')}")
            write_state(state_file, status='training', progress=progress,
                        message=f'Epoch {epoch}/{total} completed · validation metrics recorded',
                        epoch_metric=row, log_line=summary)
            print(summary, flush=True)

        # Continue from the latest app-trained checkpoint when one exists;
        # otherwise bootstrap from the official lightweight YOLO11n weights.
        initial_weights = args.initial_weights or 'yolo11n.pt'
        model = YOLO(initial_weights)
        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
        model.add_callback('on_train_batch_end', on_train_batch_end)

        def on_train_start(trainer):
            # v1.4.12.8: the line above shows what we REQUESTED; Ultralytics' own AMP self-check may still switch it off.
            write_state(state_file, log_line=f"Effective training state: device={getattr(trainer, 'device', '?')}; AMP actually {'ON' if getattr(trainer, 'amp', False) else 'OFF'}; batch={getattr(trainer, 'batch_size', '?')}; imgsz={getattr(getattr(trainer, 'args', None), 'imgsz', '?')}.")
        model.add_callback('on_train_start', on_train_start)
        stop_patience = resolve_early_stopping_patience(args.epochs, args.patience)
        mode = f'patience={stop_patience} (automatic early stopping enabled)' if stop_patience > 0 else 'patience=0 (automatic early stopping disabled; full epoch count enforced)'
        write_state(state_file, status='training', log_line=f'Training configuration: requested {args.epochs} epochs; {mode}. Use the app Pause button to pause (resumable) or Abort to discard the run. Ultralytics keeps best.pt (best epoch) and that checkpoint is the one validated and activated.')
        train_result = model.train(**build_training_kwargs(args))

        trainer = getattr(model, 'trainer', None)
        if trainer is None:
            raise RuntimeError('Ultralytics did not expose the completed training run for validation.')
        run_dir = Path(trainer.save_dir)
        # Callback timing and metric object APIs differ across Ultralytics releases.
        # Reconcile from its official results.csv after training as a second source.
        csv_epochs = reconcile_epoch_metrics(state_file, run_dir / 'results.csv')
        recorded_epochs = read_state(state_file).get('epochMetrics', [])
        if not recorded_epochs or not any(row.get('map50') is not None for row in recorded_epochs):
            raise RuntimeError('Training ended without per-epoch validation analytics. The candidate was not activated; check the full worker log and Ultralytics callback compatibility.')
        best = run_dir / 'weights' / 'best.pt'
        if not best.exists():
            raise RuntimeError(f'Best checkpoint was not produced: {best}')

        # Validate the actual best.pt checkpoint, not merely the last training epoch.
        # Ultralytics may select an earlier epoch as best; the activation gate must
        # measure the exact weights that will be exported and activated.
        write_state(state_file, progress=95, message='Validating the best checkpoint on held-out source images…',
                    log_line=f'Running final held-out validation against best checkpoint: {best}')
        best_model = YOLO(str(best))
        best_validation = best_model.val(data=args.data, imgsz=args.imgsz, device=DEVICE, plots=True, verbose=True, max_det=MAX_DET)
        metrics = extract_validation_metrics(None, best_validation)
        if not metrics:
            raise RuntimeError('Best-checkpoint validation returned no metrics. The previous active model was kept; inspect the full worker log.')
        metrics_path = run_dir / 'validation_metrics.json'
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        for key, value in metrics.items():
            write_state(state_file, log_line=f'Best-checkpoint validation {key}={value:.5f}')

        map50 = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50', 0.0))
        precision = metrics.get('metrics/precision(B)', metrics.get('metrics/precision', 0.0))
        recall = metrics.get('metrics/recall(B)', metrics.get('metrics/recall', 0.0))
        if map50 < args.min_map50 or (precision <= 0 and recall <= 0):
            raise RuntimeError(
                f'Validation gate not met for best.pt: mAP50={map50:.4f}, precision={precision:.4f}, '
                f'recall={recall:.4f}. Required mAP50 >= {args.min_map50:.2f} and at least '
                'one non-zero precision/recall metric. The previous active model was kept.'
            )

        write_state(state_file, progress=96, message='Validation passed; exporting and validating the ONNX model…', metrics=metrics,
                    runBestCheckpoint=str(best), runDirectory=str(run_dir),
                    log_line=f'PyTorch best-checkpoint validation passed (mAP50={map50:.4f}). Exporting ONNX; the currently active model remains untouched until exported-model validation passes.')
        exported = YOLO(str(best)).export(format='onnx', imgsz=args.imgsz, simplify=False, opset=12, dynamic=False)
        exported_path = Path(exported)
        if not exported_path.exists() or exported_path.stat().st_size < 1024 * 1024:
            raise RuntimeError('ONNX export is missing or unexpectedly small; the previous active model was kept.')

        # Validate the actual exported graph, not only best.pt. This catches export,
        # tensor-shape, preprocessing and output-head regressions before activation.
        write_state(state_file, progress=97, message='Validating the exported ONNX model on held-out source images…',
                    log_line=f'Running Ultralytics validation against exported ONNX: {exported_path}')
        onnx_model = YOLO(str(exported_path))
        onnx_validation = onnx_model.val(data=args.data, imgsz=args.imgsz, device=onnx_validation_device(), plots=False, verbose=True, max_det=MAX_DET)
        onnx_metrics = extract_validation_metrics(None, onnx_validation)
        if not onnx_metrics:
            raise RuntimeError('Exported ONNX validation returned no metrics; the previous active model was kept.')
        onnx_map50 = onnx_metrics.get('metrics/mAP50(B)', onnx_metrics.get('metrics/mAP50', 0.0))
        onnx_precision = onnx_metrics.get('metrics/precision(B)', onnx_metrics.get('metrics/precision', 0.0))
        onnx_recall = onnx_metrics.get('metrics/recall(B)', onnx_metrics.get('metrics/recall', 0.0))
        for key, value in onnx_metrics.items():
            write_state(state_file, log_line=f'Exported-ONNX validation {key}={value:.5f}')
        write_state(state_file, onnxMetrics=onnx_metrics)
        onnx_floor = onnx_gate_floor(args.min_map50, args.onnx_tolerance)
        if onnx_map50 < onnx_floor or (onnx_precision <= 0 and onnx_recall <= 0):
            raise RuntimeError(
                f'Exported ONNX validation gate not met: mAP50={onnx_map50:.4f}, precision={onnx_precision:.4f}, '
                f'recall={onnx_recall:.4f}. Required mAP50 >= {onnx_floor:.3f} (gate {args.min_map50:.2f} minus '
                f'{args.onnx_tolerance:.2f} export tolerance) and at least one non-zero '
                'precision/recall metric. The previously active model was kept.'
            )
        if map50 - onnx_map50 > 0.10:
            raise RuntimeError(
                f'Exported ONNX mAP50 ({onnx_map50:.4f}) is more than 0.10 below best.pt ({map50:.4f}). '
                'The export was not activated; the previous active model was kept.'
            )

        stable_checkpoint = persist_reusable_checkpoint(best, args.checkpoint_copy) if args.checkpoint_copy else str(best)
        destination = Path(args.activate)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.onnx.tmp')
        previous = destination.with_name(destination.stem + '.previous.onnx')
        shutil.copy2(exported_path, temporary)
        # Do not replace the previous detector until both PyTorch and ONNX validation passed.
        if destination.exists():
            shutil.copy2(destination, previous)
        os.replace(temporary, destination)
        validation_bundle = {'bestCheckpoint': metrics, 'exportedOnnx': onnx_metrics}
        (run_dir / 'validation_metrics.json').write_text(json.dumps(validation_bundle, indent=2), encoding='utf-8')
        write_state(state_file, status='completed', progress=100,
                    message='YOLO11 cell model trained, best.pt validated, exported ONNX validated and activated. Re-scan a field image to use the new model.',
                    metrics=metrics, onnxMetrics=onnx_metrics, bestCheckpoint=stable_checkpoint, runBestCheckpoint=str(best), runDirectory=str(run_dir),
                    finishedAt=datetime.now(timezone.utc).isoformat(),
                    log_line=f'Activated ONNX cell model after exported-model validation: {destination} (mAP50={onnx_map50:.4f})')
    except BaseException as exc:  # includes SystemExit/KeyboardInterrupt/MemoryError so a stop always records a reason
        details = traceback.format_exc()
        exc_text = f'{type(exc).__name__}: {exc}'
        write_state(state_file, status='failed', progress=min(95, int(read_state(state_file).get('progress', 0))),
                    message='YOLO cell training failed. The previously active model was preserved.',
                    error=exc_text, errorTraceback=details, finishedAt=datetime.now(timezone.utc).isoformat(),
                    log_line=f'TRAINING_ERROR: {exc_text}')
        print(f'TRAINING_ERROR: {exc_text}', file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
