# PV CrackVision v1.4.3 validation

## Implemented

- Training Centre now displays a per-epoch table captured after the Ultralytics validation pass: precision, recall, mAP50, mAP50-95, training/validation box, class and DFL losses, and timestamp.
- Added a live persisted log panel, full raw stdout/stderr worker log download, and epoch-metrics CSV download.
- Job state retains up to 500 epoch records and 300 recent log entries; refresh recovery keeps the same data.
- Runtime setup now probes the existing Python environment and verifies `torch`, `torchvision`, and `ultralytics` imports. It skips installation when healthy, no longer upgrades pip on every setup, installs only missing packages, and attempts repair if installed packages fail import verification.
- Subsequent YOLO cell training initializes from the most recent app-trained `best.pt` checkpoint when found. A validation-passing checkpoint is copied to the stable path `training/models/cell-best.pt`, with `cell-best.previous.pt` retained on replacement. A unique run folder is created for every job.
- Retained `server/models/cell.onnx` as the active inference model and preserved previous ONNX weights until validation and export complete.
- Full worker tracebacks are retained in job state on failure. If an epoch callback produces no per-epoch mAP50 row, the candidate is rejected rather than completing with an empty analytics table.
- Detection status now reports classical fallback when the ONNX model is present but `onnxruntime-node` is missing; inference fallback reasons are included in inspection warnings.
- Setup and training-status polling errors are surfaced in the UI instead of being swallowed silently.

## Automated validation

- Node test suite: **24/24 passed** (`node --experimental-strip-types --test` across the seven included test files).
- Python worker unit tests: **7/7 passed** (`python -m unittest training.test_job_state`).
- Python syntax: **passed** (`python -m py_compile training/train_cell_yolo.py`).
- TypeScript/TSX syntax transpilation: **4/4 passed** (`src/App.tsx`, `server/cellTraining.ts`, `server/cellDetector.ts`, `server.ts`) using TypeScript transpilation diagnostics.
- Shell launcher syntax: **passed** (`bash -n start.sh`).
- ZIP integrity: checked after packaging.

## Not verified in this environment

- Full `npm ci` timed out, so the complete Vite production build and semantic TypeScript typecheck could not be completed; required `@types/node` and `vite/client` dependencies were unavailable.
- No live YOLO training job, Windows worker-log file capture, Windows restart/refresh scenario, ONNX export, or ONNX inference was run in this packaging environment. Those need the application's Windows training runtime and the user's local workstation.
- Per-epoch callback timing follows Ultralytics' documented `on_fit_epoch_end` hook, which runs after validation and checkpoint save. The callback's final best-model evaluation is ignored to avoid duplicating or overwriting the last training epoch. See https://docs.ultralytics.com/usage/callbacks.
