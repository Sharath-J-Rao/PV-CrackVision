# PV CrackVision v1.4.2 validation

## Implemented

- Copy placement now supports `Space`: enter placement mode, move the cyan preview to the destination cell, then press Space to commit. Clicking remains available; Escape cancels. Space is captured before the browser page-scroll behavior while placement mode is active.
- Cell YOLO training runs in a detached local Python worker. The worker updates a durable state file with job ID, PID, progress, status, metrics, and a rolling training log.
- Browser refresh restores the last server-reported training state. If the Node web process restarts while the worker is alive, the app reads the persistent job file and reconnects to the worker's progress. A genuinely stopped worker is reported as interrupted rather than silently reverting to `not_ready`.
- Cancellation can signal the saved worker PID after a page refresh or backend restart.
- Removed the in-session `npm install onnxruntime-node` call after training. This prevents package-manager changes in the live application from triggering a development-server reload. Existing `start.bat` and `start.sh` install ONNX Runtime on the next launch once a trained cell model exists.
- Corrected Ultralytics validation metrics extraction: `trainer.metrics` may be a plain dictionary, not an object with `.results_dict`. The worker now handles dictionaries, metric objects, validator metrics, and the `model.train()` return value. Added tests for dictionary and metric-object forms.
- Dataset preparation now rebuilds only `training/cell_yolo/dataset`; prior YOLO run checkpoints and logs are retained.

## Automated validation

- Node automated suite: **24/24 passed** (`node --experimental-strip-types --test` across all seven included test files).
- Python worker: `python3 -m py_compile training/train_cell_yolo.py` passed.
- Python durable-state and metric-extraction smoke tests: **3/3 passed** (`python3 -m unittest training.test_job_state`).
- TypeScript/TSX transpilation checks: **5/5 passed** (`src/App.tsx`, `InspectionCanvas.tsx`, `cellBoxPlacement.ts`, `server/cellTraining.ts`, `server.ts`).
- `bash -n start.sh`: passed.
- Full `tsc --noEmit` could not complete because this source archive has no installed `node_modules` and the `@types/node` / `vite/client` type definitions are unavailable in the current environment.
- Full Vite production build and browser-level keyboard testing were not run.
- ZIP integrity is checked after final packaging.

## Not verified in this environment

- A live CPU YOLO training session, Windows detached-process behavior, ONNX export, and ONNX inference using a real trained cell model require the user's workstation and training dependencies.
- Full browser interaction and the complete Vite production build require installing the project dependencies. No live training accuracy claim is made.
