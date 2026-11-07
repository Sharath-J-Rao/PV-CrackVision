# PV CrackVision v1.4.4 validation

## Changes
- Deterministic randomized source-image split (hash order), 80/20 holdout with at least 2 validation images.
- Requires at least 10 unique source images, at least 10 train boxes and 5 validation boxes.
- Rejects non-finite, zero/negative, or out-of-image cell boxes before dataset generation.
- Writes `training/cell_yolo/dataset/dataset_audit.json` with split membership and box counts.
- Epoch table is reconciled from Ultralytics `results.csv` after training, even if callback metric APIs differ.
- Final activation gate runs a fresh validation pass on the exact `best.pt` checkpoint that will be exported, rather than trusting last-epoch metrics.
- Training UI supports 640, 768 and 960px; 960px uses batch 2.
- Retains the mAP50 >= 0.15 activation gate; rejected candidates do not replace the active model.

## Validation performed
- Python worker unit tests: **8/8 passed** (`python -m unittest training.test_job_state`).
- Python syntax: **passed** (`python -m py_compile training/train_cell_yolo.py`).
- Node regression suite: **24/24 passed** across seven existing test files.
- TypeScript/TSX syntax transpilation: **4/4 passed** for `src/App.tsx`, `server/cellTraining.ts`, `server.ts`, and `server/cellDetector.ts`.
- Full semantic TypeScript check / Vite production build: **not run successfully** because this packaging environment has no installed `node_modules`; global `tsc --noEmit` reports missing `@types/node` and `vite/client`.
- ZIP integrity: checked after packaging.

## Not claimed
- No live YOLO run, field accuracy test, Windows browser run, or ONNX inference was performed in this packaging environment.
