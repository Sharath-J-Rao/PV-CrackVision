# PV CrackVision v1.4.5.7 — release notes and validation scope

## Implemented

- Direct YOLO detections in Advanced → Annotations now have their own SVG hit targets. A click resolves to the exact matching detector box and stores that exact crop as a Positive (cracked) or Negative (clean) crack-classifier label. Sparse detections are retained as selectable targets even when there are fewer than four boxes or no classical module grouping is available.
- Detector hit targets take precedence over canvas pan and overlapping manually drawn YOLO boxes while selecting crack labels. The explicit Draw cell boxes and four-point calibration modes remain higher priority when active.
- Training Settings contains persisted detector hyperparameters, including epochs, patience, freeze, image size, batch, optimizer, learning rates, momentum, weight decay, warmup, augmentation, loss gains, multi-scale, seed, determinism, cache, pretrained weights, checkpoint period and activation mAP50 threshold.
- Patience defaults to 0 (automatic early stopping disabled); positive patience is an explicit opt-in. The activation gate is applied after training and does not interrupt epochs.
- The separate crack classifier exposes its actual logistic-regression controls; freeze layers are not applicable to this handcrafted-feature classifier.
- Background-only negative images create empty YOLO label files and are included in the detector dataset. Images containing any unannotated visible PV cells must not be treated as whole-image negatives.
- Oversized enclosing detections are suppressed only when at least two similarly sized, spatially distinct cell detections repeat inside them; ordinary same-scale NMS remains.
- Shared annotations, feedback, datasets, runs, logs, checkpoints, models and Python runtime remain under `Working Folder/data/` across releases. The ONNX input tensor follows the model's exported input dimensions.

## Previously completed v1.4.5.7 source checks

The staged source recorded 59/59 Node regression tests and 14/14 Python worker tests passing, Python syntax compilation passing, and TypeScript/TSX syntax transpilation across 34 files. The UI hit-target wiring was refined after those runs in response to the latest annotation-selection report; no additional test run was requested for this last UI-only adjustment.

## Not verified

- Strict semantic TypeScript checking and the Vite production build were blocked in the available environment by missing npm type/build dependencies (`@types/node`, `vite/client`, and Vite itself).
- The direct click workflow has not been manually exercised in the user's Windows browser, and detector quality has not been measured on the user's PV image set.
- Live Windows YOLO training, ONNX inference and end-to-end cracked/clean classifier training require the user's installed runtime and dataset.

This archive should be treated as a source update for Windows verification, not as proof that the user's specific detector produces accurate PV-cell boxes.
