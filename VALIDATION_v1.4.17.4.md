# PV CrackVision v1.4.17.4 — validation record

Anomaly-detection release (crack engine switched from the supervised classifier to a clean-cell anomaly detector; classifier archived). Run on Linux, Node 22, Python 3.13, with `npm install --ignore-scripts` (the `onnxruntime-node` download is blocked here, so Node-side ONNX inference was not exercised).

## Passed

- `npm test` — 189/189 passed (includes new `server/v14170.test.mjs`: engine default/override/legacy-file fallback, anomaly settings clamping, separate anomaly model files + meta validation, `anom` job mode). `server/v14162.test.mjs` was updated for the new per-engine "stage did NOT run" warning wording.
- `npx tsc --noEmit` (`npm run lint`) — no errors.
- `npx vite build` — built.
- Python worker suite — 43 run, all passed, 1 skipped (see below). New `training/test_crack_anomaly.py` covers greedy coreset, AUROC, both threshold modes, clean-only split fallback, CLI validation.

## Not verified here

- `training/test_crack_anomaly.py::Scorer` (torch scorer, ONNX export, ONNX-vs-torch parity, dynamic batch) was **skipped**: no working CPU torch could be installed in this sandbox. The trainer also re-checks parity at run time and refuses to activate on mismatch, but a first real training run is the real test. Run `npm run test:training-worker` on a machine with the training runtime.
- No end-to-end training or detection on real images; no model weights are bundled, so **train the anomaly detector before the next crack run**.
- Electron/Windows packaging not rebuilt.

## v1.4.17.4 review pass (expert panel: data linkage, inference/UI wiring, trainer numerics)

Passed on Linux, Node 22: `npm test` 190/190, `tsc --noEmit` clean, `vite build` OK, `check-release-version` OK, Python worker suite 40 run / all OK / 1 skipped (torch Scorer + ONNX parity, no torch in this sandbox).
Verified unchanged: data root, `cell-level-v1` feedback schema, cell_annotations, review_state; trainer fits only clean cells; crop geometry/preprocessing parity with the Node scorer; engine default and routes; UI panels mounted.
Fixed: legacy decisionThreshold on anomaly scale, threshold-on-validation-score, AUROC ties, EXIF 5-8 swap in /api/feedback/batch, auto-negative tagging + readiness warnings.
Still not verified here: a real anomaly training run, torch/ONNX parity, Node ONNX inference (`onnxruntime-node` download blocked), Windows packaging. Train the anomaly detector once and check the held-out report (AUROC, threshold mode) before trusting it in the field.
Known and left as is: Pillow vs sharp Lanczos differ slightly; 1px crop rounding difference on exact .5 cell sizes.
