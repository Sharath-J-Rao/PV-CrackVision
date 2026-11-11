# PV CrackVision v1.4.5.6 — validation record

## Changes under test

- YOLO early stopping is user-configurable and defaults to `patience=0`, which disables automatic early stopping so the worker continues to the requested epoch count. Positive patience explicitly opts into plateau-based stopping. No wall-clock time limit is passed. The existing Cancel action remains available; runtime/system failures can still terminate a process.
- Separate persisted configuration for YOLO cell detection and the logistic-regression crack classifier.
- Background-only negative-image records generate empty YOLO label files and are kept separate from positive cell annotations.
- Pattern-aware suppression prefers repeating individual-cell boxes over a larger enclosing overlap, followed by ordinary same-scale NMS.

## Checks performed in this environment

| Check | Result |
|---|---:|
| Node regression suite | 53/53 passed |
| Python worker/dependency unit suite | 14/14 passed |
| Python syntax compilation | Passed |
| TypeScript/TSX syntax transpilation | 34 files passed, no syntax diagnostics |
| Patience control tests | Passed (0 disables stopping; positive values are forwarded; values are bounded by requested epochs) |
| Training-kwargs mapping test | Passed (including epochs, patience, freeze, optimizer, image size and CPU mode) |
| Negative-sample storage regression | Passed |
| Oversized/repeating-cell overlap regression | Passed |
| Windows launcher health-check script included | Static review passed; Windows execution unavailable here |
| ZIP integrity | Passed (`unzip -tq`) |

## Not verified here

- Full semantic TypeScript type-check and Vite production build: `npm run lint` is blocked by missing `@types/node` and `vite/client`, and `npm run build` is blocked because `vite` is not installed. The TypeScript transpile pass checks syntax only; it is not a strict type-check.
- Live Windows YOLO training to the requested final epoch, real installed Ultralytics argument compatibility, ONNX inference on the user's trained model, and visual evaluation on the user's PV image set. These require the Windows runtime, installed packages and real dataset.

## Behavioural notes

The final validation gate is post-training and does not interrupt epochs. A candidate that fails the configured mAP50/precision/recall gate remains inactive; the previously active model is preserved. Background-only negative samples must contain no visible PV cells to detect. Do not mark an entire PV-module image negative when any cells remain unboxed.
