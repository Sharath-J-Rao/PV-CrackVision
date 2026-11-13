# PV CrackVision v1.4.5.9 validation notes

## Changes in this release

- Detections is split into two explicit workflows: Cell detections first; Crack detection second.
- Upload no longer auto-runs crack analysis. The user launches the cell-only pass across all queued images.
- Added `/api/cell-detect/upload`, which decodes/orients the uploaded source image, limits the maximum dimension to 8,000 px, executes the direct cell detector, and returns its own image dimensions, source, boxes, warning and diagnostics.
- The crack-analysis endpoint calls `analyzeInspectionImage(..., { runDirectCellDetection: false })`, so it does not invoke the direct YOLO cell detector again. Classical PV-region/cell-grid analysis remains part of the crack pipeline to locate crack findings.
- Crack batch execution is gated until the cell-detection pass has completed for every queued image.
- Cell review is bound to the exact frame ID. Saving maps editor coordinates back to the normalized full-resolution source image and replaces the previous complete annotation box set for that image.

## Checks run in this environment

- Node regression suite: **71 tests passed, 0 failed** (cell geometry, annotation hit testing, box placement, annotation replacement, model/training utilities, ONNX input dimensions, training split, shared data path migration, and settings).
- Python worker regression suite: **14 tests passed, 0 failed**.
- Project package metadata: `package.json`, lockfile root, and `metadata.json` all report version **1.4.5.9**.
- TypeScript/TSX transpile syntax check: **38 files transpiled; 0 syntax errors**. This is not a strict typecheck.

## Not verified here

- No Windows desktop/browser end-to-end test was run.
- No local ONNX Runtime inference was executed against the user's `cell.onnx` model in this environment.
- No production Vite build or strict TypeScript check was completed because this working tree does not currently have `node_modules` installed. The source-only Node test suite does not substitute for those checks.
- Commercial-grade no-error behavior on every field image cannot be guaranteed; images from domains not represented in the training data still require operator review and possibly additional annotations/retraining.
