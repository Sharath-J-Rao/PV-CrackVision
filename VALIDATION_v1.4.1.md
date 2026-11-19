# PV CrackVision v1.4.1 validation

## Implemented

- Direct, single-class YOLO11n PV-cell detection trained from individual cell bounding boxes. There is no trained module detector or module-outline annotation stage.
- In-app training runtime setup, epoch selection, progress reporting, logs, cancellation, ONNX export, activation, and previous-model rollback.
- Training/validation split by complete source image. Candidate activation requires validation metrics and mAP50 >= 0.15; failed candidates leave the active model unchanged.
- Direct YOLO cell boxes are merged into automatically localized PV surface regions as the cell grid used by downstream cell-level crack classification and cell mapping. The classical geometry remains the fallback when the YOLO model is absent or does not return enough assignable boxes.
- Separate cell-box annotation store for YOLO and reviewed positive/negative cell-crop feedback store for crack classification.
- Separate Training Centre, advanced-only Annotation workspace, and advanced-only Detection Diagnostics. Advanced mode is off by default.
- The normal Inspection workspace does not expose annotation or raw diagnostic controls.

## Automated checks

- Geometry tests cover direct cell grid construction, row/column assignment, per-region assignment, and the whole-image fallback path.
- Annotation editing now supports direct click-and-drag movement, Shift-click multi-selection, group movement, and click-to-place copy preview. Copied dimensions and group spacing are preserved; group movement and placement are clamped to image bounds.
- Existing test suite covers cell geometry, cell pitch, annotation hit-testing, crack-model feature vectors, and training validation gates.
- `npm test`: 23/23 passed, including four cell-box placement and group-move tests.
- TypeScript/TSX syntax transpilation: updated `InspectionCanvas.tsx` and `cellBoxPlacement.ts` passed (2/2).
- `python3 -m py_compile training/train_cell_yolo.py`: passed.
- ZIP integrity: checked after packaging.

## Not verified in this environment

- A live YOLO training run, PyTorch/Ultralytics bootstrap, ONNX export/inference against a real trained model, and full browser UI interaction require the user's workstation and internet-connected runtime.
- Full Vite production build and semantic TypeScript typecheck were not completed because dependencies and type definitions are not installed in this environment (`node_modules`, `@types/node`, and `vite/client` are absent).
- CPU-only YOLO training can be slow. The five-image minimum is a technical gate, not a dataset-size recommendation or an accuracy claim.
