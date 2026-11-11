# v1.4.8.8 validation

## Changes
- Added a Detections-workspace mirror of the existing YOLO cell-training monitor. It reads the same `/api/cell-training/status` state and exposes job status/progress, dataset counts, epoch metrics, worker log, final validation summary, traceback, and existing CSV/full-log downloads.
- Added a selected-image cell-inference action. It reuses the existing `/api/cell-detect/upload` path and is mutually exclusive with the all-images cell-inference batch.
- Added a direct run action in the image viewer when the selected image has not yet been through cell inference.
- Updated the release version to 1.4.8.8.

## Automated checks
- Node regression suite: **86/87 passed**. The one failure is `server/classifierV2.test.mjs`, which could not resolve the existing `sharp` dependency because the uploaded source folder has no installed `node_modules`. The 82 tests outside that dependency-specific classifier file pass.
- `python -m unittest training.test_job_state training.test_install_package_dependencies`: **14/14 passed**.
- TypeScript parser: `src/App.tsx` and `src/components/CellTrainingMonitor.tsx` both parse without diagnostics.
- Isolated strict TypeScript check of `CellTrainingMonitor.tsx`: passed using temporary React/JSX declarations because project dependencies/types are not installed.
- Vite production build and full-project strict TypeScript check: not run; dependencies are not installed in the uploaded source folder.
- ZIP integrity: passed after packaging.

## Runtime validation boundary
The project was edited in a Linux container. The Windows desktop UI, actual ONNX inference, and an active local YOLO training process were not available for end-to-end verification in this environment.
