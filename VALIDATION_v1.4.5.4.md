# PV CrackVision v1.4.5.4 — validation report

## Changes

- Fixed the runtime failure in `server/cellTraining.ts`: `createDataset()` previously returned an undeclared `validationBoxes` shorthand even though the accumulated counter was named `valBoxes`. The returned summary now explicitly maps `validationBoxes: valBoxes`.
- Fixed `server/crackModel.ts` referencing `CELL_MODEL_SCHEMA` without importing it locally. This made persisted crack-classifier model loading silently fall back to no model after a saved model file was read.
- Exported/imported the `CellAxisCandidate` type used in `server/cvPipeline.ts`, resolving another strict TypeScript diagnostic.
- Extracted the dataset summary shape into `server/cellTrainingDatasetSummary.ts` and added a regression test for the validation-box count and total consistency.
- Added explicit TypeScript annotations for canonicalized training records and feedback aggregation to avoid empty-array inference as `never[]` in strict mode.
- Retained v1.4.5.3 fixes for PEP 508 `extra` marker evaluation and for protecting only installed CPU `torch` and `torchvision` packages (not optional `torchaudio`).
- Added `npm run verify:release` to run Node regression tests, Python worker tests, strict TypeScript checking, and the production Vite build in an environment with dependencies installed.
- Corrected the launcher requirement to match Vite 8: Node 20.19+ or 22.12+. The previous major-only Node 20 check could accept unsupported releases.
- Replaced POSIX-only environment-variable assignment in `start` and `offline` npm scripts with a cross-platform Node launcher.
- Pip operations now ignore machine-level pip configuration as well as inherited index variables, preventing hidden `extra-index-url` or `find-links` settings from redirecting package resolution.

## Checks run in the packaging environment

- `npm test`: **30/30 passed** (includes dataset-summary and Node-version regression tests).
- `python -m unittest training.test_install_package_dependencies training.test_job_state`: **11/11 passed**.
- `python -m compileall -q training`: **passed**.
- `node --check scripts/node-runtime.mjs` and `node --check scripts/run-server.mjs`: **passed**.
- A broad source-level TypeScript diagnostic pass identified the undefined `validationBoxes` reference, the missing local `CELL_MODEL_SCHEMA` binding, the unexported `CellAxisCandidate` type, and the `never[]` inference in the YOLO source-record merge. These source issues were corrected; the post-fix unresolved-identifier diagnostic sweep is clear. Full dependency-backed `tsc --noEmit` could not be completed because npm dependencies were unavailable.
- TypeScript/TSX syntax transpilation: **29 source files passed**.
- ZIP integrity and version metadata consistency: checked during packaging.

## Not run / environment limitations

- `npm ci` did not complete in this environment (dependency installation timed out); therefore `npm run lint` and `npm run build` could not be completed here.
- No live Windows Python/PyTorch/Ultralytics installation or end-to-end YOLO training was run from this environment. Those remain target-machine validation items.
