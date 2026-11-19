# Validation — PV CrackVision v1.4.8.9

## Changes
- Crack classifier monitor moved to a full-width position immediately below the Training Centre heading.
- Live job status, current algorithm, active-model architecture, CNN epoch analytics, GBM tree progress, full worker log, error traceback, final held-out validation and CSV/log downloads are shown together.
- Per-epoch F1 was added to the CNN metrics, persisted job state, UI table and CSV export. Existing job-state files without F1 are loaded with F1 derived from precision and recall.
- GBM now yields after every completed tree in the UI training path, allowing the status endpoint and monitor to update during ensemble/GBM training.

## Automated checks performed
- Focused crack telemetry tests: **3/3 passed**.
- Python worker tests: **14/14 passed**.
- Node regression suite excluding the single classifier model test blocked by the absent `sharp` dependency: **85/85 passed**.
- Full Node regression command: **89 passed, 1 blocked** because `sharp` is not installed in this source-only environment (`ERR_MODULE_NOT_FOUND`).
- TypeScript/TSX syntax transpilation: **46 files, 0 syntax diagnostics**.
- Strict project type-check: blocked because `@types/node` and `vite/client` are not installed.
- Vite production build: not run because `node_modules` is absent.
- Windows browser/runtime validation: not performed in this environment.

No claim is made that Windows UI, production bundling or the complete strict type-check passed.
