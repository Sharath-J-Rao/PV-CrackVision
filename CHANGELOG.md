## v1.4.17.8 - CPU speed gate for the anomaly crack model

## v1.4.17.7 - recover models from earlier versions, GPU crack detection/training, Crack Training Data tab

- **Fix: trained model from the previous version was not used.** The one-time legacy migration only scanned sibling releases inside the Working Folder, copied by "newer file wins", remembered files forever and could copy a pair's `.onnx` and `.json` from different releases. New `server/modelRecovery.ts` runs on every start (and on demand) over every known location (active models folder, Working Folder/parent/user `data/models`, each earlier release's `server/models` and `data/models`), validates complete pairs (`cellAnomaly`, `cellClassifier`, and `cell.onnx` only if missing) and installs the newest one when the active folder has none or an older one. Pairs are copied `.onnx` first, `.json` last; replaced files are kept as `*.before-recovery`; sources are never touched.
- **GPU.** Crack models are warmed up at start and the device in use (GPU provider or CPU) is reported in `/api/classifier/status`, with the reason when a GPU provider was skipped. The anomaly trainer now trains on CUDA when PyTorch sees a GPU (feature extraction, coreset, scoring), exports on CPU, and logs the device.
- **New tab "Crack Training Data"** (Advanced): counts of cracked/clean cells and images, breakdown by source, filterable thumbnail grid of every training cell with per-cell removal, active model info (version, trained at, validation, device) and a "Find models from previous versions" button.
- **Tests:** `server/v14177.test.mjs`.


## v1.4.17.6 - bundled anomaly detector: works out of the box, training optional
- **Bundled anomaly model.** The crack anomaly detector now resolves like the cell detector: a valid model you trained (any older version's `cellAnomaly.onnx` + `cellAnomaly.json` in the data folder) always wins, otherwise the verified pair in `resources/models` (manifest SHA-256, size, app-version check) is used, otherwise there is no model. A broken user pair is skipped instead of blocking a good bundled one.
- **UI.** Anomaly panel shows whether the active model is "bundled with the app" or "trained by you"; training becomes "Retrain on my data (overrides bundled)"; "Clear" becomes "Revert to bundled model" and only removes your own files.
- **Packaging.** `cellAnomaly.onnx/json` added to the model manifest script, electron-builder `extraResources` and the CI required-models check. Add the two files to `resources/models/`, run `npm run models:manifest`.
- **Unchanged:** detection, scoring, thresholds, training code and data formats.
- **Tests:** `server/v14176.test.mjs` (user-wins, bundled fallback, corrupt/unlisted/too-new rejection).

## v1.4.17.5 - minor release: single-source version (workflow, publisher and tests read package.json)

## v1.4.17.4 - hotfix: blank page and anomaly training crash (no detection, model, threshold or data-format change)
- **Fix: blank page.** The Google Fonts removal in v1.4.17.3 left a stray `<noscript>` in `index.html`; with scripting on, the browser treated the rest of the page, including the `/src/main.tsx` script, as text, so React never mounted. Tag removed; regression test in `server/commercialReadiness.test.mjs`.
- **Fix: "The training process exited unexpectedly (code 1)" on the anomaly detector.** The app-managed embedded Python (`python312._pth`) does not put the script's folder on `sys.path`, so `train_crack_anomaly.py` could not import `train_crack_classifier` and died before writing any state. The script now adds its own folder to `sys.path`. Regression test: `training/test_anomaly_embedded_import.py` (runs the trainer with `python -I`).
- **Setting: built-in classical ridge detector is now OFF by default.** `detectClassicalModuleSpiderWebs` only runs when "Built-in Classical Ridge Detector" is ticked in Detection settings → Crack search (config flag `enableClassicalRidge`, server accepts only a literal `true`). The learned crack engine is unaffected. Test: `server/classicalRidgeToggle.test.mjs`.
- **Diagnostics:** when a training worker dies before recording its own failure, the last lines of `worker.log` are now shown in the failure message and the job log.

## v1.4.17.3 - commercial-readiness audit (no detection, classification, training or data-format change)
- **Accuracy:** unchanged. No edits to the CV pipeline, cell detector, classifier, anomaly model, thresholds or training code paths.
- **Privacy/offline:** removed the Google Fonts requests (HTML, CSS and CSP). The app now makes zero third-party requests; system font stacks are used (Segoe UI / Cascadia Mono on Windows).
- **Support:** Electron now logs renderer crashes, hangs, uncaught exceptions and unhandled rejections to `desktop.log` and auto-recovers a crashed renderer (max 3 times; review state is persisted server-side). Added `SUPPORT.md`.
- **Reliability:** friendly message when the port is busy, graceful shutdown (SIGINT/SIGTERM) and crash logging for standalone runs; the desktop build keeps Electron's own lifecycle. Fixed a Python helper that wrote a stray `None.tmp` when no state file was configured.
- **Legal/compliance:** added `THIRD_PARTY_NOTICES.md` (generated, drift-checked in tests and CI), `license`/`description` in `package.json`, and flagged the Ultralytics AGPL-3.0 question for legal review in `SECURITY.md` (renamed from `SECURITY_v1.4.16.1.md`).
- **CI/supply chain:** production `npm audit`, CycloneDX SBOM, SHA-256 checksums, code-signing secrets wiring, concurrency cancel.
- **Size:** installer ships only the `en-US` Chromium locale pack.
- **Tests:** added `server/commercialReadiness.test.mjs` (privacy, metadata, notices, support files, lifecycle guards).

## v1.4.17.2 - efficiency audit (no schema, data-folder or model change)
- **Size:** removed 5 unreferenced sample images (~18 MB) from `public/assets/aistudio` that were shipped inside `dist` and the installer; removed the unused `motion` dependency (+3 transitive packages); removed stray `None.tmp` and committed `__pycache__` bytecode.
- **Perf:** training-data backup export/import no longer block the Node event loop (async file reads/writes/renames instead of sync calls on up to 600 MB bundles).
- **Startup:** Google Fonts stylesheet is now non-render-blocking and the two preconnects are gone, so the offline app no longer stalls first paint waiting for the network; Segoe UI added to the font fallback.
- **Housekeeping:** stale `v1.4.16.4` page title replaced with a version-free title.

