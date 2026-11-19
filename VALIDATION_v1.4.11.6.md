# PV CrackVision v1.4.11.2 — validation

Remediates every finding of the v1.4.10 read-only audit (see CHANGELOG.md).

Verified in the build environment: `tsc --noEmit` clean (strict), `vite build` OK, 123/123 Node tests (incl. `server/v1411.test.mjs`), 16/16 Python worker tests, release identity check, server boot with `HOST=0.0.0.0` (forced to loopback), `/api/data-folder/usage` async, `/api/app/info` = 1.4.11.

NOT verifiable on Linux (first Windows CI run must confirm): packaged app launch/smoke test, `app.relaunch()` after a data-folder change on an installed and a portable build, signing/icon.
Not changed by design: two-stage image upload (optional cache), batch concurrency limits.

## v1.4.11.2 addendum
Ground-gate fix (cvPipeline.ts `applyGroundGate`): detections inside reviewed/YOLO cells bypass the soil/PV colour mask. Re-verified: strict tsc clean, vite build OK, 126/126 Node tests (3 new in `server/v14112.test.mjs`). Not verified here: behaviour on your real images — re-run crack detection on the image that showed "15 detection(s) rejected" and confirm the count drops.

## v1.4.11.3 addendum
Inspection tab modes (App.tsx). Re-verified: strict tsc, vite build, Node + Python tests. The UI itself was not clicked through here; confirm on Windows.

## v1.4.11.4 addendum
Class weighting + cell-edit propagation. Re-verified: strict tsc, vite build, Node + Python tests (new `server/v14114.test.mjs`). UI flows not clicked through on a real image here; confirm on Windows.

## v1.4.11.5 addendum
Missed-cell feedback for YOLO. Re-verified: strict tsc, vite build, Node + Python tests (new `server/v14115.test.mjs`, including a real dataset build). No YOLO training run was executed here; check the dataset log line and recall on a held-out image after retraining.

## v1.4.11.6 addendum
Missed-crack feedback for the crack classifier. Re-verified: strict tsc, vite build, Node + Python tests (new `server/v14116.test.mjs`). No retraining on real data here.
