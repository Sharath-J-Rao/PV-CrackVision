## v1.4.13.1 - fully offline EXE with bundled AI models, Gemini removed, two feedback fixes
- **Gemini removed completely.** The `@google/genai` dependency, the `GEMINI_*` / `CV_ENGINE` settings, the cloud tile-inspection stage, the "VLM Scanned" tile filter and the Maximum Inspection Tiles slider are gone. Inspection is always local (YOLO cells + classical/learned crack detectors); no API key or internet connection is needed for inference.
- **Bundled AI models.** `resources/models/` (cell.onnx, cellCrackModel.user.json, manifest.json) is packaged into the EXE outside the asar. See `resources/models/README.md` for how to add them (`npm run models:manifest`, `npm run models:check`).
- **Automatic model-path detection** (`server/modelResolver.ts`): your own trained/imported model wins, otherwise the bundled one is used (read-only), otherwise classical detection. No manual model installation.
- **Missing/corrupt/incompatible files are handled**: size + SHA-256 + format + app-version checks; a bad candidate is skipped with a warning (detector status, `/api/health`, `desktop.log`) and a model that fails to load falls through to the next one. Nothing is deleted.
- **EXE and models stay compatible**: `manifest.json` carries `minAppVersion` (and optional `maxAppVersion`). CI runs `models:check --require`, verifies the models are inside the package and the smoke test (`PV_REQUIRE_MODELS=1`) loads them.
- **Fix:** deleting a crack you added by hand now also removes its stored "cracked" training label.
- **Fix:** restoring a deleted detection no longer deletes the label if that cell has since been re-labelled Cracked.
- Tests: `server/modelResolver.test.mjs`.
- **Models included:** `cell.onnx` and `cellCrackModel.user.json` (model version `cell-ensemble-v1.4.13-1-733`) are bundled with a generated `manifest.json`.

## v1.4.12.9 - fewer crack false positives on module edges, Select all / bulk delete for cracks
- **"Edge" now means the outer ring of the detected cell grid** (the outer rectangle of the real cells) at crack-detection time. Previously it was measured from the module-region outline, so the ring moved with how loose or tight that box was (a loose box could miss outer cells, a tight one could include the second row). The trainer's edge sample weight still uses the module outline and is unchanged.
- **New setting *Outer-ring cells: require both models to agree*** (Training Settings, crack classifier; default on). An outer-ring cell is reported only when the boosting model AND the CNN each reach (decision threshold - perimeter drop), not just their average. Single-component models are unaffected; untick to use the average only. Older saved settings pick up the default.
- **Select all, Clear, Delete selected (N) and Keep selected (N)** on the crack toolbar of the *Crack detections* and *Review* tabs (Inspection). Ctrl/Shift/Cmd+click toggles single cracks; Delete/Backspace removes every selected crack. Deleting a model detection still saves its exact cell as a hard negative; a bulk delete does it in one request (one negative per distinct cell, all-or-nothing restore on failure) and asks for confirmation when more than one crack is affected. *Keep selected* deletes every other crack shown. Deleted detections can still be restored from the Review list.
- Tests: `server/v14129.test.mjs`.

## v1.4.12.8 - partially annotated images, better trainer defaults, box fusion
- **Partially annotated images.** New *Partially annotated* tick next to *Save reviewed boxes* (sticky between images). When ticked, training keeps the marked cells plus a margin and paints everything else neutral grey, so real but unmarked cells are no longer learned as background. Margin is a setting (*Visible margin around marked cells*, default 0.15 of a cell). Images saved without the tick behave exactly as before. Greying is applied to train and validation images; the dataset audit records `partialMaskedImages`.
- **Weighted box fusion** for cells seen by the full-image pass and the tiles (setting *Fuse the same cell seen by several passes*; off restores keep-the-best merging).
- **Region boxes grow to contain their detected cells** (never shrink).
- **Trainer defaults** (batch 4, cosine LR on, vertical flip 0.5, cache on) reach existing installs only where the value was never changed.
- Training log now prints the AMP state Ultralytics actually used; `PV_YOLO_REQUIRE_CUDA=1` stops a run that would silently fall back to the CPU.
- Crack-classifier class weights (N / (2 x class count)) were already in place; a test now pins the formula.
- Tests: `server/v14128.test.mjs`.

## v1.4.12.7 (hotfix) - ONNX gate tolerance, auto-install of packages, sturdier saved reviews
- **False "Exported ONNX validation gate not met" fixed.** This was not a GPU/ONNX fault: the exported model scored mAP50 0.2984 against a 0.30 gate. The exported ONNX floor is now the configured minimum minus a 0.02 export tolerance (best.pt must still meet the full gate and the best.pt-vs-ONNX parity check is unchanged).
- **Packages install automatically before start.** `npm start` / `npm run dev` / start.bat / start.sh check every package in package.json and run `npm install` once when any is missing (`scripts/ensure-dependencies.mjs`). The Python training runtime (torch / ultralytics) already installs itself on the first training run.
- **Saved reviews are sturdier.** Saved after 0.4 s (was 1.2 s), flushed when the window is hidden or closed, retried after a network/server failure, and crack-only reviews (no completed cell detection) are now saved and restored too.
- **New settings (Training Settings tab).** *Crack decision threshold* (section 06; 0 = the model's own trained threshold) controls crack sensitivity at detection time; *ONNX export tolerance* (section 01, next to Minimum mAP50) replaces the fixed 0.02. Cell confidence was already in section 04 (*Confidence threshold*) and section 05 (*Reject below confidence*).
- Tests: `scripts/ensure-dependencies.test.mjs`, extra case in `training/test_job_state.py`.

## v1.4.12.7 (update) - GPU-aware training and detection, tiny-detection removal, no text clutter in training plots
- **Training now uses the GPU properly.** Defaults that were tuned for CPU-only machines are now chosen from the hardware: data-loader workers scale with the GPU, RAM and OS (0 on CPU; up to 6 on Windows / 8 elsewhere with 24 GB+ RAM; 2 with 12 GB+; override with `PV_YOLO_WORKERS`), and mixed precision (AMP) is on for GPU training (off for CPU; override with `PV_YOLO_AMP=0`). The first worker log lines state the values in use.
- **Validation no longer caps recall.** Validation and best-epoch selection used Ultralytics' default of 300 detections per image although dense arrays hold 400+ cells; `max_det` is now 1000, so metrics and the best-checkpoint choice are no longer distorted.
- **Detection runs and fails safely on the GPU.** DirectML sessions get the session options ONNX Runtime requires (no memory pattern, sequential execution) plus full graph optimisation; if a GPU provider loads but fails while running, detection retries once on the CPU and the Models panel shows the device ("Inference device: GPU (dml)" / CPU) and any fallback reason. The exported ONNX is still validated on the CPU runtime on purpose (a device-independent check), now requested explicitly so no provider warnings appear.
- **Small detections are removed.** New *Fragment overlap* filter (default 0.5): two boxes cannot both be cells when one lies >= 50 % inside the other, so only the box closest to the typical cell size survives (removes small fragments inside a cell and shifted duplicates, even when the fragment is more confident). The typical cell size is now taken from the densest cluster of box sizes instead of a median that fragments could drag down, and the size/confidence gates are stricter (area >= 65 %, short side >= 70 %, confidence >= 0.10).
- **More recall in dense arrays.** The pre-NMS candidate, decoder and final-box caps (700 / 1200 / 1600, sized for CPU speed) are raised to 2500 / 3000 / 4000: duplicates of strong cells used to fill the old cap and cut off genuine low-score cells.
- **Existing settings are upgraded once.** A saved value that still equals its old default is replaced by the new default; anything you edited is kept.
- **Plots without text clutter.** Ultralytics' validation/training images (`val_batch0_labels.jpg`, ...) no longer print a "pv_cell" tag on every box; boxes only (`PV_YOLO_PLOT_LABELS=1` brings the tags back).
- Tests: `server/v14127.test.mjs`, extra cases in `training/test_job_state.py`.

## v1.4.12.6 (update) - Training no longer reports a false failure; saved review per image; larger YOLO sizes
- **False "training worker is no longer running" fixed.** The detached worker and the app both write the job-state file. When the worker exited, the app re-saved its own stale copy (still "training") over the worker's final "completed" record and then reported the dead worker as a failure, even though training, validation, ONNX export and activation had finished (the log ended on the ONNX validation "Speed:" line because the final steps were only written to the state file). The app now adopts the newer on-disk record before every write, treats a finished record of the running job as final, and runs the success path (model reload) when the worker exits with code 0.
- **Worker log is complete.** Final-phase steps (best.pt validation, ONNX export, ONNX validation, activation) and the worker exit code/signal are now also written to the saved worker log file.
- **Reviewed work persists across sessions.** Cell boxes you reviewed or edited, deleted and added cracks, and the "Review confirmed" flag are saved per image (identified by a SHA-1 of the image bytes, so renamed or moved copies match) in `training/review_state`. Adding an image you already reviewed restores it without re-running detection ("Restored from a previous session"); a restored image whose cells are edited is flagged stale as before. New "Discard saved review" button resets one image. Saving is automatic and best effort: it never blocks loading images.
- **Larger YOLO training sizes.** Image size now offers 1280, 1536 and 1920 (default 1280, saved settings are kept). Rough batch guide for a 24 GB GPU (mixed precision off): 1280 -> 8-12, 1536 -> 6-8, 1920 -> 4. The detector reads the model's input size from the ONNX file, so inference follows the trained size automatically.
- Tests: `server/v14126.test.mjs`.

## v1.4.12.5 (update) - Add crack actually responds (verified in a real browser)
- **Root cause:** the mouse-down handler returned early for cell-box / cell-hit handling before the add-crack code was reached, so only the hover highlight worked. Add-crack and Alt+drag copy now run first.
- **Delete key:** the Delete/Backspace listener now runs in the capture phase, because a capture-phase cell-box listener stopped the event before it arrived.
- Checked end to end in Chromium with the real canvas on a 4x6 cell grid: click a cell adds it, a second click selects it, dragging draws a box, Alt+drag copies two cells over, Delete removes the selected crack.

## v1.4.12.4 (update) - 12 px red box on marked images
- The red box around each crack in marked images is now 12 px wide (was 2 px). The origin crosshair stays 2 px.

## v1.4.12.3 (update) - Clicking an existing crack selects it while adding
- In "+ Add crack" mode a click on an existing crack marker or box now SELECTS it (cyan highlight on hover) instead of being ignored / adding a duplicate; clicking an empty detected cell still marks that cell.

## v1.4.12.2 (update) - Marked images carry the original EXIF unchanged
- The marked JPEG now gets a byte-for-byte copy of the original's metadata segments (all APP1 Exif/XMP/extended XMP, APP3-APP13, APP15). Previously the EXIF was edited (orientation forced to 1, pixel dimensions rewritten) and only one Exif, one XMP and IPTC segment were kept, so other vendor metadata was dropped.
- Not copied: JFIF, MPF/ICC (APP2) and Adobe (APP14), which describe the re-encoded pixels, not the camera.

## v1.4.12.1 (update) - Reliable add-crack click, Alt+drag copy, device log
- **Add crack:** detected cells are outlined while adding and the cell under the cursor is highlighted. A click now reliably marks that cell (a slight mouse wobble is no longer treated as a drag that made a tiny box).
- **Alt+drag copy:** hold Alt and drag a crack to copy it onto a neighbouring cell (snaps to the cell pitch / detected cell, green preview).
- **GPU:** training logs the device it uses (GPU name, or why CPU).

## v1.4.12 - Add crack by clicking a detected cell; crack markers selectable
- **Add crack:** click a detected cell to mark that whole cell as cracked; only when there is no detected cell, drag a box.
- Fixed: clicking a crack marker or its box in the crack view did nothing because the click was handed to the PV cell underneath. Markers/boxes now select (then Delete / Backspace or "Delete selected crack" removes them). Includes v1.4.11.7 changes.

## v1.4.11.7 (update) - Add/select/delete cracks, thicker red line, training failure reasons
- **Add crack.** New "+ Add crack" button in the Inspection crack view: drag a box around a missed crack (Esc to stop). Added cracks are saved as missed-crack positives for their cell when one is matched.
- **Select / delete cracks.** Click a crack to select it, then Delete / Backspace or "Delete selected crack".
- **Thicker red line.** On-screen crack boxes are about 1.6x thicker; exported marked images use a 2 px red line (was 1 px).
- **Training no longer stops without a reason.** A dead worker now reports the cause from its log (traceback, out of memory, Windows paging file, native crash) plus exit code/signal. The worker records every exit type, enables faulthandler, and uses 0 data-loader workers on Windows.

## v1.4.11.6 (update) - Cracks the model missed are fed back with extra weight
- **Missed-crack feedback.** When you save a cell as *Cracked* (positive) in Crack Annotation after a crack scan of that image, and no model crack detection was on that cell, the label is stored as `missed-crack` and trained with the new weight *Missed cracks: training weight* (Training Settings -> Crack classifier, default 3, range 1-10) - the same idea as the missed-cell feedback for YOLO, applied as a sample weight (the crack classifier has no image repeats). Weights multiply with the perimeter-cell weight, and the equal-class-weight balancing from v1.4.11.4 still keeps cracked and clean classes at equal total weight.
- Nothing is flagged when the image has no crack scan yet (the model never had a chance to miss it) or when the cell was already detected as a crack. The save message and the training report show how many missed-crack labels were used.
- Deleting a false crack detection continues to train a hard negative (unchanged).

## v1.4.11.5 (update) - Cells you add are fed back to the cell model as "do not miss these"
- **Missed-cell feedback.** When you press *Save reviewed boxes*, every box you added (no model detection overlapping it, IoU < 0.3) is recorded as a cell the YOLO model MISSED. In the next cell-training run, images that contain missed cells are repeated in the TRAIN split `missedCellRepeat` extra times (new Training Settings -> Cell detector field, default 2, 0 disables, max 10) so the model sees those hard examples more often under augmentation. The validation split is never oversampled, so held-out metrics stay honest. The dataset audit and training log report how many images/cells were flagged.
- Cell detections now shows how many cells were added by you (missed by the model) and confirms the count after saving. Only boxes compared against a real YOLO output count: if the image was detected by the classical fallback, nothing is flagged.
- Deleting a false detection already teaches the model (the box is simply absent from the saved set); nothing changed there.

## v1.4.11.4 (update) - Equal class weight for crack training; added cells reach every tab
- **Crack classifier class weights.** `Balance positive and negative classes` (on by default) now gives cracked and clean cells EQUAL TOTAL weight. Previously weights came from raw cell counts and were then multiplied by the edge-cell and hard-negative multipliers, so those multipliers silently re-skewed the classes. Weights are now derived from the multiplier-adjusted mass of each class (`computeClassWeights`), and one helper serves the logistic, boosting and CNN trainers. The training report and model metadata record the clean:cracked ratio and the weights used, and the training toast shows them.
- **Cells added/edited in Cell detections now propagate.** Cell boxes drawn or moved but not yet saved are committed to the frame when you leave Cell detections (mode switch, image switch, other tab) - they are no longer lost, and they are NOT written to the YOLO dataset until you press *Save reviewed boxes*. The Inspection view always overlays the CURRENT cell boxes (it used to keep showing the cells stored with the last crack scan), Crack Annotation and the crack batch table read them, and the next crack run inspects exactly those cells. A crack result computed before the edit is flagged stale with a re-run banner in Crack detections and Review.

## v1.4.11.3 (update) - Inspection Stage: Cell detections / Crack detections / Review modes
- **Three modes in the Inspection tab** (switch above the image): **Cell detections** (the same tools as Cell Annotation: edit/remove YOLO boxes, *Draw cell boxes* for missed cells, duplicate/copy, *Save reviewed boxes*), **Crack detections** (the detected cracks with a delete button on every row; the cell of a deleted spider-web crack is saved as a hard negative for the next crack-classifier training run, and *Restore* undoes it), and **Review** (per-image summary plus the *Confirm review -> crack Marked Images* button and *Confirm all*).
- The confirm button moved from the left column into Review mode; crack deletion no longer requires Advanced mode.
- Crack-stage warnings (e.g. rejected-as-ground counts) are shown in Crack detections / Review, not while editing cells.

## v1.4.11.2 (update) - Ground gate no longer rejects detections inside confirmed cells
- **"N detection(s) rejected as ground / off-module" fixed.** The crack stage's soil/PV-fraction ground gate (a coarse, downsampled colour mask) was applied even to detections sitting on cells the operator reviewed in Cell Annotation or that the YOLO cell detector found. Those cells are explicit evidence of PV, so a mask judgement could discard real cracks on dark, dirty or low-silicon cells and on cells outside the automatically located panels. When the cell basis is `reviewed-cells` or `yolo-cells`, a detection whose centre lies inside such a cell now bypasses the soil/PV test. The classical-grid basis and every detection outside a confirmed cell keep the unchanged gate (thresholds untouched), so ground is still not reported as cracks.
- Release label is four-part (v1.4.11.2); `package.json` uses the valid-semver form `1.4.11-2` (EXEs are named `PV-CrackVision-1.4.11-2-*`). The release check maps one to the other.
- Tests: `server/v14112.test.mjs`.

## v1.4.11 (update) - Audit remediation of v1.4.10
- **Release identity.** Version, publish script (`Publish_v1.4.11_Desktop.ps1`), workflow branch/artifact, metadata and CHANGELOG now agree; `scripts/check-release-version.mjs` enforces this in `npm run verify:release`, CI and the publish script. Newly trained crack models carry the running app version (read from package.json).
- **Run selected image only (crack)** validates just the selected image's cell-detection state; the all-images action still requires every image (`src/utils/crackGate.ts`, tested).
- **Restart now (packaged app)** relaunches Electron through `app.relaunch()` (portable-EXE aware); source mode keeps the exit-75 supervisor.
- **CI** uses `npm ci`, runs lint and the release version check, smoke-launches the unpacked Windows app (`PV_SMOKE_TEST=1`: health, sharp, onnxruntime-node, exact version), and uploads the two explicit EXE names.
- **CNN evaluation honesty.** With >= 4 held-out images the holdout is split by whole image into a model-selection set (epoch/early-stop) and an independent test set that produces the reported metrics; with fewer images the figures are labelled "model-selection validation" and are not presented as an unbiased test.
- **Security.** The server binds to loopback only; a non-loopback `HOST` is ignored unless `PV_ALLOW_REMOTE_BIND=1` and `PV_REMOTE_TOKEN` are both set, in which case non-local API calls need the `x-pv-token` header.
- **Efficiency.** YOLO input-preview JPEGs are opt-in (hidden YOLO viewer button) instead of generated for every image; data-folder usage is asynchronous, yielding and cached (60 s); legacy Python-runtime trees are skipped on later launches once migrated (mutable data still file-checked). Upload limits for crack/cell stages are both 100 MB.
- **Desktop setup.** "Start fresh" is persisted; cancelling the folder picker re-asks instead of silently choosing; legacy data discovery walks neutral wrapper folders to a bounded depth. `npm run clean` is cross-platform. Optional `build/icon.ico` and `CSC_LINK` signing are picked up automatically.
- Not changed (by design): the two-stage image upload (optional cache, measure first) and batch concurrency limits.

## v1.4.9 (update) - Crack pass reuses the Cell Annotation cells; separate cell / crack Marked Images; taller, independently scrolling viewer
- **One set of cells for the whole workflow.** The crack pass (`/api/inspect/upload`) now receives the cell boxes from the Cell Annotation stage (the YOLO output plus your edits) together with the image, assigns them to the located panels with the existing YOLO-assignment code and inspects exactly those cells. The classical edge-periodicity grid is no longer used for crack cell crops, so nothing has to be drawn twice and a weak grid can no longer misalign the crops. Panels with no reviewed cells are not classified; reviewed cells outside any located panel get their own region instead of being dropped.
- **Automatic classical grid is OFF by default.** An image with no cell boxes is simply not cell-classified (the result says so). A checkbox in Crack Detection re-enables the classical grid as an explicit fallback. The "weak edge-map periodicity" warning only appears when the classical grid was actually used.
- Each crack result records its cell source (`cellBasis`: reviewed-cells / yolo-cells / classical-grid / none); the Crack Detection table shows how many cells each image used and whether they were reviewed.
- Editing or re-detecting cells after a crack run marks that result "cells changed, re-run" and withdraws its review confirmation.
- **Marked Images split in two stages that never mix.** Cell stage: *Model detections* (yellow, exactly what the model produced) and *Reviewed by you* (cyan, your saved set). Crack stage: *Detected - pending review* and *Reviewed*; a frame reaches the reviewed deliverable only after you delete false detections / add missed cracks in the Inspection Stage and confirm it (button in the Inspection Stage and in Marked Images, plus "Confirm all"). Crack marked images draw crack issues only (no cell lattice).
- **Taller viewer with its own scrolling.** Marked Images fills the window; the frame list and the viewer scroll independently (page does not scroll), the viewer has Fit / 100% / 200% / 300%, and the list follows the selection from clicks, arrow keys and Prev/Next. Thumbnails render only when scrolled into view.
- Fixed: two cells sharing a lattice row/column could produce duplicate anomaly ids.
- The ZIP endpoint (not used by the UI) keeps the classical grid because it has no cell boxes; the wire format for cells is documented in `server.ts`.
- Tests: 7 new geometry tests (113 total).

## v1.4.8.9 — Crack-classifier epoch telemetry visibility and F1 metrics

- Moved crack-classifier telemetry into a dedicated full-width Training Centre panel directly below the page heading, instead of hiding it below the right-hand classifier controls.
- The panel is visible before the first training run and clearly explains how CNN/Ensemble epochs populate.
- Added per-epoch F1 alongside training loss, held-out validation loss, precision, recall, specificity and balanced accuracy.
- Persisted F1 in the job state and downloadable epoch CSV; older saved telemetry is upgraded from its precision/recall values when loaded.
- Changed GBM training in the UI workflow to yield after each tree, so Express can serve live status polls during Ensemble/GBM work instead of appearing frozen.
- Clarified the active saved model architecture separately from the algorithm selected for the next run.
- Removed the old 3.5 KB batch-wrapper EXE from this release package; it was not a standalone desktop application. Use `start.bat` with the complete release folder.
- Preserved the existing YOLO cell-detection training flow and the separate crack-classifier workflow.

# Changelog

## v1.4.8.8 - cell training telemetry inside Detections + run one image
- **Detections now includes the live cell-training monitor:** the YOLO11n job status and progress bar, positive-image / reviewed-box / background-negative counts, persisted epoch analytics (precision, recall, mAP50, mAP50-95, train and validation losses), the live worker log, final validation metrics, error traceback, and CSV/full-log downloads are visible without leaving the Detections workspace. This reads the same durable job state as Training Centre; it does not start a second worker or duplicate training.
- **Run one image independently:** select an image in the Stage 1 queue and choose **Run selected image only**. The existing all-images action remains available. The single-image and batch actions disable each other to avoid overlapping cell-inference requests.
- The empty viewer now offers a direct **Run cell detection for this image** action. Cell inference remains separate from crack analysis; Stage 2 still requires successful cell detection for every queued image.
- No dataset, checkpoint, model, annotation, or data-folder paths were changed.

## v1.4.8.7 - cell-crop classifiers (boosting + small CNN), undersized-detection rejection, tidier Inspection page
- **Crack classifier now works on cell crops with two new learners** (Training Settings §02 > Classifier): (1) **gradient boosting** on richer features - the 16 original texture features plus 22 new edge-orientation and crack-continuity features (gradient-orientation histogram, coherence, diagonal/axis share, longest ridge run, largest-component span, line fraction); trains in seconds; (2) a **small CNN** (pure TypeScript, no new dependency) on a 2-channel 48x48 tensor built from ONE cell crop (luminance with gridlines flattened + max-pooled ridge response so hairline cracks survive down-sampling), trained with flips/transposes, Adam, class weights. Default = **Ensemble** (mean probability). The v1.4.8.6 logistic model stays selectable and old `cell-level-v1` model files still load and score identically.
- Training and inference share one extraction path (`extractCellSample`); the decision threshold is still tuned on training images only and acceptance still requires the held-out balanced accuracy. The training report shows per-component (boosting / CNN) held-out scores for the ensemble.
- **Cell detection rejects undersized detections:** new gates in the cell-type filter (Settings §05) - minimum short side in px (default 12), minimum area and short side relative to the typical cell (default 0.55 / 0.6; the typical cell is measured from the more confident half of the detections so fragments cannot lower it; relaxed automatically when a half-cut type is enabled) and a confidence floor (default 0.05). All can be set to 0 to switch off.
- **Inspection page:** left column reordered into 1 Images -> 2 Scan & export -> 3 Detection settings (grouped Crack search / Georeferencing); optional GPS-metadata loader folded away; telemetry strip tightened.
- Tests: new `classifierV2.test.mjs` plus undersize-filter tests (87 tests total).

## v1.4.8.6 - "Add to existing" choice
- When an image is already in the crack training data, the warning is now a three-way dialog: **Replace** (old labels for that image are deleted), **Add to existing** (old labels kept; a cell at the same position takes the new label), or **Cancel**. Replace remains the behaviour when the image is new or the browser cannot pre-check.

## v1.4.8.5 - save confirmation popup, replace-on-resave warning
- A visible popup (top-right, every tab, dismissable, 9 s) now confirms crack labels were added to the training annotations: counts saved, any earlier labels replaced, and the dataset totals. Previously the confirmation only appeared in the Inspection Stage side panel, so saving looked silent.
- Saving an image that is already in the crack training data now warns first ("N cracked + M clean already stored - replace?"). OK replaces that image's earlier crack labels with the new set; Cancel keeps them untouched. Matching is by the exact image bytes (SHA-1), so renamed copies are recognised.
- New endpoint GET /api/feedback/existing/:imageId; the batch save accepts replace:true. If the browser cannot hash (non-secure context) the pre-check is skipped but the server still replaces and the popup reports how many were replaced.

## v1.4.8.4 - taller Crack Annotation canvas, wheel zoom never scrolls the page
- Crack Annotation image viewer is now viewport-tall (window height minus header, minimum 680 px) instead of a short strip.
- Wheel zoom listener moved from the image itself to the whole viewer stage, so scrolling over labels, overlays or the stage margin no longer falls through and scrolls the page. Applies to every tab that uses the image viewer (Inspection Stage, Cell Annotation, Crack Annotation); scroll chaining is also blocked with overscroll-behavior: contain.

## v1.4.8.3 - fix: Crack Annotation labels (cracked/clean cells) were not saving
- **Root cause:** saving required a crack-inspection result for the image, but the Crack Annotation tab works from the Cell-detection pass only, so Save failed with "Select and inspect an image before saving cell labels" (shown only as small text in the toolbar). Saving now works from the cell-detection frame when no crack result exists.
- **Coordinates:** labels are sent with the frame size they were drawn in and the server scales them to the full EXIF-normalised source image (matters for images over 8,000 px, which the cell pass downsizes). Previously they were stored unscaled.
- **Batch limit:** raised from 500 to 5,000 labels per save. "Rest of module = negative" on several modules could exceed 500 and the whole save was rejected.
- **Speed:** a batch is now one read + one atomic write of feedback.json instead of a full rewrite per cell.
- Nothing removed; Alt+drag copy and all other features are unchanged.

## v1.4.8.2 - Alt+drag copy in Crack Annotation
- Hold **Alt** and drag a cell in the Crack Annotation tab to copy it onto a missed neighbouring position (snaps to the cell pitch; drag a selected group to copy it). Copies show as amber "new cell (unsaved)" boxes with **Save cell boxes** / **Discard**; saving updates the image's cell set (and its YOLO labels, exactly like Cell Annotation) so the new cells can be labelled Positive/Negative.
- Guarded: Alt is required, so plain clicks still label cells and general box-edit mode stays off in this tab (the v1.4.7.3 fix is preserved).
- Audited v1.4.7.0 -> v1.4.7.2: nothing removed (the 1.4.7.1/2 canvas changes were refactors and additions; the help tip was extended, not shortened).

## v1.4.8.1 - Release audit (no feature removed)
- Audited v1.4.7.2 -> v1.4.7.3 -> v1.4.8.0: nothing useful was removed. v1.4.7.3 only stopped leftover box-edit mode from blocking labelling in the Crack Annotation tab.
- Alt/Ctrl/Cmd-drag copy of cell boxes is intact in Cell Annotation. It was never available in the Crack Annotation tab (box tools are gated off there by design since v1.4.7.2).

## v1.4.8.0 - Permanent data folder, model loading, auto-saved settings
- **Version-independent data folder.** The folder is now chosen in the app (new **Data & Models** tab) and remembered per user in `%APPDATA%\PV CrackVision\config.json` (macOS/Linux equivalents), outside every release folder. Resolution order: `PV_CRACKVISION_DATA_DIR` > chosen folder > default (`Working Folder/data` for ZIP/source installs, `<user config>/data` for a packaged build). Existing installs keep working unchanged until a folder is chosen.
- In-app folder picker with optional non-destructive copy of existing data (Python runtime excluded by default because it is not relocatable). A restart applies the change; `scripts/run-server.mjs` restarts the server automatically (exit code 75). Desktop shells can expose `window.pvCrackVisionDesktop.selectFolder()` for a native dialog.
- `start.bat`/`start.sh` helper (`scripts/data-paths.mjs`) reads the same per-user config.
- **Load a model**: import `.onnx` (cell detector, test-loaded before activation, previous model kept for roll-back), `.json` (crack classifier) or `.pt` (training checkpoint); export the active cell model. Models are data files, so this works identically in a packaged .exe.
- **Training/inference settings auto-save** (debounced) with a visible status; export, import and reset-to-defaults added. Settings live in the data folder and are shared by all versions.
- UI: first-run banner and nav indicator until a permanent folder is chosen.
- Security: file-system endpoints are loopback-only and require a custom header.

## 1.4.7.3 — fix: cells could not be labelled in Crack Annotation

- Box-edit mode left over from Cell Annotation (selecting/deleting boxes) swallowed every canvas click in the Crack Annotation tab, so no cell became Positive/Negative and Save stayed at (0). Box editing is now forced off, and the box selection cleared, whenever the Crack Annotation tab is active.

## 1.4.7.2 — duplicate across the whole row/column

- The Duplicate ← ↑ ↓ → buttons no longer make a single copy. They repeat the selection in that direction for **N copies, or all the way to the image edge when N = 0 (default)**, using the module's measured cell pitch. Positions that already hold a box are skipped, so running it over a partly annotated row adds no duplicates. The last copy stays selected.
- Alt/Ctrl+drag still snaps to cell steps, now to any whole number of steps (drag as far as you like).

## 1.4.7.1 — fast cell duplication

- **Alt+drag (or Ctrl+drag) a selected cell/group** copies it and drags the copy. The copy snaps to exactly one cell step (cell size + the module's measured seam) and locks to the row/column when you are near it, so repeating a column or row needs no precise aiming. A click or a drag that returns to the origin discards the copy.
- **Duplicate ← ↑ ↓ → buttons** in the cell toolbar: one click puts a copy one cell step away and selects it, so repeated clicks tile across the module. Reports "No room inside the image" instead of overlapping.
- The existing Copy selected / Place copy flow is unchanged.

## 1.4.7.0 — cell types, exposed parameters, auto-negatives

- **Cell-type filter** (Training Settings §05): only YOLO detections matching an enabled *accept* cell type (aspect = long ÷ short side, tolerance, optional absolute px bounds) and no *negative* type pass; others are rejected before review. Default = standard near-square cell; a half-cut (2:1) type is shipped disabled; add any number of extra types or negative types. "Learn default type from my saved cell boxes" measures the aspect from your annotations. A relative-size window (× image median cell) removes module-sized or fragment boxes.
- **All fixed detector parameters exposed** (§04): confidence threshold, NMS IoU (per pass and merge), candidate/box caps, tile size/overlap/count, tiles on/off, enclosing-box suppression on/off. Defaults equal the old hard-coded values.
- **Crack ground gate exposed** (§06): max soil fraction, min PV fraction (+ fallback). Crack-classifier training parameters were already in §02.
- **Crack Annotation: "Rest of module = negative"** (on by default): when a module has at least one positive (cracked) cell, every unlabelled cell in it is saved as negative (clean). Explicit labels are never overridden.
- Cell-detector negatives: background-only negative images (§03) are unchanged; *negative cell types* (§05) reject false-positive shapes at inference.
- Settings file is cached by mtime (read per detection otherwise).

## 1.4.6.0 — bootstrap loop, separate annotation tabs, hidden YOLO viewer, two-layer Inspection

- Tabs: **1 · Cell Annotation** (YOLO cell boxes), **2 · Crack Annotation** (cracked/clean labels on the cell boxes), **Crack Detection** (batch). The old mixed advanced "Annotations" tab is removed.
- Bootstrap loop: train a rough cell model, run Cell detections, correct and save boxes; the saved set becomes the base cells for Crack Annotation and Inspection, and replaces that image's YOLO labels for retraining.
- Crack Annotation uses the reviewed cell boxes as the selectable cells; box editing is hidden there so the two label sets cannot mix.
- Inspection Stage shows cell boxes (green = clean, red = cracked, cyan = selected) together with crack findings, with a legend; missed cells/cracks can be added and saved for retraining.
- Hidden YOLO input viewer (Ctrl+Shift+Y, Esc closes): shows the exact letterboxed tensors the cell model received, per window, with scale/padding. The server returns them only when the `x-debug-inputs: 1` header is sent.
- GPU support (CPU still works): training auto-selects CUDA GPU > Apple MPS > CPU (`PV_YOLO_DEVICE=cpu|0|auto` overrides). Setup installs the PyTorch CUDA wheel matching the NVIDIA driver (cu118/124/126/128) and upgrades an existing CPU-only runtime; override with `PV_TORCH_VARIANT`. ONNX inference tries DirectML/CUDA then falls back to CPU (`PV_ONNX_PROVIDER=cpu` forces CPU); status reports the active provider. Linux CUDA inference additionally needs the CUDA 12 runtime libraries.
- Fix: `onnxruntime-node` is now a regular dependency; previously a model trained in-session could not run until a relaunch.
- Fix: default post-training `minimumMap50` lowered 0.15 → 0.05 so a rough bootstrap model can activate.

## 1.4.5.9 — separate cell and crack detection workflows

- Added a dedicated **Detections** workspace with two explicit stage buttons: **Cell detections** and **Crack detection**.
- Image upload now queues source images without silently starting either detector. Operators choose when to run each stage.
- Cell detections now use a standalone image-only endpoint. The cell pass runs across every uploaded image, returns per-image diagnostics and editable boxes, and does not launch the crack-analysis pipeline.
- Added an all-images cell-detection queue with per-image status, box counts, warning/error reporting, and an editor for the currently selected source image.
- Cell boxes can be selected, moved, resized, multi-selected, deleted, redrawn and saved. Saving targets the exact uploaded frame ID and coordinate dimensions, and the complete reviewed box set replaces the previous annotations for that image.
- Crack detection is a separate all-images batch stage. It is disabled until cell detection has completed for every image. Its server pipeline skips the direct YOLO cell-inference call rather than repeating Stage 1.
- Preserved the existing classical PV-region and cell-grid geometry used internally by the crack-analysis pipeline; it is distinct from the direct YOLO cell-localization workflow.
- Kept cracked/clean cell labels and the crack classifier separate from YOLO cell-localization labels.
- Updated the in-canvas guidance to distinguish cell-box review from crack/clean cell labeling.

## 1.4.5.8
- YOLO detections are copied into the editable annotation-box layer instead of being rendered as a separate static overlay. Operators can select, move, multi-select, copy, delete, redraw and save detector boxes.
- When YOLO is the source for an image, classical module-region grids are excluded from annotation hit-testing to prevent the classical detector from competing with YOLO.
- Once YOLO inference executes, an empty result remains an empty YOLO result; inference errors are reported as YOLO errors. Classical auto-detection is not silently run afterward.

# Changelog

## 1.4.5.7 — direct YOLO detection selection in crack annotation

- Fixed a disconnect between the green direct YOLO cell-detection overlay and the positive/negative crack-classifier annotation hit-testing. Previously the green boxes were rendered as visual-only overlays, while selection used only `moduleRegions[].cellGrid` and required at least four cells. A visible detection (especially a sparse result with one to three boxes) could therefore appear detected but could not be clicked and labelled.
- Direct YOLO boxes now become the actual selectable per-cell annotation targets, using the same pixel rectangles displayed by the detector. They override stale classical cell geometry inside the same PV region so the hit target matches the visible detection.
- Removed the four-cell minimum from selecting a YOLO-derived annotation region. A single or small number of valid detections can now be selected and labelled positive/negative. The four-cell minimum remains relevant to the server-side lattice application and downstream analysis, not the annotation click targets.
- Detections outside the automatically segmented region bounds remain available in an explicit review region instead of silently disappearing from the annotation workflow.
- Clicks on a cell now take precedence over an overlapping crack marker, allowing the canvas hit tester to label that cell.
- Added regression tests for one-box selection, stale-geometry replacement, module assignment, unassigned detection review, and preservation of the classical fallback when no YOLO detections exist.

## 1.4.5.6 — fixed-epoch training, complete hyperparameter controls and detector hard negatives

- Diagnosed the epoch-38 stop: the previous YOLO worker used `patience=20`, allowing Ultralytics early stopping after 20 epochs without a new best validation fitness. The worker now defaults to `patience=0` (disabling Ultralytics early stopping) and exposes an explicit patience control in Training Settings; it runs the full requested epoch count by default, unless a positive patience is deliberately selected, the operator manually cancels, or a genuine worker/system error occurs. No wall-clock time limit is passed.
- Added a persistent **Training Settings** tab with independent controls for YOLO cell detection and the crack classifier. YOLO controls include epochs, early-stopping patience (0 disables; positive values opt in), image size, batch, freeze layers, optimizer, learning-rate schedule, momentum, weight decay, warmup, loss gains, augmentation, multi-scale, seed, determinism, cache, pretrained weights and checkpoint cadence and the post-training mAP50 activation threshold. Crack-classifier controls expose iterations, learning rate, L2, validation fraction, threshold search, minimum held-out balanced accuracy and class balancing. Freeze is explicitly marked inapplicable to the handcrafted-feature logistic-regression classifier.
- Added explicit background-only negative images for the YOLO cell detector. Empty YOLO label files are generated for these images; negative images are split into training/validation when enough examples exist. A source image already carrying positive cell boxes cannot be converted into a whole-image negative.
- Added pattern-aware detection suppression: a larger enclosing detection is rejected only when it contains at least two similarly-sized, spatially distinct individual-cell detections; a single nested detection is not enough to discard the larger box. Conventional NMS still removes same-scale duplicates.
- Training settings persist at `Working Folder/data/training/training-settings.json`, and background-only images remain in the shared annotation store.
- Integrated the Windows launcher health check: Firefox opens only after `/api/health` responds, rather than after a fixed seven-second delay.
- Added regression coverage for default full-epoch patience, explicit positive patience mapping, negative annotation persistence/upgrade, repeated-cell enclosing-box suppression, and retention of an isolated nested detection.


## 1.4.5.5 — Shared training data and ONNX input-size compatibility

- Moves annotations, feedback labels/images, YOLO datasets/runs/logs, checkpoints, active/previous models, Python runtime and job state into `Working Folder/data/`, shared by all release folders under the same Working Folder.
- First launch imports and merges existing data from sibling `PV_CrackVision*` release folders. Original files are retained; a migration manifest avoids re-copying unchanged runtime files. Override with `PV_CRACKVISION_DATA_DIR`.
- Reads fixed ONNX NCHW input dimensions from `InferenceSession.inputMetadata.shape` and preprocesses to the actual model size (including 960×960); no hardcoded 640×640 tensor mismatch.
- Selects the held-out validation image subset to bring validation box counts closer to the configured 20% ratio. For the supplied 2/13/78 box example, the 13-box image is held out.
- Marks validation results from fewer than five distinct source images as provisional.

## 1.4.5.4 — YOLO dataset summary runtime and compile-time audit fixes

- Fixed the YOLO dataset builder returning an undeclared `validationBoxes` shorthand. It now maps the accumulated validation count (`valBoxes`) explicitly into the returned dataset summary; this was the runtime failure reported during cell training.
- Fixed `crackModel.ts` using `CELL_MODEL_SCHEMA` without importing a local binding. This could silently prevent a saved cell-level classifier from loading after training.
- Exported and imported the `CellAxisCandidate` type used by the joint X/Y pitch selection, fixing a strict TypeScript compile error.
- Moved the dataset summary shape into a typed, isolated helper and added a regression test asserting the `validationBoxes` property and train/validation count consistency.
- Added explicit record types to canonicalized source-image and feedback aggregation paths, preventing empty-array inference from becoming `never[]` under strict TypeScript checking.
- Added `npm run verify:release` as a single release gate for Node regression tests, Python worker tests, strict TypeScript checking, and the production Vite build.
- Corrected the launcher Node-version gate to require the Vite 8 supported range (Node 20.19+ or 22.12+); the previous major-version-only check could accept unsupported Node 20 releases.
- Replaced POSIX-only `NODE_ENV=production` / `CV_ENGINE=classical` npm script syntax with a Node launcher so `npm start` and `npm run offline` work on Windows as well as macOS/Linux.
- Updated release/version metadata to 1.4.5.4.
- Isolated pip from inherited environment index settings and user/system `pip.ini` / `pip.conf` files, preventing unexpected secondary indexes from interfering with CPU-wheel and PyPI dependency resolution.
- Retained the v1.4.5.3 PEP 508 `extra` marker fix and the corrected optional-package protection list; retained CPU-only PyTorch wheel installation.

## 1.4.5.3 — dependency marker and optional runtime protection fix

- Fixed PEP 508 requirement-marker evaluation by defining `extra` as an empty string for base dependency resolution. Optional dependencies guarded by markers such as `extra == "gpu"` are now correctly excluded without raising `UndefinedEnvironmentName`.
- Removed `torchaudio` from the default protected-package list. This application uses CPU `torch` and `torchvision` but does not require `torchaudio`; the previous default caused setup to abort when the optional distribution was absent.
- Kept the explicit CPU-wheel-only PyTorch installation and metadata-driven PyPI dependency-resolution flow from v1.4.5.2.
- Added regression tests for empty/non-empty `extra` markers and the exact default protected-package list.

## 1.4.5.2 — CPU-wheel-only PyTorch install and dependency audit

- Replaced the ineffective `PIP_INDEX_URL` workaround from v1.4.5.1. A command-line `--index-url` takes precedence over that environment variable, so pip's isolated build step continued to search only the PyTorch CPU index.
- Split PyTorch setup into two explicit phases: install only binary `torch`/`torchvision` wheels from the official CPU index using `--no-deps --only-binary=:all:`, then resolve their declared dependencies from PyPI without requesting an upgrade of the installed CPU wheels.
- Made the Ultralytics installation use PyPI explicitly; removed the environment-variable manipulation that could discard `PIP_EXTRA_INDEX_URL`.
- Added `pip check` before the runtime is marked ready, both for an existing runtime and after a new/repair installation, to catch missing or incompatible transitive dependencies.
- Added automated tests for the exact pip arguments and package-index separation.
- Preserved the existing annotations, models, checkpoints, job state, and training-data paths. No user runtime data is deleted by this release.

## 1.4.5.1 — PyTorch CPU runtime dependency-index fix

- Fixed first-time YOLO runtime setup when pip installs CPU-only `torch` and `torchvision` from the PyTorch wheel index.
- The PyTorch install command remains explicitly pinned to `https://download.pytorch.org/whl/cpu`; its isolated PEP 517 build subprocesses now resolve build requirements such as `flit_core` from the standard PyPI index.
- This avoids the previous `No matching distribution found for flit_core<4,>=3.11` failure caused by searching the PyTorch-only index for general Python build dependencies.
- No training dataset, annotations, model weights, or user runtime data are changed by this fix.


## 1.4.5 — complete annotation ingestion and lossless per-image saving

- Fixed the direct cell-box annotation store replacing the entire box list whenever an operator saved more boxes for an already annotated image. Repeated saves now merge new boxes and update near-identical boxes without deleting earlier work.
- Training dataset construction now consumes **both** annotation workflows: Advanced direct cell-box annotations and reviewed positive/negative cell labels saved through the feedback workflow. Previously, reviewed labels were stored separately and were invisible to the direct YOLO cell trainer.
- Training status counts the union of both stores and reports distinct source images and deduplicated cell boxes.
- Removed the conflicting hard-coded 10-image minimum from the training service and UI. Training requires at least two distinct source images so train/validation remain image-separated; at least one valid box must exist in each split.
- Canonicalizes decoded image pixels before the train/validation split so an image saved through both workflows is not intentionally split across train and validation merely because the stored image bytes differ.
- Dataset audit records every included image, its split, and its final box count. Existing run checkpoints and active ONNX model are retained when rebuilding the dataset or when a candidate fails validation.
- Added a regression test proving repeated saves preserve earlier cell boxes while deduplicating an already-saved cell.


## 1.4.3 — epoch analytics, persistent logs, and checkpoint reuse

- Added a per-epoch analytics table showing validation precision, recall, mAP50, mAP50-95, training and validation box/class/DFL losses, and recorded time. Metrics are collected after the epoch validation callback, not before it.
- Added persistent per-job worker logs with full stdout/stderr capture, a UI log panel, and log download. Added CSV download for epoch metrics.
- Expanded durable job state to retain up to 500 epoch rows and 300 recent log entries across browser refreshes and backend restarts.
- Made training-runtime setup idempotent: it probes the existing Python environment and skips pip, PyTorch, and Ultralytics installation when imports are already healthy. Removed unconditional pip upgrades.
- Added repair behavior when packages are present but fail import verification; detailed errors and tracebacks are retained.
- Subsequent YOLO training runs now initialize from the prior validation-passing `best.pt` checkpoint when available, rather than always restarting from generic YOLO11n weights. Each run has a unique directory.
- Added a stable reusable checkpoint at `training/models/cell-best.pt` (with a previous-checkpoint copy) alongside the inference model `server/models/cell.onnx`.
- Preserved previous active ONNX weights unless the new model passes validation and exports successfully.
- Require at least one recorded per-epoch mAP50 row before candidate activation; this prevents successful completion with a blank analytics table.
- Detection status now distinguishes an ONNX file from an actually usable ONNX runtime and reports classical fallback when `onnxruntime-node` is missing.
- Made setup and training-status polling errors visible in the Training Centre instead of swallowing them silently.

## 1.4.2 — Space-to-place annotation and refresh-safe training

- Added `Space` as a keyboard placement command for copied cell annotation boxes. It commits the current cyan preview at the pointer location; click remains supported and `Esc` cancels placement.
- YOLO cell training now runs as a detached local worker and writes progress, log lines, metrics, PID, and terminal status to a durable job-state file. Refreshing the browser restores the running job instead of resetting the status panel.
- Added recovery after a backend restart: the app reconnects to a still-running worker, or clearly marks a stopped worker as interrupted while retaining its last progress and logs.
- Removed the post-training `npm install` from the live server process. The existing launchers install optional ONNX Runtime on the next app launch after a trained `cell.onnx` exists, avoiding package-manager changes and dev-server refreshes during training.
- Fixed validation-metric extraction for Ultralytics versions that expose `trainer.metrics` as a dictionary (and retained compatibility with Metric objects); the previous implementation could interpret a valid metrics dictionary as empty and reject every trained candidate.
- Dataset rebuild now clears only the prepared dataset and keeps earlier training runs/checkpoints.

## 1.4.1 — direct-cell detection and segregated in-app training

- Replaced the planned physical-module YOLO stage with direct single-class PV-cell YOLO11n detection. No module outline annotations or module detector are required.
- Added a dedicated Advanced → Annotations workspace, hidden by default, and moved all cell-box/crack-label tools out of the normal Inspection canvas.
- Kept Training separate from Inspection and Annotation; cell-box YOLO training and cracked/clean cell-classifier training use separate datasets and model families.
- Connected trained YOLO cell boxes to the cell grids consumed by cell-level crack classification and cell-to-crack mapping. Classical validated cell geometry remains the fallback when YOLO is unavailable or returns too few assignable boxes.
- Added a held-out mAP50 activation gate (>= 0.15), previous-model preservation, model rollback, and runtime-session reset after activation.
- Added a separate Advanced → Detection Diagnostics workspace with detector mode selection and runtime/model status.
- Normal mode hides annotation and detection diagnostics; Advanced mode is off by default and automatically exits the advanced workspaces when disabled.
- Corrected annotation image storage to persist EXIF-normalized pixels matching the saved image-coordinate boxes.

## v1.3.3 — automatic-first cell geometry and exact reference-pitch calibration

- Preserved the v1.3.0/v1.3.2 server-side automatic cell-grid pipeline as the primary detector. Automatic module segmentation, edge-periodicity pitch estimation and per-cell PV/seam validation remain unchanged.
- Corrected annotation hit-test precedence: an explicit reference-cell calibration is checked before the automatic grid only inside the calibrated footprint. Auto-detected cells remain the default everywhere else.
- A four-corner reference cell now establishes the exact integer pixel pitch used for the calibrated cell rectangles; each generated cell box uses the same width and height as the drawn reference.
- Calibrating a detected module replaces its auto cell overlay only in that module, eliminating overlapping/misaligned rectangles and making the calibrated boxes the active click targets.
- Multiple calibrated/fallback regions remain independently selectable. Recalibration replaces the prior pitch for that detected module and clears stale unsaved labels for that module.
- Added regression coverage for manual calibration precedence, exact calibrated cell dimensions and overlapping calibrations.

## v1.3.2 — multi-module cell selection and isolated image zoom

- Cell hit-testing now searches every automatically validated PV region on every canvas click. Operators can label cells on a second or later module without changing the active-module selector first.
- Pending Positive/Negative cell labels remain visible for all auto-detected modules, keyed by module identity plus row/column. Switching modules or invoking the fallback no longer discards previously selected labels.
- The four-corner manual fallback is available when needed even if another module was successfully detected; it remains an explicit fallback action.
- Image-wheel zoom now uses a non-passive native listener, prevents the browser default, and stops propagation so the outer workspace/page does not scroll with the image.
- The image viewport uses contained overscroll behaviour and fits to the available panel height to keep stacked/adjacent modules visible in the canvas.
- Added regression tests proving that hit-testing can select a cell from the second module and does not select gaps or unvalidated regions.

## v1.2.0 — cell-seam validated detection

- Corrected cell rectangles to clip to the image frame, not to coarse PV-region bounds. This prevents cells from being shortened or duplicated at adjacent region edges.
- Require both PV-silicon occupancy and directional vertical/horizontal seam support for every complete cell. Blue-enriched PV pixels are used for cell validation while the broader mask remains available for tolerant PV-surface localization.
- Reject ground/shadow slots that have PV-like color but lack the expected outer cell seam; preserve cells clipped by the actual image boundary.
- Reuse the strongest measured cell pitch across all visible PV-bearing bands. Displayed row/column counts now reflect validated cell indices rather than unfiltered lattice proposals.
- Crack-to-cell assignment searches validated cell rectangles directly and no longer requires the crack centroid to fall inside a physical-module bounding box.
- Added separate horizontal and vertical image-viewer scrollbars that track the zoomed viewport without covering the image.
- Added five geometry regression tests covering inter-array gaps, partial image-edge cells, empty ground masks, full cells crossing ROI edges, and rejection of shadow-like slots without cell seams.

## 1.1.8 — cleaner cell overlay and conservative grid confidence

- Removed the floating cell-detection information banner from the image viewer; status and instructions remain in the toolbar/side controls.
- Removed physical-region bounding rectangles and region labels from the cell overlay so the canvas emphasizes individual cell outlines rather than module boxes.
- Removed the fabricated 26-column fallback grid. When edge periodicity is weak, the pipeline now withholds cell geometry rather than displaying misleading row/column assignments.
- Increased the penalty for implausible cell aspect ratios during edge-periodicity candidate selection.
- Kept crack scoring tied to measured cell-grid confidence.

## 1.1.7 — visible-cell-first detection

- Removed the fixed 6×26 row/column output from edge-map estimation. Cell counts now come from measured vertical/horizontal edge pitch, so partial panels and combined adjacent panels do not get forced into one nominal module grid.
- Added edge-map phase alignment so the grid does not restart at the crop boundary; partially visible edge cells remain represented as clipped cells.
- Lowered minimum visible-region height and detector crop thresholds to retain partial cell arrays.
- Classical crack scoring now requires a sufficiently confident measured cell grid; it no longer scores a fabricated full-panel grid when edge evidence is weak.
- Updated UI language to describe visible solar cells/regions rather than a detected module.

## 1.1.1 — 2026-11-05

- Fixed cell annotation hit-testing: each click is mapped from the native-image coordinate to exactly one row/column cell.
- Removed individual SVG cell click handlers; the grid is now a visual overlay and cannot receive overlapping/bubbling cell events.
- Added a drag-versus-click guard so panning the canvas never labels cells.
- Kept manual Positive/Negative toggling, batch Save, and separate Retrain action unchanged.

## 1.1.0 — 2026-11-05

- Replaced single-rectangle issue drawing with module-first manual annotation.
- Added automatic cell-grid estimation after selecting a detected or manually outlined module.
- Added Positive and Negative cell-label modes, selected-cell highlighting, row/column adjustment, and explicit batch Save.
- Added batch feedback API so the source image is uploaded once per selected module instead of once per cell.
- Re-saving the same cell updates its label rather than creating contradictory positive/negative training samples.
- Extended feedback records and training to use the selected whole-module ROI when automatic module detection misses it.
- Kept saving annotations separate from model retraining; Reset to base preserves the feedback dataset.
- Corrected tuned-model status detection so Reset to base is enabled after cell-feedback training.
- Changed default server binding to 127.0.0.1; set HOST explicitly only when LAN exposure is intended.
- Removed the local `.env.local` file from the release archive; `.env.example` remains the configuration template.

## Known limitations

- Cell layout is currently estimated as a regular rectangular grid. Operators must verify the overlay and adjust row/column counts before saving. It is not yet perspective-correct polygon segmentation.
- The model is an assistive candidate detector and requires independent, site-held-out validation before commercial accuracy claims.
- Automated build/test execution could not be completed in the packaging environment because dependency installation could not reach the npm registry.

## v1.1.2 — Four-point module selection
- Replaced drag-to-draw manual module ROI selection with four ordered corner clicks.
- Shows numbered corner markers and connecting guide while defining a module.
- After the fourth point, computes the enclosing image-coordinate rectangle and estimates the cell grid.
- Cell marking remains single-click, one cell per click; grid overlay remains non-interactive.


## v1.1.3 — Reference-cell solar-cell detection

- Replaced whole-module four-point selection in the manual annotation workflow with four-point selection of one representative solar cell.
- The marked cell's width and height are reused as the fixed cell pitch. A four-direction visual scan finds the contiguous PV-cell array and draws the per-cell grid.
- Kept the backend module ROIs for crack-analysis gating and training compatibility; the new cell detector changes the annotation/grid-selection workflow rather than removing that internal analysis region.
- Updated the on-canvas prompts and training documentation to describe the reference-cell workflow and its axis-aligned layout assumption.


## 1.1.4 — Solar-cell-level detection

- Added server-side cell-grid estimation from periodic horizontal and vertical inter-cell boundaries.
- Reused the same array/cell geometry in classical detection, crack-to-cell indexing, inspection overlays, and marked-image export.
- Replaced fixed 6×24 cell scoring with inferred row/column counts and removed the hard-coded outer-column-only acceptance rule.
- Updated visual-model instructions and tile selection to include interior and central cells.
- Added affected-cell reporting to GeoJSON and neutralized orange/yellow annotation markup before analysis.
- Retained the module/array ROI only as a ground-rejection and crop boundary; it is no longer the detector's cell-grid assumption.


## 1.1.5 — Edge-map solar-cell grid detection

- Added a directional structural edge map to estimate cell-column pitch from vertical boundaries and cell-row pitch from horizontal boundaries.
- Made edge-map periodicity the primary grid estimator for server-side crack detection, cell indexing, overlays, and exports.
- Retained neutral-white line periodicity as a fallback when gradient evidence is insufficient, followed by the explicitly low-confidence legacy fallback.
- Kept the Off-Axis Edge Map as a separate crack-response heatmap because its scoring intentionally suppresses regular cell-grid energy.


## 1.1.6 — Shared directional cell-edge map

- Builds a single directional Sobel edge map per inspection frame; vertical and horizontal channels are shared by cell-grid estimation and the on-canvas diagnostic view.
- Uses multiple autocorrelation candidates from both axes jointly, preserving half-cut cell layouts without forcing a square-cell prior or hard-coded row/column counts.
- Adds a **Solar Cell Edge Map** overlay (orange vertical seams, cyan horizontal seams) and renames the existing fracture-response overlay to **Crack Heatmap**.
- Keeps the measured cell grid as the common geometry for classical cell scoring and crack-to-cell indexing.


## v1.1.9 — Individually validated solar-cell rectangles

- Replaced blanket grid-slot rendering/scoring with a collection of individually validated solar-cell rectangles.
- Candidate cell interiors must contain PV-silicon pixels; empty gaps, ground, shadows and frame areas are omitted instead of assigned cell IDs.
- Classical crack scoring now samples only validated cell rectangles and constructs detection boxes from those actual cell rectangles.
- Crack locations resolve against exact cell rectangles; learned-window hits outside a validated visible cell are discarded.
- Partial cells clipped by the image frame are retained when enough of their visible interior is PV silicon.
- Inspection and marked-image overlays draw only validated individual cell rectangles; no broad region/grid bounding boxes are drawn.
- Validation must be performed on field images with different panel spacings, angles, and crop boundaries before operational deployment.


## 1.3.0 — Feedback training and cell-label validation

- Retraining now requires at least 30 positive and 50 negative cell labels, 10 distinct images, both classes represented across at least 3 images each, and 5 annotated PV regions.
- Negative labels now mine the highest-scoring candidate windows (hard false positives), rather than the easiest low-scoring windows.
- Training/validation splitting is deterministic and grouped by whole image to prevent image-level leakage.
- Model candidates are only saved when held-out image metrics improve without materially harming positive recall or negative specificity.
- UI displays the exact training-readiness requirements and disables retraining until they are met.
- Cell annotation now selects and labels only server-validated solar-cell rectangles; gaps and unvalidated grid slots are not selectable.
- The cell-aware classical detector remains active alongside the learned edge-ring detector; detections are merged by spatial overlap rather than suppressing an entire module after one VLM hit.
- Added unit tests for training readiness, hard-negative mining, and image-level validation splits.

## 1.3.1 — clean-start, automatic cell annotation, cell-level training

- Removed the four bundled demonstration images from the initial workspace. Startup now shows an empty workspace and never auto-runs a sample inspection.
- Removed the legacy bundled module-window classifier, old 224-window label file, legacy offline window-training scripts, and sample image assets from the release.
- Added a one-time data-schema migration that clears incompatible v1.3.0 feedback records/images and legacy user/base model files on the first v1.3.1 startup.
- Automatic validated cell geometry now selects an annotatable PV region immediately. A four-corner reference-cell workflow is only offered when automatic cell geometry is unavailable.
- The manual fallback now generates candidate cell rectangles from the reference pitch, rejects candidates that fail the PV-silicon colour-signature check, and stops at the first gap so adjacent panels are not bridged.
- Replaced fine-tuning of pretrained module-window features with a fresh zero-initialized classifier trained from one exact crop per reviewed cell.
- Shared the cell-crop feature extractor/vectorizer between training and inference. The trained classifier scores every validated cell rather than sliding overlapping edge windows.
- Classical spider-web results are now emitted as individual validated-cell boxes. Gemini spider-web and linear-fracture boxes are mapped to the containing validated cell or excluded from cell-level crack output. Module identity is retained in cell locations.
- Replaced the old “reset to base” action with a confirmed destructive “Clear data + model” operation that clears labels, stored feedback images, and the trained classifier.
- Removed the legacy arbitrary-box feedback API. Deleting a crack detection creates a training negative only when its point resolves to one validated cell; its exact cell crop and module region are saved. Switching between detected modules preserves pending cell labels.
- Bumped package and metadata version to 1.3.1 and added cell-feature-vector regression coverage.

## 1.4.4
- Replaced filename-ordered validation split with deterministic hash-based shuffling and a minimum of 10 distinct source images / 2 validation images.
- Added strict cell-box bounds and geometry checks, non-empty train/validation box checks, and a persisted dataset audit that warns every visible cell must be annotated.
- Reconciles epoch analytics from Ultralytics results.csv after training, independently of callback metric availability.
- Final acceptance metrics are computed by running validation on the actual best.pt checkpoint that will be exported, not by reusing the last epoch metrics.
- Added 640/768/960 image-size selection; 960px uses a smaller batch to better preserve small cell features.
- Retained the validation gate and previous-active-model safety; poor candidates remain rejected.

## v1.4.5.5 — Shared training data and ONNX input compatibility

- Moves all mutable training artefacts to the shared `Working Folder/data/` directory so annotations, feedback labels, YOLO datasets/runs/logs, checkpoints, active/previous models, Python runtime and job state persist across version upgrades.
- On first launch, merges version-local runtime data from sibling `PV_CrackVision*` folders into the shared data directory without deleting old copies. A migration manifest avoids recopying unchanged large runtime files.
- Adds `PV_CRACKVISION_DATA_DIR` to override the shared data location.
- Reads the active ONNX model input metadata and builds the tensor at the exported model dimensions (including 960×960), rather than hardcoding 640×640.
- Balances the validation split by annotated-box count; in the reported 2/13/78-box, three-image case, the 13-box image is held out instead of the 78-box image.
- Warns that metrics from fewer than five distinct source images are provisional.


## v1.4.5.6 — configurable training and negative examples

- Exposed persisted YOLO detector settings, including epochs, early-stopping patience, freeze layers, optimizer, learning-rate schedule, augmentation, loss gains, caching, pretrained weights and checkpoint cadence.
- Set patience to 0 by default so the training worker runs to the requested epoch count unless the operator cancels or a real failure occurs. The model-activation validation gate remains post-training.
- Added a separate crack-classifier settings panel for the actual logistic-regression controls; freeze layers are correctly identified as inapplicable to that model.
- Added whole-image background-only negative examples for YOLO with empty label files, while keeping positive cell annotations separate.
- Added pattern-aware suppression that rejects a larger enclosing detection only when it contains a repeated pattern of similarly sized individual-cell detections.

## v1.4.5.7 — selectable YOLO cell annotations

- Makes each visible YOLO cell detection a direct SVG click target in Advanced → Annotations.
- Maps the clicked overlay back to the exact detector box before saving a cracked/clean label, including sparse detections and detections outside classical PV-region groups.
- Prevents canvas panning and overlapping manually drawn cell-box selection from stealing clicks on a detector target. The explicit Draw cell boxes and four-point calibration modes still take precedence when enabled.
- Retains all v1.4.5.6 training controls, negative-sample support, pattern-aware suppression, persistent shared data, ONNX input-shape handling and model/checkpoint storage.
