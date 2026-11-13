# PV CrackVision

**Current release: v1.4.15.0.** This release introduces the simplified end-user workflow (upload → cell detections → crack detection → download with original file names); annotation, training and data tabs moved under Advanced mode.

Cell localization (YOLO) and cracked/clean cell classification are separate model workflows with separate datasets and separate training monitors.

PV CrackVision is a React + Express + Sharp application for inspecting photovoltaic imagery. Solar-cell geometry is detected from image structure; crack annotations, training examples, and learned crack inference now use one validated cell as the unit of data.

## Release version management

`package.json` is the single canonical version field. For the next release, run `npm run release:set -- 1.4.15.0` (replace with the desired four-part release label). The script updates the npm lockfile, app metadata, Windows workflow branch/artifact, publisher script, current validation filename and current-release references. The app UI and EXE version are read from package metadata; do not edit version strings independently.

## Run locally

Windows: double-click `start.bat`; it waits for `/api/health` before opening the browser. macOS/Linux: run `./start.sh`.

Manual: `npm install && npm run dev`, then open `http://localhost:3000`. Node.js 20.19+ or 22.12+ is required (matching the Vite 8 runtime requirement). Production: `npm run build && npm start`.

PV CrackVision runs fully offline: inference needs no API key and no internet connection. The trained AI models are bundled with the desktop EXE (see `resources/models/README.md`).

## Training controls and cell-detector background samples

- The **Training Settings** tab independently exposes YOLO cell-detector parameters and crack-classifier parameters. The crack classifier supports **Ensemble (CNN + gradient boosting)**, **CNN only**, **Gradient boosting only**, and **Legacy logistic regression**. CNN controls include epochs, learning rate and batch size; GBM controls include tree count, depth, learning rate, minimum leaf size and subsampling.
- Cell-detector training runs for the full requested epoch count. Early-stopping patience defaults to 0 (disabled), so a validation plateau does not stop training; a positive patience value is an explicit opt-in to early stopping. Use **Cancel** to stop manually. The final validation gate only decides whether the completed candidate may replace the active model.
- Background-only images can be added from Training Settings as negative YOLO samples. Use only images with no visible target PV cells (for example, ground, roof, sky or vegetation). A PV image with unboxed visible cells must never be labelled as a whole-image negative.
- Inference post-processing suppresses a larger detection when it encloses or substantially overlaps smaller repeating individual-cell detections, then performs normal same-scale NMS. This prefers the per-cell pattern over a large enclosing box.
- The training settings JSON is stored under `Working Folder/data/training/` and reused across release folders. Negative images are stored in the shared `data/training/cell_annotations/` directory.

## Guided detection workflow (v1.4.5.9)

The **Detections** workspace has two distinct stages. They are deliberately separate: cell localization is not crack classification.

1. Upload one or more PV images. Uploading only queues the images; it does not silently start crack analysis.
2. Open **Detections → Cell detections** and choose **Run cell detections for all images**. This calls the standalone cell-localization endpoint once for each queued source image and shows per-image status, box count, detector source, warnings, and inference diagnostics.
3. Review each image in the editor. Select a false box and delete it; drag boxes to move them; use the corner handles to resize; Shift-click to multi-select; use **Draw cell boxes** to add missed individual cells. Draw one tight box per visible PV cell, not one rectangle around an entire module. Save the complete corrected box set for that exact image. These saved boxes are YOLO cell-localization training labels.
4. After the cell-detection pass has completed for every queued image, switch to **Detections → Crack detection** and choose **Run crack detection for all images**. This is the separate crack-analysis pass and does not repeat the direct YOLO cell-detector invocation. Crack results are reviewed per image in the Inspection workspace.
5. To train the crack classifier, label cells separately as Positive (cracked) or Negative (clean) in the crack annotation workflow. Cell boxes train localization; cracked/clean labels train crack classification. Do not treat them as interchangeable labels.

If YOLO returns no boxes, the cell review view reports its source and diagnostics instead of pretending boxes exist. Use **Draw cell boxes** to add the missing cell proposals, then save the corrected set. A completed detection pass alone does not mean the image was reviewed or its boxes saved. The crack pass inspects exactly the cell boxes from this stage (the YOLO output plus your edits), so cells are never drawn twice; the classical PV-region step is used only to group cells into panels. The automatic classical cell grid is off by default and can be re-enabled as an explicit fallback in **Crack Detection**.

## Clean-start behaviour

- The workspace opens empty. No demonstration images are preloaded and no image is inspected until the user uploads it.
- Bundled sample imagery and the previous window-level classifier / training labels are not included in this release.
- On first startup, the app imports annotations, feedback, datasets, logs, checkpoints, models and the managed Python runtime from sibling PV CrackVision release folders into `Working Folder/data/`. It preserves the old source folders and merges annotation indexes. If an incompatible feedback schema is encountered, it is moved to a timestamped `.legacy-*` backup rather than permanently deleted.
- No crack classifier is shipped. Classical cell-geometry and rule-based inspection can still run; the learned cell classifier remains absent until the user has collected enough reviewed cell labels and a fresh model passes its held-out-image gate.

The shared-data migration is designed to preserve user files. Keep the original release folders until the shared directory has been checked; they are not deleted by the migration.

## Cell geometry and annotation

1. In v1.4.5.9, upload queues one image, multiple images, a folder, or a ZIP. Run the explicit **Cell detections** stage before the separate **Crack detection** stage.
2. When one or more PV surface regions have validated cell geometry, their individual cells are selectable directly on the image. You can click cells in any region without first selecting a region. **There is no four-point step on the automatic path.**
3. Choose **Positive** for visibly cracked cells or **Negative** for clean cells, then click the actual cell rectangles in the image. Gaps, ground, shadows, hardware, and unvalidated lattice slots are not selectable.
4. If automatic geometry misses a PV surface region or does not produce a validated cell field, the **Calibrate cell pitch (4 points)** control is available after the frame inspection result is ready. Use it only for the missing region; labels already selected in other regions are preserved. Mark the four corners of one representative visible cell in clockwise order. The fallback estimates the neighbouring pitch, stops at the first gap/non-PV position so it cannot bridge adjacent panels, and checks every proposed cell centre for a PV-like silicon signature; cells that fail that check are omitted. It is a fallback, not a replacement for automatic edge-map geometry.
5. Save reviewed labels. The source image is stored once and each annotation records its exact cell rectangle, label, image identity, and associated PV-region bounds for dataset-diversity checks.

The manual fallback is an axis-aligned, colour-signature method. Strong perspective, unusual panel geometry, severe glare, occlusion, and low resolution can still require better imagery or manual review. It does not claim polygon-level perspective rectification. The reference-pitch calibration is currently held in the active canvas session; saved labels preserve exact cell crop boxes, but the full manually reconstructed grid is not persisted as a reusable server-side inference grid. Later detection continues to use the automatic PV-surface geometry as a fallback.

## Cell-level training and inference

- **One reviewed cell = one training sample.** The trainer extracts a fixed-size feature representation from the exact saved cell crop using the same `extractCellFeatures` and `cellFeatureVector` functions used during inference.
- The current crack classifier can train a compact convolutional neural network (CNN), gradient-boosted trees (GBM), or their ensemble from scratch on one crop per reviewed solar cell. Legacy logistic regression remains available only for compatibility; it is not the default CNN/ensemble path. Training uses image-level train/validation separation so crops from one source image do not leak across both sets.
- The initial readiness gates require at least 8 positive cells, 8 negative cells, 4 distinct images, positive and negative labels on at least 2 images each, and 2 distinct annotated PV regions.
- The validation split is grouped by whole image. The decision threshold is selected using training images only. A candidate is saved only if both classes are represented in the holdout and it meets the configured minimum held-out balanced-accuracy gate.
- During inference, the trained classifier scores every individually validated cell. Crack detections use the full cell rectangle and identify its row/column. The classical crack detector also emits individual cell boxes instead of merging neighbouring cells into a large region-sized detection. Gemini spider-web and linear-fracture detections must map to a validated cell or are omitted from the cell-crack result.
- The model is an assistive candidate detector, not a certified diagnostic. Balanced accuracy on this small internal holdout is not a substitute for independent site-held-out testing or field accuracy validation.

## Training data controls

The **Clear data + model** action permanently deletes all saved positive/negative labels, stored feedback source images, and the trained crack classifier. It requires confirmation. This action does not delete the separate YOLO cell-detection checkpoint. To start YOLO cell detection from the generic base weights again, back up and remove `Working Folder/data/training/models/cell-best.pt` deliberately.

Runtime feedback is stored under `Working Folder/data/training/feedback/`; the trained model is written to `Working Folder/data/models/cellCrackModel.user.json`. Both are excluded from the release archive and ignored by Git.

## Output and georeferencing

**Marked Images** keeps the two stages apart: the *Cell stage* shows model detections and your reviewed cells separately, and the *Crack stage* shows crack findings pending review versus those you confirmed (only confirmed findings are the deliverable; crack images draw issues only). The list and the viewer scroll independently. Download an individual marked JPG from the Inspection Stage or use **Marked Images** for batch output. GeoJSON retains the image-level Point representation and includes crack/cell attributes. Camera GPS, GeoTIFF coordinates, and uploaded metadata are used where available; imagery without GPS remains pixel-local unless the user explicitly enables the site-reference fallback.

## Checks

- `npm test` runs the full Node regression suite, including cell geometry, annotation editing/coordinate transforms, multi-image inference tiling, YOLO output decoding, training data paths, and settings.
- `npm run test:training-worker` runs the Python worker-state and per-epoch metrics tests.
- `npm run build` runs the Vite production build.
- `npm run lint` runs strict TypeScript checking.


### Cell-level annotation from detector output

In **Advanced mode → Annotations**, each visible green YOLO cell box is a direct click target. Select **+ Positive** for a cracked cell or **− Negative** for a clean cell, click the exact detected cell box, then save the selected labels. The click target resolves to the exact detector box, not a nearby classical grid slot. The Draw cell boxes and four-point calibration tools retain priority while those tools are active.

All writable user training data is stored in `Working Folder/data/`, resolved even when a release ZIP is nested inside a wrapper folder. It is shared by every release under the same Working Folder. See `VALIDATION_v1.4.5.7.md` for checks completed and environment-specific limitations.


## v1.4.1 — direct cell detection and in-app training

### User-facing workflow

- **Inspection** is the normal-user workspace: upload images, run inspections, review findings, and export marked images.
- **Training** is a separate workspace with independent Cell Detection and Crack Classification panels. It does not contain image annotation controls.
- **Advanced mode** is off by default. It reveals the separate **Annotations** and **Detection Diagnostics** workspaces. Turning Advanced mode off returns the user to Inspection and hides both advanced workspaces.

### Train direct PV-cell detection

1. Upload and inspect a PV image. Enable **Advanced mode**, then open **Annotations**.
2. Select the source image and use **Draw cell boxes**. Draw a tight rectangle around each complete visible PV cell. Do not draw module outlines, cell rows, busbars, or whole arrays. Draft boxes can be selected, copied, pasted, deleted, and saved.
3. Repeat on at least **2 distinct source images**. This is the minimum needed for an independent image-level train/validation split; a larger, representative dataset is strongly recommended for meaningful validation.
4. Open **Training → Direct cell detection (YOLO11n)**. Click **Set up in-app training runtime** once, choose epochs, and start training. On Windows, PV CrackVision can install a private Python runtime when no compatible Python is present, then install CPU-only PyTorch and Ultralytics within its managed training environment. Internet access and several GB of disk space are needed for first setup and model downloads.
5. The app creates train/validation partitions by source image, trains YOLO11n on CPU, records a metrics row after every validation epoch, validates on held-out images, and exports ONNX. The epoch table reports precision, recall, mAP50, mAP50-95, training/validation box, class and DFL losses. A failed validation gate leaves the previous active ONNX model unchanged.
6. Training runs as a detached local worker. Its job ID, process ID, progress, per-epoch metrics, and recent logs are saved in `Working Folder/data/training/.cell-yolo-job-state.json`. The Training Centre displays a scrollable epoch table and log panel; use **Download CSV** for epoch analytics and **Full worker log** for the raw training output. Refreshing the browser restores the same job. Keep the PV CrackVision server running.
7. After the first successful YOLO training, close and relaunch PV CrackVision once using `start.bat` (Windows) or `start.sh` (macOS/Linux). The launcher installs the optional `onnxruntime-node` CPU inference package before opening the app; it is deliberately not installed by mutating `node_modules` during a live training session, which can trigger dev-server reloads.

The YOLO cell detector supplies individual cell boxes to the inspection pipeline. The existing image-structure stage still localizes PV surface regions for grouping and georeferencing; it is **not** a trained module detector and does not require module annotations. When an image has no cell boxes (no cell model installed, or nothing kept after review) the crack pass does not classify it unless the classical-grid fallback is enabled in **Crack Detection**.

### Train crack classification separately

Use the Positive and Negative cell-label controls in **Advanced → Annotations** to save reviewed cracked and clean cell examples. These records are stored separately from the YOLO cell-box dataset. The Training Centre's **Crack classification** panel trains the separate cell-level crack classifier from these reviewed crops. Cell-box labels are never reused as crack labels, and crack labels are never reused as YOLO detection boxes.

### Platform and validation notes

- Windows can bootstrap an app-managed Python 3.12 runtime if Python 3.10–3.12 is not already installed. macOS/Linux currently require Python 3.10–3.12 to be available so the app can create its own environment.
- Training currently uses CPU for compatibility and may be slow. No GPU acceleration is claimed.
- A successful mAP50 gate is an engineering guardrail, not a guarantee of field accuracy. Validate with larger, site-held-out datasets before operational decisions.
- The app does not bundle pre-trained PV-cell weights. The first training run starts from the official YOLO11n base checkpoint downloaded by the app. After a validation-passing run, the reusable PyTorch checkpoint is saved at `Working Folder/data/training/models/cell-best.pt`; subsequent runs fine-tune from that checkpoint by default. The inference model is saved separately at `Working Folder/data/models/cell.onnx`. Back up both files/folders when upgrading releases if you want both inference and continued training. Runtime setup checks installed imports and skips pip/PyTorch/Ultralytics installation when they are already usable.


### Cell annotation box editing (v1.4.1)

- **Move an existing box:** click and drag it to the new cell.
- **Select multiple boxes:** click a box, then `Shift`-click other boxes to add or remove them.
- **Move a selected group:** drag any selected box; the group moves together and keeps its spacing.
- **Copy to a specific cell:** click `Copy selected`, then `Place copy`; move the pointer until the cyan preview is centred on the destination cell and click to place. The preview retains the original box dimensions and group spacing. Press `Esc` to cancel.

The previous automatic diagonal paste offset has been removed.


## Data folder, models and settings (v1.4.8.0)
Open **Data & Models** to choose where annotations, trained models, the Python runtime and settings are kept. The choice is stored per user (outside the release folder) so upgrading or re-extracting a release never loses it. Load a trained model (.onnx / .json / .pt) from the same screen; training and inference settings save automatically.

## Windows launch

Extract the release folder and double-click `start.bat`. It provisions a compatible Node runtime and installs npm dependencies on first launch, so internet access is required for first-time setup. This source release does **not** include a standalone desktop installer or an all-in-one application EXE; do not use an old wrapper executable from an earlier package as a substitute for the complete application folder.


## Crack-classifier training analytics (v1.4.8.9)

The Training Centre now shows a dedicated full-width **Crack-classifier epoch analytics & training log** panel immediately below the page heading, before the two model-training cards. For CNN and Ensemble runs it shows every completed epoch's train loss, held-out validation loss, precision, recall, specificity, F1 and balanced accuracy. For GBM-only and legacy-logistic runs, it displays the selected algorithm and tree progress where applicable rather than inventing CNN epochs.

The monitor also shows the current run stage, progress, crop counts, active saved-model architecture (CNN/GBM/Ensemble/legacy), final held-out metrics, errors/tracebacks and the live worker log. Use **Download epoch CSV** and **Download full log** to save the current run's records. Persistent files are stored under `Working Folder/data/training/crack_classifier/` (`job-state.json`, `training.log`, and `epoch-metrics.csv`). Starting a new classifier run replaces the current monitor records, so download them first if you need to archive them.

## Windows desktop application (Electron)

The Windows desktop build uses Electron 45 and packages the React production UI, Express/TypeScript backend, Node runtime, and native image/ONNX dependencies together. It produces two real Windows x64 executables: an assisted NSIS installer (`PV-CrackVision-1.4.8.9-x64.exe`) and a portable no-install executable (`PV-CrackVision-1.4.8.9-Portable.exe`). Neither is a batch-file wrapper. The installer does not require a separately installed Node.js or a browser.

### Build on Windows

Run `Build-Windows-Desktop.ps1` from this project folder in PowerShell. It installs the project dependencies, Electron 45.2.3 and electron-builder 26.18.1, creates the Vite production build, and writes both EXE files to `desktop-dist/`. An internet connection is required for the first dependency/Electron download.

### Push the desktop release and build through GitHub Actions

Download the desktop publish kit, extract it, and run `Publish_PV_CrackVision_v1.4.8.9_Desktop.ps1`. With the source ZIP beside the script, the archive path is detected automatically. By default it uses `D:\Python_Projects\PV-Crack Vision` as your local Git clone. The script extracts the ZIP to a temporary folder, creates `release/v1.4.8.9-desktop` from `origin/main`, overlays the release source, checks whitespace, and pushes that branch without changing `main`. You can override the clone path with `-RepositoryPath`. The push triggers `.github/workflows/windows-desktop.yml`; the Windows runner tests the project, builds both EXEs, and uploads them as the Actions artifact `PV-CrackVision-v1.4.8.9-Windows-x64` (retained for 30 days).

### Existing models and training data

The EXE contains the application and its runtime, **not your personal trained models or image datasets**. Those remain outside GitHub and outside the installer. This is intentional: it prevents your annotations, original images, checkpoints, and private models from being overwritten during an application upgrade.

The desktop app first tries to find `Working Folder\data` above the portable EXE or application folder. When it finds that folder, it uses it directly, preserving the existing `cell.onnx`, `training\models\cell-best.pt`, `models\cellCrackModel.user.json`, annotations, feedback, datasets and training logs. If the installer is placed elsewhere and no data folder was configured previously, first launch asks you to choose the existing `data` folder or start with a new profile. The selected folder is saved in `%APPDATA%\PV CrackVision\config.json`; uninstalling the application does not delete this data.

For the safest upgrade, keep your current `Working Folder\data` as-is, extract the portable app beneath `Working Folder`, or select that exact `data` folder when the installer asks. Do not copy the data folder into GitHub or into `node_modules`. If you have not yet trained a model, the desktop app can still run the local classical inspection path; learned detection requires the corresponding model to be present in the selected data folder.


## Cell-training backup compatibility (v1.4.15.0)

The Training Centre can export and import a portable ZIP containing the direct cell-detector annotation index, original reviewed source images, and the reusable YOLO checkpoint when present. Import is merge-only: source images already present locally are left unchanged, and a bundled checkpoint is restored only when no local checkpoint exists. Backups are integrity-checked by image hash, and annotation records from older PV CrackVision releases are accepted when their optional newer fields are absent. Existing release-to-release data migration remains enabled. Export a backup before using **Reset cell model + training data** if you may want to reuse the annotations later. The reset does not remove the Python training runtime, training settings, or the separate crack-classifier dataset/model.
