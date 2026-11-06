# PV CrackVision

**Version 1.1.7** — visible-cell-first detection uses measured edge pitch and phase, including partial cells and cropped panels; the crack heatmap remains separate from cell geometry.

Aerial PV-module glass-fracture inspection (React + Express + sharp). Runs fully locally.

## Run locally
Double-click `start.bat` (Windows) or run `./start.sh` (macOS/Linux): installs on first run, starts the server, opens the browser.
Manual: `npm install && npm run dev` -> http://localhost:3000 (`PORT` / `HOST` override). Needs Node 20+.
Production: `npm run build && npm start`.

## Engines
| Mode | When | Detects |
|------|------|---------|
| Gemini + classical | `GEMINI_API_KEY` set in `.env.local` / `.env` | spider-web, impact, linear fracture, residue |
| Classical-offline  | no key, placeholder key, or `CV_ENGINE=classical` (`npm run offline`) | solar-cell-level spider-web clusters across interior and perimeter cells (conservative; may miss faint cracks) |

If Gemini fails mid-run (quota/network), the image is still returned with a `warnings` note and
the classical result. `/api/health` reports the active engine.

## Env (see `.env.example`)
`GEMINI_API_KEY`, `CV_ENGINE`, `METADATA_JSON`, `GEMINI_PRIMARY_MODEL`, `GEMINI_FALLBACK_MODEL`, `PORT`, `HOST`,
`RATE_LIMIT_PER_MIN` (default 600), `CV_DEBUG=1` (logs per-cell crack densities).

## Output: marked images
The result of an inspection is the original frame with every detected issue drawn on it (boxes, crosshair, id/confidence,
solar-cell grid, header with verdict). Download one frame from the Inspection Stage, or open **Marked Images** to preview
each frame, download single JPGs, or get all of them as one ZIP. Images are rendered in the browser at full resolution.

## Solar-cell detection (used during every inspection)

The server builds a directional structural edge map from image gradients. Vertical edge responses estimate column pitch and horizontal edge responses estimate row pitch. Grid phase is aligned to the observed edge pattern rather than restarted at a detected panel/ROI boundary. Visible row and column counts come from measured pitch, not a fixed 6×26 module template; cropped and partial panels can therefore yield partial rows/columns and adjacent panels do not get merged into one nominal module grid. The exact same directional map is available in the Inspection Stage as **Solar Cell Edge Map** (orange = vertical seams, cyan = horizontal seams). **Crack Heatmap** remains a separate fracture-response view; it is not used as cell geometry because it suppresses regular cell-boundary energy. The offline crack detector evaluates interior and perimeter cells; the VLM prompt and tile selection also reserve coverage for central cells. Crack records are reported as `Cell R# · C#`, and GeoJSON includes `affected_cells`. If the image does not contain enough visible grid-line evidence, classical cell-level crack scoring is skipped for that region rather than inventing a full-panel grid.

This is automatic and does not require the operator to select a reference cell before running detection. The **Select Reference Cell** action below is for manual training labels only.

## Manual training data collection (reference-cell first)

1. Run an inspection and open the image in the Inspection Stage.
2. Click **Select Reference Cell (4 points)**. Mark the four corners of one complete, representative solar cell in this order: top-left, top-right, bottom-right, bottom-left. Use the orange example box as the sizing reference.
3. CrackVision uses that cell's width and height as the repeated cell pitch. It scans outward from the reference cell in four directions and stops at the panel frame/gap or the end of the visible image. The cyan overlay shows the inferred cell array. Review it and use **Rows** / **Cols** +/- if one or more edge cells need adjustment.
4. Choose **Positive** and click once inside every cell containing a confirmed crack. Choose **Negative** and click once inside representative clean cells. The canvas maps each click coordinate to exactly one cell; dragging pans the image and does not label cells. A selected cell is highlighted green (positive) or red (negative). Clicking a selected cell again with the same class clears it; selecting it with the other class changes its label. Re-saving the same cell updates its previous label rather than creating a contradictory duplicate.
5. Click **Save**. All selected labels for that array are uploaded together and stored in `training/feedback/`. Saving adds examples to the training dataset; it does not immediately change the active model.
6. Repeat for other arrays and images. When the dataset is ready, click **Retrain model**. Re-scan images to evaluate the tuned model. **Reset to base** removes the user-trained model but deliberately preserves the saved feedback dataset.

### Annotation rules
- Label at the **cell level**, after marking one reference cell. Do not draw a separate box around each crack.
- Positive = the selected cell visibly contains the target glass-crack pattern. Negative = the selected cell is a clear example without that target defect.
- If one crack crosses several cells, label each affected cell positive. This teaches the classifier the affected cell appearance; it does not represent the continuous crack as a single segmentation mask.
- Add clean cells from the same module and difficult negatives from other images (busbars, fingers, dust, scratches, glare, shadows, cables and tracker hardware).
- Avoid uncertain, blurred, heavily occluded or too-small examples. Do not label an image/cell negative simply because the crack is not visible at the available resolution.
- The manual annotation grid uses the marked cell width/height as a fixed pitch. It assumes a regular, axis-aligned cell layout; strong perspective distortion, varying cell sizes, occlusion, low contrast, or two touching arrays with no visible frame/gap can affect the inferred boundary. Always inspect and adjust row/column counts before saving. The annotation overlay is a regular rectangular grid, not a perspective-warped segmentation.

Training is a separate, explicit action. The Save button persists your labels; **Retrain model** fits a user model using the saved positive and negative feedback. The original base model remains available for rollback. Feedback is retained when resetting the model.

## Image-centre Point (GeoJSON, still computed internally)
Each image's Point is its footprint centre, chosen in this order (`centroid_method` property):
`geotiff_center` > `lrf_target` (DJI laser-rangefinder target, when `LRFStatus=Normal`) > `gimbal_offset`
(camera shifted along gimbal yaw by relAlt*tan(90+pitch) for oblique shots) > `camera_nadir` > `metadata_json_nadir`
(camera GPS from a loaded raw-image metadata JSON, matched by filename; nadir assumed) > `site_reference` (opt-in).
Load metadata via the UI button, `POST /api/metadata`, or `METADATA_JSON=/path/file.json`. The raw camera position is kept
in `camera_wgs84_lon_lat`. Merged GeoJSON defaults to images with at least one detected issue (toggle in export view).

## Ground rejection
Detections must sit inside a localized PV-array analysis region: clipped to the array interior, then dropped if >30% soil or <50% PV silicon
(config `rejectGround`, default on; thresholds `GROUND_*` in `server/cvPipeline.ts`).

## Notes
- Images without GPS get pixel-local coordinates and are left out of the merged EPSG:3857 layer
  (count in `metadata.skipped_without_georeference`). Enable "site reference" in the UI to place them.
- `npm run lint` = strict `tsc --noEmit`.

### Historical v1.1.2 module-selection workflow (superseded)
The former **Select Module (4 points)** action marked the module boundary. It was replaced in v1.1.3 by the current **Select Reference Cell (4 points)** annotation workflow described above.

### v1.1.6 — Shared directional edge map for cell detection
- Builds one directional cell-edge map and uses its vertical/horizontal responses for cell-pitch estimation.
- Evaluates multiple autocorrelation peaks from both axes together to avoid choosing a strong sub-cell repeat in isolation.
- Exposes the same cell-boundary map in the Inspection Stage (orange vertical seams, cyan horizontal seams).
- Renames the previous Off-Axis Edge Map control to **Crack Heatmap** to distinguish fracture scoring from cell-boundary geometry.
- Keeps the inferred grid shared by classical cell scoring, crack-to-cell mapping, overlays, and exports.

### v1.1.5 — Edge-map solar-cell grid detection
- Added directional structural edge responses for vertical and horizontal cell-boundary periodicity.
- Made edge-map periodicity the primary grid estimator across crack scoring, cell indexing, overlays, and exports.
- Retained neutral-white line periodicity and an explicitly low-confidence legacy fallback for difficult imagery.

### v1.1.4 — Solar-cell-level detection
- Added automatic horizontal/vertical cell-pitch estimation from repeated PV inter-cell gridlines.
- Shared one detected cell grid across classical crack scoring, cell-location assignment, browser overlays, and marked-image exports.
- Removed fixed 6×24 cell scoring and the hard-coded perimeter-column-only filter; interior-cell signals can now qualify.
- Updated VLM guidance to inspect all solar cells and guaranteed center-tile selection where structured array tiles are used.
- Added affected-cell locations to GeoJSON and ignored orange/yellow markup outlines during image analysis.
- Retained a low-confidence legacy grid only when the image does not provide enough line evidence.


### v1.1.9 viewer and grid-confidence changes

The floating cell-detection banner has been removed. The automatic overlay no longer draws physical-region bounding boxes or region labels, and the backend no longer fabricates a nominal 26-column grid when edge evidence is weak. Low-confidence geometry is withheld instead of presented as a valid cell grid.


### v1.2.0 cell-level detection and validation

The detector now uses edge-map periodicity to propose a shared cell pitch, then validates each proposed solar cell independently with PV-silicon occupancy and all visible cell seams. Candidate cells are not truncated at coarse PV-region boundaries. Only image-frame cropping creates partial cells. Ground and shadow regions that lack true cell seams are not emitted as cells. Run `npm run test:cells` for the cell-geometry regression suite. The inspection viewer has separate horizontal and vertical scroll controls for zoomed images.


## User-feedback training safeguards (v1.3.0)

The in-app trainer requires 30 positive cell labels, 50 negative cell labels, 10 distinct images, positive labels across at least 3 images, negative labels across at least 3 images, and 5 annotated PV regions. These are conservative operational gates, not a guarantee of accuracy.

Negative cell annotations mine the highest-scoring false-positive windows in the labelled cell. A deterministic 25% holdout split is made by whole image (never by windows from the same image). The candidate model is evaluated on that held-out set and is saved only when balanced accuracy or log loss improves without reducing either class recall/specificity by more than 3 percentage points. Otherwise, the candidate is rejected and the currently active model remains unchanged.

The annotation UI uses the same server-validated cell rectangles rendered by automatic cell detection. It no longer fabricates clickable boxes from every row/column lattice slot.


### v1.3.0 feedback training and validated-cell annotation

- Labels can only be selected from individually validated cell rectangles returned by the inspection pipeline; clicks in gaps or on ground are ignored.
- Training readiness is checked against minimum positive/negative counts, distinct-image coverage, per-class image coverage, and annotated PV-region count.
- Negative feedback mines high-scoring false-positive windows.
- Candidate tuning is evaluated on whole images held out from training; model weights are only persisted when held-out metrics improve while preserving both class rates.
- Classical cell-aware crack scoring is no longer silently disabled when the learned edge-ring model is loaded. Detections are combined by spatial overlap, not whole-module suppression.
