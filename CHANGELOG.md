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

# Changelog

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
