# Offline crack model (assistive)

- `server/models/crackModel.json` – logistic regression (16 hairline features + per-image relative features) on 210px
  edge-ring windows at canonical scale (module width -> 2200px). Runs in TypeScript (`server/crackModel.ts`), no Python at runtime.
- `training/tools/extract.mts` – feature extractor (same code path as production). `training/py/train.py` – leave-one-table-out
  training/export. `training/labels.json` – 224 windows I reviewed visually (labels are NOT expert-verified; unreviewed windows
  may contain cracks).
- Gates: module localisation + shadow veto, ground gate, frame-bar inner box, colour gate, cell-texture gate (`hasCellTexture`).

## Honest limits
- Leave-one-table-out AUC 0.98 is inflated by easy negatives. On 10 unseen DJI `_Z` frames the first version produced 69 hits,
  most on ground/hardware; the texture gate cut this to 6, but their precision has NOT been confirmed. Treat output as candidates
  for human review, not as a verdict. Retrain after reviewing real hits (add false hits as negatives) using `training/py/train.py`.
- Metadata: filenames with ` (1)` suffixes and `_hash` are normalised before matching the GeoJSON camera positions.

## User feedback (in-app)
`training/feedback/feedback.json` + `images/` hold the issues users added (label 1) and false detections they deleted (label 0),
with boxes in the analysed image's pixel space. The in-app **Retrain model** (`server/trainer.ts`) fine-tunes the current weights on them
and writes `server/models/crackModel.user.json`, which takes precedence over `crackModel.json` until reset.


## Module-first manual labels (v1.1.0)

The Inspection Stage now lets an operator select a complete module (using an existing module ROI or a manual drag), estimate its regular cell grid, label individual cells Positive or Negative, and save the labels in one batch. The selected module ROI is stored with each label so the trainer can use the operator-selected module even when automatic module localisation misses it.

- Positive cells are stored with label `1`; negative/clean cells with label `0`.
- Save persists labels; **Retrain model** is a separate explicit operation.
- **Reset to base** removes `crackModel.user.json` but preserves `training/feedback/`.
- This reference-cell workflow is retained for describing the annotation interaction; v1.3.0 now maps the reference selection onto server-validated solar-cell rectangles and does not permit unvalidated grid slots to be labelled.
- The base training labels are still not expert-verified. Manually supplied labels should be as accurate and representative as possible.


## In-app feedback model safeguards (v1.3.0)

The web trainer has hard readiness gates: 30 positive cell labels, 50 negative cell labels, 10 distinct images, positive labels from at least 3 images, negative labels from at least 3 images, and 5 distinct annotated PV regions. Negative labels mine the highest-scoring windows in the labelled cell so the trainer learns from false alarms, not already-easy clean patches.

The trainer creates a deterministic holdout by whole image. No windows from a validation image are used in coefficient updates. It reports held-out recall and specificity before/after fine-tuning and refuses to overwrite the user model unless balanced accuracy or log loss improves without a class-rate regression greater than 3 percentage points.
