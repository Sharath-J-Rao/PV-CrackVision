# PV CrackVision v1.4.5 validation report

## Changes verified in source

- Direct cell-box saves for an existing image merge boxes instead of replacing the previous list.
- Reviewed cell labels from `training/feedback/feedback.json` and Advanced direct boxes from `training/cell_annotations/annotations.json` are both ingested by the direct YOLO cell-training dataset builder.
- Status counts unique source-image records across both stores and includes both annotation workflows.
- The hard-coded 10-image restriction is removed. Two distinct source images are required to preserve an image-level train/validation holdout; both splits must contain at least one valid box.
- Dataset creation writes `dataset_audit.json` with per-image split and box counts, total/train/validation counts, and the source workflows.
- Candidate activation still requires final validation of the actual `best.pt`, successful ONNX export, and a passing validation gate. The previously active ONNX model is not replaced on failure.

## Automated checks

The archive includes a regression test for repeated same-image annotation saves. The source was syntax-checked and the ZIP archive integrity was checked during packaging. A full dependency installation/build and a real Windows YOLO training run require the user's Windows machine, Python runtime, and actual labelled PV images; those are not represented as completed here.

## Important training-data condition

YOLO object detection treats every unboxed visible cell as background. Fully annotate all visible cells in every source image used for direct cell localization. The reviewed positive/negative cell labels are valid cell boxes and are now included automatically. A two-image holdout is technically independent but statistically limited; add more distinct representative images for a reliable generalization estimate.
