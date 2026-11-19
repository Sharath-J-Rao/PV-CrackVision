# PV CrackVision v1.4.10

- Cell detection: `auto` = YOLO first; a genuine classical cell-grid fallback now runs when cell.onnx / onnxruntime is missing, YOLO fails or finds no cells. `yolo11n` stays strict (no fallback); `classical` runs the classical grid only.
- Crack CNN: configurable early stopping (patience, default 8) on held-out balanced accuracy; the best epoch's weights are always restored.
- YOLO cell training: patience now defaults to 30 (configurable); best.pt (best epoch) is validated and activated.
- Pause / Resume (and Abort while paused) for both the crack-classifier and YOLO cell trainings.
- Crack stage: "Run selected image only" button.
- Module-perimeter cells: lower decision threshold (`edgeThresholdDrop`), reported first, and higher training weight (`edgeSampleWeight`).
- Review: deleting a false detection saves a hard negative (`source: rejected-detection`) trained with `hardNegativeWeight`.
