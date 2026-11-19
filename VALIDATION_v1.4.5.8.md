# v1.4.5.8 validation notes

## Changes
- YOLO cell detections are promoted into the editable annotation-box state on the annotation canvas. They are no longer drawn only as a separate static detection overlay.
- Existing box editing operations (selection, move, multi-select, copy/paste, delete, redraw and save) operate on those promoted YOLO boxes.
- Classical module-region grids are excluded from annotation hit-testing for images whose detector source is YOLO.
- Once YOLO inference executes, zero detections remain an empty YOLO result and inference errors are reported as YOLO errors; the detector no longer silently invokes classical fallback after an attempted YOLO run.

## Checks run in this environment
- 22 targeted Node tests passed: annotation hit testing, YOLO annotation target creation, box placement/movement and detection overlap geometry.
- ZIP integrity checked after packaging.

## Not claimed
- No Windows browser interaction test was performed.
- No full TypeScript semantic check or Vite production build is claimed.
- Actual model accuracy and ONNX inference against the user's installed model remain dependent on their local runtime and model.
