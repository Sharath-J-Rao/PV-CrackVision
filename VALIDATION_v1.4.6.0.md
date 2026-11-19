# v1.4.6.0 validation
- `npx tsc --noEmit`: clean. `npm test`: 71 pass. Python worker tests: 14 pass. `vite build`: OK.
- Server started; /api/health and /api/cell-detect/upload (no model installed) respond without error.
- NOT verified: onnxruntime-node inference (its postinstall is blocked in the build sandbox), a real YOLO training run, the UI in a browser, and the viewer with a real model. Run through once with a real image before the demo.
