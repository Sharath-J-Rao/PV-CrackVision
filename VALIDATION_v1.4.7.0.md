# v1.4.7.0 validation
- `npx tsc --noEmit`: clean. `npm test`: 76 pass (4 new cell-type filter tests). `vite build`: OK.
- Server started; /api/training-settings returns the new sections with defaults equal to the old hard-coded values; /api/cell-type-filter/learn returns a clear error with no saved boxes.
- NOT verified: UI in a browser, onnxruntime inference with a real model, a real training run, the "Rest of module = negative" save against real labels. Do one pass with a real image before the demo.
