# v1.4.8.5 validation
- tsc --noEmit and vite build run on a scratch copy. Tests not run, no test files changed.
- Verify in browser: save labels -> popup; save the same image again -> confirm dialog with counts; OK replaces (counts in Training Centre do not double), Cancel keeps old labels.
