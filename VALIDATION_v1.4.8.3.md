# v1.4.8.3 validation
- `tsc --noEmit` and `vite build`: see result in the release notes of the hand-off message (run on a scratch copy).
- Tests were not run and no test files changed (ask-first rule).
- Verify in the app: Cell Annotation -> run Cell detections -> Crack Annotation -> label cells -> Save; counts under Training Centre 02 · Crack classification (Cracked / Clean / Images) should rise. Retry with an image larger than 8,000 px and confirm labels sit on the right cells after training.
