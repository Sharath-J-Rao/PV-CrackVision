# v1.4.8.6 validation
- tsc --noEmit and vite build run on a scratch copy. Tests not run, no test files changed.
- Verify in browser: save module A, then module B on the same image -> dialog; Add to existing keeps A and adds B; Replace keeps only B; Cancel changes nothing.
