# PV CrackVision v1.3.0 — validation report

## Checks completed

- `npm test`: **11/11 passed**.
  - 5 solar-cell geometry regressions (gaps, partial image-edge cells, empty ground, ROI crossing, shadow slot without a seam).
  - 6 training safeguards (readiness thresholds, image/class diversity, hard-negative selection, grouped image holdout, and held-out acceptance/rejection).
- TypeScript/TSX syntax pass using the globally available TypeScript transpiler: **19 files parsed with no syntax diagnostics**.
- Release package version consistency checked: `package.json`, `package-lock.json`, and page title are all `1.3.0`.

## Checks not completed

- `npm ci` timed out in this environment, so project dependencies could not be installed.
- Consequently, `npm run build` and the full `tsc --noEmit` typecheck were not run. The direct `tsc` attempt confirms the missing `@types/node` and `vite/client` declarations.
- The end-to-end server trainer was not run against live image feedback because its runtime dependencies (`sharp`, Express, Vite, and application modules) are not installed in this workspace. The included tests exercise the pure training/validation helpers and cell-geometry regression suite.

Do not interpret the passing unit tests as a field-accuracy certification. Review held-out drone images before operational use.
