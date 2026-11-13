# Bundled AI models (resources/models)

The desktop EXE ships with its AI models so it works **offline with no manual model installation**. Put the two model
files in THIS folder, then build. Nothing else needs to be edited in the code.

## Files to add

| File | What it is | Where to get it |
|------|-----------|-----------------|
| `cell.onnx` | YOLO cell detector (ONNX) | Your data folder: `<data folder>/models/cell.onnx`, or in the app **Data & Models → Export active cell model** |
| `cellCrackModel.user.json` | Crack classifier | Your data folder: `<data folder>/models/cellCrackModel.user.json` |

Do **not** add `.pt` checkpoints (training only) or `cell.previous.onnx`.

## Steps

1. Copy the two files into `resources/models/` (keep the exact file names).
2. Run `npm run models:manifest` — writes `manifest.json` (SHA-256, size, app version the models need).
   Optionally label them: `node scripts/model-manifest.mjs generate --model-version 2026-11-cracks-v3`.
3. Run `npm run models:check` — must print "Bundled model check passed."
4. Commit and push (`git add resources/models && git commit && git push`). Files over 50 MB trigger a GitHub warning and
   over 100 MB are rejected — use Git LFS (`git lfs track "resources/models/*"`) if `cell.onnx` is that large.
5. The GitHub Actions build runs `models:check --require` (the build fails if a model is missing or corrupt), packages the
   folder into the EXE (`resources/models` next to the app) and its smoke test confirms both models loaded.

## How the app uses them

1. A model you trained or imported in the app (in the data folder) always wins.
2. Otherwise the bundled model is used (read-only; never modified or deleted).
3. If a file is missing, corrupt (size/SHA-256 mismatch), not a valid model, or built for a different app version, it is
   skipped with a warning (shown in the detector status and `desktop.log`) and the next option is used; if none is valid
   the app falls back to classical detection instead of crashing.

## Keeping the EXE and models compatible

`manifest.json` records the minimum app version (`minAppVersion`, set to the current app when generated) and optionally
`maxAppVersion`. After upgrading the app to a version that changes the model format, regenerate the manifest only once
you have re-exported compatible models. Never edit the manifest by hand — regenerate it.
