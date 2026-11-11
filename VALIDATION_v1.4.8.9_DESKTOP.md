# v1.4.8.9 Windows desktop packaging notes

## Included

- Electron 45.2.3 desktop entry point (`electron/main.cjs`).
- Existing Express/TypeScript server runs in Electron's bundled Node runtime through `tsx`; no external Node installation or batch launcher is required for the packaged app.
- Production UI served from Vite `dist/` in local loopback BrowserWindow.
- YOLO Python worker entry points (`training/train_cell_yolo.py` and `training/install_package_dependencies.py`) are included and unpacked outside `app.asar`; the external Python process receives those physical paths and runs with the writable data folder as its working directory.
- Electron Builder 26.18.1 config for Windows x64 NSIS installer and portable executable.
- First-run shared-data resolution: explicit environment path, saved user config, adjacent `Working Folder/data`, then per-user `%APPDATA%` data; interactive selection preserves a pre-existing user data folder when app is installed elsewhere.
- User data is stored outside the install folder and is not deleted on uninstall.
- GitHub Actions workflow builds/tests Windows artifacts and uploads the EXE files.
- PowerShell scripts for local build and branch-based GitHub publishing.

## User data and model handling

The source release and GitHub branch do not include trained weights, annotations, datasets, user images, training checkpoints, Python environments, API keys, or user training logs. `.gitignore` excludes the `data/` tree and the legacy in-source training/model locations. The packaged app uses the selected shared data folder. It does not overwrite or reset model files during install/uninstall.

Known model paths include `data/models/cell.onnx`, `data/training/models/cell-best.pt`, and `data/models/cellCrackModel.user.json`; annotations and training data are under `data/training/`. Actual presence depends on the user's existing local data folder.

## Verification performed in this environment

- Electron data-root unit tests: run with Node's built-in test runner.
- JavaScript syntax checks for Electron entry/config and PowerShell static review.
- Project ZIP archive integrity.

## Not verified here

This environment is Linux, has no Windows GUI, has no locally installed Electron/electron-builder packages, and could not reach the npm registry during the build probe. Therefore no genuine Windows installer/portable EXE was compiled or launched here. The included Windows GitHub Actions workflow is the build route: it must complete successfully on GitHub Actions before its EXE artifacts can be treated as verified deliverables. Windows end-to-end runtime, native `sharp`/`onnxruntime-node` loading, data-folder discovery, and user-machine smoke testing remain pending that run. The generated EXEs will be unsigned unless a code-signing certificate is configured.
