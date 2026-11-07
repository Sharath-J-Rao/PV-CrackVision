# PV CrackVision v1.4.5.3 — validation report

## Corrective changes

- Fixed PEP 508 marker evaluation in `training/install_package_dependencies.py` by supplying `extra=""` when evaluating the base dependency set. This prevents `UndefinedEnvironmentName: 'extra' does not exist` when installed package metadata contains optional-extra requirements.
- Removed `torchaudio` from the default protected-package list in `server/pythonInstallConfig.ts`. This application uses CPU `torch` and `torchvision`; it does not install or require `torchaudio`. The previous default incorrectly aborted dependency resolution when that optional distribution was absent.
- Kept CPU `torch`/`torchvision` binary-wheel installation isolated to the official PyTorch CPU index and dependency resolution isolated to PyPI, with the installed CPU versions pinned during resolution.
- Added regression coverage for PEP 508 `extra` markers and the generated default protected-package arguments.
- Aligned `package.json`, `package-lock.json`, and `metadata.json` to version `1.4.5.3`.

## Checks run in the build environment

- Node regression suite: **28/28 passed** (`npm test`).
- Python unit tests: **10/10 passed** (`python -m unittest training.test_install_package_dependencies training.test_job_state`).
- Python syntax compilation: passed for the dependency helper, YOLO training worker, and dependency-helper tests.
- JSON parsing and version consistency: passed for `package.json`, `package-lock.json`, and `metadata.json`.
- Static installation-path audit: the managed package installation paths use the explicit CPU-wheel/PyPI helper flow; the default protected list contains only `torch` and `torchvision`.
- ZIP integrity and archive-root/version consistency: passed after packaging.

## Not run / not claimed

- A live Windows PyTorch/torchvision/Ultralytics installation was not run in this Linux build environment.
- `npm run build` and `npm run lint` were not run because this extracted source has no `node_modules` directory and dependency installation/build was not performed here.
- No end-to-end YOLO training or ONNX export was performed as part of this corrective release.

The prior v1.4.5.2 runtime failures are specifically covered by the new marker and protected-package regression checks. Network- and Windows-specific installation success remains to be confirmed on the target machine.
