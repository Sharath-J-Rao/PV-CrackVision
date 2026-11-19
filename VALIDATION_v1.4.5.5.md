# PV CrackVision v1.4.5.5 validation report

## Delivered changes
- Shared writable data root resolves to `Working Folder/data/`, including when a release ZIP is nested inside a named wrapper folder. On the reported Windows layout this is `D:\Python_Projects\PV-Crack Vision\Working Folder\data`. Override with the process environment variable `PV_CRACKVISION_DATA_DIR`.
- The server's first startup imports and merges version-local direct cell annotations/images, reviewed feedback/images, YOLO datasets/runs/logs, checkpoints, active/previous models, Python virtual environment, embedded Python, Ultralytics configuration, runtime marker and durable job state from sibling/nested `PV_CrackVision*` release directories. Original source folders are never deleted. If a copied Windows venv cannot launch, training setup probes the still-present legacy release venv before deciding that a new environment is required.
- A migration manifest prevents recopying unchanged large runtime files. Changed data files are imported on later version switches; annotation arrays are merged by image/id and cell boxes are unioned. Saved job-state paths and YOLO dataset YAML paths are rebased to the shared location.
- Unknown legacy feedback-schema folders are preserved under timestamped `.legacy-*` backups rather than deleted.
- Windows and POSIX launchers check for a trained ONNX model in both the shared data root and old nested release folders before deciding whether `onnxruntime-node` must be installed, so first-launch migration cannot hide an existing model from the launcher.
- ONNX input shape is read from `InferenceSession.inputMetadata[].shape`; image preprocessing/letterboxing and tensor creation use the model's fixed input dimensions, including 960×960 and rectangular inputs.
- Validation images are chosen to bring held-out box counts near the configured 20% ratio. For the reported 2/13/78 annotated-box example, the 13-box image is held out and the 80-box remainder is used for training.
- Training logs warn that metrics from fewer than five distinct source images are provisional.

## Automated checks run
- Node regression suite: 46/46 passed. Includes geometry, annotation editing, direct-cell mapping, dataset summary, validation split balancing, ONNX shape/preprocessing, shared-data path resolution, nested-release discovery, legacy data migration/job-state path rebasing and ONNX launcher detection.
- Python worker/dependency suite: 11/11 passed.
- Python bytecode compilation: passed.
- TypeScript/TSX syntax transpilation: 32 files, zero syntax errors.
- POSIX launcher and Node launcher syntax checks: passed.
- Package, lockfile and app metadata versions: all 1.4.5.5.
- ZIP integrity and root-folder/version consistency: checked at packaging.

## Not verified in this environment
- Full strict `tsc --noEmit` and Vite production build: npm dependencies/type packages are not installed in this environment; the previous attempt failed because `@types/node` and `vite/client` were unavailable. Syntax transpilation is not a substitute for a strict type-check.
- Native Windows launch, Windows shared-folder migration, reuse of the copied Windows Python virtual environment, and actual `onnxruntime-node` inference with the user's trained ONNX file.
- Live CPU PyTorch/Ultralytics installation and end-to-end YOLO training on the user's dataset.

## Dataset interpretation
The user's 35-epoch run showed zero metrics for epochs 1–32, then precision 0.27634, recall 0.51898, mAP50 0.26698 and mAP50–95 0.19091 for epochs 33–35. The initial zeros were early-epoch results, not the final run result. The previous 2/13/78 box split was highly imbalanced; v1.4.5.5 balances the validation box count, but three source images are still too few for reliable generalization estimates.
