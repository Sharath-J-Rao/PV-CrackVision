# PV CrackVision training workflows (v1.4.5.7)

Training is managed from the PV CrackVision UI. Cell localization and crack classification use different stores and model families. Configure their hyperparameters independently in the **Training Settings** tab; settings persist in the shared `Working Folder/data/training/training-settings.json`. The YOLO detector defaults to `patience=0`, which disables automatic early stopping and completes the requested epoch count; a positive patience explicitly opts into plateau-based early stopping. Use Cancel to stop manually. The crack classifier is logistic regression on handcrafted features, so layer freezing is not applicable to that model.


## 1. Direct PV-cell localization (YOLO11n)

1. Open **Inspection**, upload and inspect a PV image.
2. Enable **Advanced mode** and use the separate **Annotations** workspace.
3. Draw one tight bounding box per complete visible PV cell and save the boxes. Do this for at least two distinct source images (the minimum for an image-separated train/validation split); use more images for meaningful validation.
4. Open **Training → Direct cell detection (YOLO11n)**, set up the in-app runtime, open Training Settings to tune epochs, patience, image size, batch, freeze layers, optimizer, learning rate, augmentations and other detector parameters, then start training.
5. The app builds the YOLO dataset, keeps source images intact across the train/validation split, trains on CPU, validates, exports ONNX, and activates `Working Folder/data/models/cell.onnx` only if mAP50 >= 0.15 and validation metrics are present.

Each epoch records precision, recall, mAP50, mAP50-95, training/validation box, class and DFL losses after the validation pass. Confidence is an inference-time score/threshold, not a standalone epoch validation metric. The application also retains the full raw worker output under `Working Folder/data/training/cell_yolo/logs/`. Setup probes installed imports first and skips package installation when the existing Python environment is healthy. The shared writable runtime and user data are stored in `Working Folder/data/`, outside every versioned source folder. An upgrade imports existing version-local annotations, feedback, YOLO datasets/runs/logs, checkpoints, ONNX models and the Python runtime into this shared directory without deleting the old copies. The default on Windows is exactly `D:\Python_Projects\PV-Crack Vision\Working Folder\data` for the user’s current Working Folder layout; every release nested anywhere below that folder resolves to the same location. All subsequent versions under the same Working Folder reuse the same data directory. The first migration may take time if the previous Python environment or YOLO run history is large; it copies without deleting the original release data. Set `PV_CRACKVISION_DATA_DIR` to override the location. On Windows, the app can bootstrap private Python 3.12 if needed. On macOS/Linux, Python 3.10–3.12 must be available for the app to create its private environment. First setup requires internet access and several GB of free disk space.

The detector predicts individual cell boxes directly from the original image. Existing classical PV-surface localization remains a grouping/georeferencing fallback and does not require module ROIs or a module detector.

## 2. Cell-level crack classification

The Training Settings tab exposes classifier iterations, learning rate, L2 regularization, validation fraction, decision-threshold search and class balancing.

Use Positive/Negative cell annotations in **Advanced → Annotations**. These reviewed crops are stored in `Working Folder/data/training/feedback/` and are separate from YOLO cell boxes in `Working Folder/data/training/cell_annotations/`. Train the crack classifier from the Training Centre's independent **Crack classification** panel. The crack model is not a YOLO model in this release.

## Data safety

- Annotation source images and labels are local runtime data and are not bundled in the release ZIP.
- The current active ONNX model is `Working Folder/data/models/cell.onnx`; the previous version is `Working Folder/data/models/cell.previous.onnx` and can be restored from the Training Centre. The UI reports classical fallback when `onnxruntime-node` is missing instead of incorrectly reporting YOLO as active.
- The reusable PyTorch fine-tuning checkpoint is `Working Folder/data/training/models/cell-best.pt`, with its previous copy at `Working Folder/data/training/models/cell-best.previous.pt`. Preserve `Working Folder/data/` when moving to a later version to continue training from learned weights.
- Epoch analytics are persisted in the job state and can be exported as CSV. Full raw stdout/stderr is retained under `Working Folder/data/training/cell_yolo/logs/`.
- Training candidates are not activated unless validation passes the configured mAP50 gate.


## PyTorch CPU package installation (v1.4.5.7)

The app installs the CPU runtime in separate, protected phases:

1. `torch` and `torchvision` are installed as binary wheels only from `https://download.pytorch.org/whl/cpu`, with `--no-deps --only-binary=:all:`. No dependency or source-build lookup is performed against the PyTorch-only index.
2. `training/install_package_dependencies.py` reads the installed wheels' `Requires-Dist` metadata and resolves those dependencies from `https://pypi.org/simple`. It pins the installed `torch`/`torchvision` versions in a temporary constraints file, so dependency resolution cannot replace the selected CPU wheels.
3. The Ultralytics wheel is installed from PyPI with `--no-deps`; the same metadata-driven helper then resolves its dependencies from PyPI while preserving the exact installed CPU runtime versions.

Pip index environment overrides (`PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, and `PIP_FIND_LINKS`) are removed for these operations. `pip check` and real imports of `torch`, `torchvision`, and `ultralytics` must succeed before the runtime is marked ready. Healthy environments are reused on later setup checks. This addresses the v1.4.5.1 failure where a command-line PyTorch-only index prevented pip's isolated build step from finding `flit_core`, and also prevents a later PyPI dependency resolution from silently replacing CPU wheels.
