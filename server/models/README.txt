PV CrackVision model storage

The writable model directory lives in your data folder (chosen under Data & Models; default Working Folder/data/models/)
Active inference model: cell.onnx
Previous inference model: cell.previous.onnx
User-trained crack classifier: cellCrackModel.user.json

The reusable PyTorch training checkpoint is stored at: Working Folder/data/training/models/cell-best.pt
Training annotations, feedback, YOLO datasets, runs, logs, Python runtime and checkpoints are stored below Working Folder/data/training/ and Working Folder/data/runtime/.
The first launch of v1.4.5.7 imports existing version-local artifacts from sibling PV_CrackVision release folders. Source files are copied/merged; old folders are not deleted.
