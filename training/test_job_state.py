"""Small smoke tests for the detached worker's durable state serialization."""
import os
import pathlib
import runpy
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock


WORKER = runpy.run_path(str(pathlib.Path(__file__).with_name("train_cell_yolo.py")))


class TrainingWorkerStateTests(unittest.TestCase):
    def test_progress_and_logs_survive_worker_state_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = pathlib.Path(directory) / "job-state.json"
            WORKER["write_state"](state_path, status="training", progress=3, message="Starting", log_line="Worker started")
            WORKER["write_state"](state_path, status="training", progress=52, message="Epoch 5/10", log_line="Epoch 5/10 completed")
            state = WORKER["read_state"](state_path)
            self.assertEqual(state["status"], "training")
            self.assertEqual(state["progress"], 52)
            self.assertEqual(state["log"], ["Worker started", "Epoch 5/10 completed"])

    def test_fixed_epoch_mode_disables_early_stop_before_requested_epoch(self):
        for epochs in (10, 38, 100, 200, 500):
            patience = WORKER["resolve_early_stopping_patience"](epochs, 0)
            self.assertEqual(patience, 0)
        self.assertEqual(WORKER["resolve_early_stopping_patience"](200, 12), 12)
        self.assertEqual(WORKER["resolve_early_stopping_patience"](10, 100), 10)

    def test_yolo_training_kwargs_keep_requested_epochs_and_expose_freeze_and_optimizer(self):
        args = SimpleNamespace(
            data="dataset.yaml", epochs=200, imgsz=960, batch=2, project="runs", run_name="test",
            patience=0, freeze=8, optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.9, weight_decay=0.0005,
            warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, cos_lr=False,
            close_mosaic=10, mosaic=1.0, mixup=0.1, degrees=2.0, translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            box_loss=7.5, cls_loss=0.5, dfl_loss=1.5, multi_scale=0.0, seed=7,
            deterministic=True, cache=False, pretrained=True, save_period=10,
        )
        kwargs = WORKER["build_training_kwargs"](args)
        self.assertEqual(kwargs["epochs"], 200)
        self.assertEqual(kwargs["patience"], 0)
        self.assertNotIn("time", kwargs)
        self.assertEqual(kwargs["freeze"], 8)
        self.assertEqual(kwargs["optimizer"], "AdamW")
        self.assertEqual(kwargs["imgsz"], 960)
        self.assertEqual(kwargs["device"], "cpu")

    def test_explicit_positive_patience_is_forwarded_to_ultralytics(self):
        args = SimpleNamespace(
            data="dataset.yaml", epochs=200, patience=15, imgsz=960, batch=2, project="runs", run_name="test",
            freeze=0, optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.9, weight_decay=0.0005,
            warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, cos_lr=False, close_mosaic=10,
            mosaic=1.0, mixup=0.0, degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, box_loss=7.5, cls_loss=0.5, dfl_loss=1.5,
            multi_scale=0.0, seed=0, deterministic=True, cache=False, pretrained=True, save_period=10,
        )
        self.assertEqual(WORKER["build_training_kwargs"](args)["patience"], 15)

    def test_gpu_aware_defaults_and_overrides(self):
        resolve_workers, resolve_amp = WORKER["resolve_workers"], WORKER["resolve_amp"]
        # CPU / MPS never use loader processes or AMP.
        self.assertEqual(resolve_workers(device="cpu", cpu_count=32, ram_gb=64, windows=True), 0)
        self.assertFalse(resolve_amp(device="cpu"))
        # GPU on a large Windows machine: several workers, capped below the Linux ceiling.
        self.assertEqual(resolve_workers(device="0", cpu_count=32, ram_gb=64, windows=True), 6)
        self.assertEqual(resolve_workers(device="0", cpu_count=32, ram_gb=64, windows=False), 8)
        self.assertEqual(resolve_workers(device="0", cpu_count=4, ram_gb=64, windows=True), 2)
        # Limited or unknown memory stays conservative.
        self.assertEqual(resolve_workers(device="0", cpu_count=16, ram_gb=16, windows=True), 2)
        self.assertEqual(resolve_workers(device="0", cpu_count=16, ram_gb=8, windows=True), 0)
        self.assertTrue(resolve_amp(device="0"))
        with mock.patch.dict(os.environ, {"PV_YOLO_WORKERS": "3", "PV_YOLO_AMP": "0"}):
            self.assertEqual(resolve_workers(device="0", cpu_count=32, ram_gb=64, windows=True), 3)
            self.assertFalse(resolve_amp(device="0"))
        self.assertEqual(WORKER["MAX_DET"], 1000)

    def test_plot_labels_are_hidden_without_changing_boxes(self):
        calls = []
        class FakeAnnotator:
            def box_label(self, box, label="", color=(1, 2, 3), txt_color=(4, 5, 6), rotated=False):
                calls.append((box, label, color))
        fake_plotting = types.ModuleType("ultralytics.utils.plotting"); fake_plotting.Annotator = FakeAnnotator
        modules = {"ultralytics": types.ModuleType("ultralytics"), "ultralytics.utils": types.ModuleType("ultralytics.utils"), "ultralytics.utils.plotting": fake_plotting}
        with mock.patch.dict(sys.modules, modules):
            self.assertTrue(WORKER["hide_plot_labels"]())
            self.assertTrue(WORKER["hide_plot_labels"]())  # idempotent: not wrapped twice
            FakeAnnotator().box_label([0, 0, 5, 5], "pv_cell", color=(9, 9, 9))
            FakeAnnotator().box_label([1, 1, 6, 6], label="pv_cell 0.9")
        self.assertEqual(calls[0], ([0, 0, 5, 5], "", (9, 9, 9)))
        self.assertEqual(calls[1][1], "")
        with mock.patch.dict(os.environ, {"PV_YOLO_PLOT_LABELS": "1"}):
            self.assertFalse(WORKER["hide_plot_labels"]())

    def test_onnx_gate_allows_small_export_tolerance_only(self):
        floor = WORKER["onnx_gate_floor"]
        self.assertAlmostEqual(floor(0.30), 0.28)
        self.assertGreaterEqual(0.2984, floor(0.30))  # the reported real-world case: must now pass
        self.assertLess(0.25, floor(0.30))            # a genuinely poor export is still rejected
        self.assertEqual(floor(0.01), 0.0)            # never negative

    def test_ultralytics_metrics_dict_is_read_as_a_dict(self):
        trainer = type("Trainer", (), {"metrics": {"metrics/mAP50(B)": 0.61, "metrics/precision(B)": 0.72}})()
        metrics = WORKER["extract_validation_metrics"](trainer)
        self.assertEqual(metrics["metrics/mAP50(B)"], 0.61)


    def test_each_epoch_metric_row_contains_precision_recall_map_and_losses(self):
        trainer = SimpleNamespace(
            metrics={
                "metrics/precision(B)": 0.81,
                "metrics/recall(B)": 0.74,
                "metrics/mAP50(B)": 0.78,
                "metrics/mAP50-95(B)": 0.46,
            },
            tloss=[0.42, 0.19, 0.08],
            loss_names=["box_loss", "cls_loss", "dfl_loss"],
            save_dir="/path/that/does/not/exist",
        )
        row = WORKER["build_epoch_metric_row"](trainer, 7, "2026-11-06T12:00:00+00:00")
        self.assertEqual(row["epoch"], 7)
        self.assertEqual(row["precision"], 0.81)
        self.assertEqual(row["recall"], 0.74)
        self.assertEqual(row["map50"], 0.78)
        self.assertEqual(row["map50_95"], 0.46)
        self.assertEqual(row["box_loss"], 0.42)
        self.assertEqual(row["cls_loss"], 0.19)
        self.assertEqual(row["dfl_loss"], 0.08)


    def test_epoch_row_reads_validation_losses_from_ultralytics_results_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            results = pathlib.Path(directory) / "results.csv"
            results.write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss\n"
                "0,0.80,0.70,0.75,0.42,0.40,0.20,0.08,0.45,0.23,0.09\n",
                encoding="utf-8",
            )
            trainer = SimpleNamespace(metrics={}, tloss=None, loss_names=[], save_dir=directory)
            row = WORKER["build_epoch_metric_row"](trainer, 1)
            self.assertEqual(row["precision"], 0.80)
            self.assertEqual(row["val_box_loss"], 0.45)
            self.assertEqual(row["val_cls_loss"], 0.23)
            self.assertEqual(row["val_dfl_loss"], 0.09)

    def test_results_csv_is_a_durable_fallback_for_each_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            results = root / "results.csv"
            results.write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss\n"
                "1,0.31,0.42,0.18,0.09,1.1,0.8,0.4,1.2,0.9,0.5\n"
                "2,0.51,0.62,0.38,0.19,0.9,0.6,0.3,1.0,0.7,0.4\n",
                encoding="utf-8",
            )
            rows = WORKER["read_results_csv_rows"](results)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["epoch"], 1)
            self.assertEqual(rows[1]["map50"], 0.38)
            self.assertEqual(rows[1]["val_box_loss"], 1.0)

    def test_epoch_metrics_are_persisted_and_replaced_by_epoch_number(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = pathlib.Path(directory) / "job-state.json"
            first = {"epoch": 1, "precision": 0.5, "recall": 0.4}
            corrected = {"epoch": 1, "precision": 0.6, "recall": 0.45}
            second = {"epoch": 2, "precision": 0.65, "recall": 0.5}
            WORKER["write_state"](state_path, status="training", epoch_metric=first)
            WORKER["write_state"](state_path, status="training", epoch_metric=corrected)
            WORKER["write_state"](state_path, status="training", epoch_metric=second)
            rows = WORKER["read_state"](state_path)["epochMetrics"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["precision"], 0.6)
            self.assertEqual(rows[1]["epoch"], 2)


    def test_reusable_checkpoint_is_saved_and_previous_checkpoint_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "new-best.pt"
            destination = root / "training" / "models" / "cell-best.pt"
            source.write_bytes(b"new-checkpoint")
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"old-checkpoint")
            saved = WORKER["persist_reusable_checkpoint"](source, destination)
            previous = destination.with_name("cell-best.previous.pt")
            self.assertEqual(saved, str(destination))
            self.assertEqual(destination.read_bytes(), b"new-checkpoint")
            self.assertEqual(previous.read_bytes(), b"old-checkpoint")

    def test_metric_object_results_dict_is_supported(self):
        metric_object = type("Metric", (), {"results_dict": {"metrics/mAP50(B)": 0.48}})()
        trainer = type("Trainer", (), {"metrics": metric_object})()
        metrics = WORKER["extract_validation_metrics"](trainer)
        self.assertEqual(metrics["metrics/mAP50(B)"], 0.48)


if __name__ == "__main__":
    unittest.main()
