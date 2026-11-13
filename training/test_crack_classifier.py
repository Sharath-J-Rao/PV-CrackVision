"""Unit tests for the pure helpers of train_crack_classifier.py (no torch / ultralytics needed)."""
import unittest
import numpy as np
from training import train_crack_classifier as t


def cell(img, label, left=0, top=0, size=40, **extra):
    return {"imageFile": img, "label": label, "left": left, "top": top, "width": size, "height": size, "category": "spider_web_crack", **extra}


class Helpers(unittest.TestCase):
    def test_windows_training_preserves_bounded_dataloader_parallelism(self):
        self.assertEqual(t.resolve_workers(windows=True, cpu_count=32), 4)
        self.assertEqual(t.resolve_workers(windows=True, cpu_count=4), 4)
        self.assertEqual(t.resolve_workers(windows=True, cpu_count=2), 2)

    def test_non_windows_training_keeps_bounded_dataloader_workers(self):
        self.assertEqual(t.resolve_workers(windows=False, cpu_count=32), 4)
        self.assertEqual(t.resolve_workers(windows=False, cpu_count=2), 2)
        self.assertEqual(t.resolve_workers(windows=False, cpu_count=0), 0)

    def test_windows_worker_executable_uses_pythonw_when_available(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python_exe = root / "python.exe"
            pythonw_exe = root / "pythonw.exe"
            python_exe.touch(); pythonw_exe.touch()
            with mock.patch("multiprocessing.set_executable") as set_executable:
                chosen = t.configure_windows_worker_executable(str(python_exe), windows=True)
            self.assertEqual(chosen, str(pythonw_exe))
            set_executable.assert_called_once_with(__import__("os").fsencode(str(pythonw_exe)))
            self.assertIsNone(t.configure_windows_worker_executable(str(python_exe), windows=False))

    def test_buffered_crop_matches_node_rule(self):
        self.assertEqual(t.buffered_crop(1000, 1000, cell("a", 1, 100, 100, 100), 0.03), (97, 97, 203, 203))
        self.assertEqual(t.buffered_crop(150, 150, cell("a", 1, 120, 120, 40), 0.0), (120, 120, 150, 150))  # clamped, no buffer

    def test_split_is_by_image_and_keeps_both_classes_in_validation(self):
        grouped = {f"img{i}": [cell(f"img{i}", i % 2)] for i in range(8)}
        train, val = t.split_by_image(grouped, 0.25, 1)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(train) + len(val), 8)
        self.assertTrue({1, 0} <= {c["label"] for n in val for c in grouped[n]})

    def test_threshold_is_highest_value_reaching_target_recall(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        p = np.array([0.95, 0.9, 0.7, 0.6, 0.65, 0.3, 0.2, 0.1])
        thr, met = t.pick_threshold(y, p, 0.75)
        self.assertTrue(met)
        self.assertEqual(t.metrics_at(y, p, thr)["recall"], 0.75)
        self.assertGreaterEqual(thr, 0.7)  # higher thresholds would drop below the target; lower ones only add false positives

    def test_outer_cell_detection(self):
        mod = dict(moduleLeft=0, moduleTop=0, moduleWidth=600, moduleHeight=400)
        self.assertTrue(t.is_outer_cell(cell("a", 0, 5, 5, 40, **mod)))
        self.assertFalse(t.is_outer_cell(cell("a", 0, 280, 180, 40, **mod)))
        self.assertFalse(t.is_outer_cell(cell("a", 0, 5, 5, 40)))  # no module box -> unknown -> not outer

    def test_tile_origins_cover_the_image_edge(self):
        self.assertEqual(t.tile_origins(500, 1280, 960), [0])
        xs = t.tile_origins(3000, 1280, 960)
        self.assertEqual(xs[-1], 3000 - 1280)
        self.assertTrue(all(b - a <= 960 for a, b in zip(xs, xs[1:])))

    def test_softmax_helper_keeps_probabilities(self):
        probs = t.softmax_if_needed([[0.2, 0.8]])
        self.assertAlmostEqual(float(probs[0][1]), 0.8, places=5)
        logits = t.softmax_if_needed([[0.0, 2.0]])
        self.assertAlmostEqual(float(logits.sum()), 1.0, places=5)


class TelemetryRegression(unittest.TestCase):
    """v1.4.14.4: trainer.tloss is a dict in newer Ultralytics ("'dict' object has no attribute 'sum'")."""
    def test_scalar_loss_accepts_dict_tensor_like_and_garbage(self):
        class Tensorish:
            def detach(self): return self
            def float(self): return self
            def sum(self): return self
            def item(self): return 1.5
        self.assertAlmostEqual(t.scalar_loss({"loss": Tensorish(), "aux": 0.5}), 2.0)
        self.assertAlmostEqual(t.scalar_loss(Tensorish()), 1.5)
        self.assertAlmostEqual(t.scalar_loss(np.array([0.25, 0.75])), 1.0)
        self.assertAlmostEqual(t.scalar_loss([1, 2]), 3.0)
        for bad in (None, {}, "x", float("nan")):
            self.assertIsNone(t.scalar_loss(bad))

    def test_save_dir_prefers_trainer_and_falls_back_to_result(self):
        from types import SimpleNamespace as NS
        self.assertEqual(str(t.trainer_save_dir(NS(trainer=NS(save_dir="/a")), NS())), "/a")
        self.assertEqual(str(t.trainer_save_dir(NS(), NS(save_dir="/b"))), "/b")
        with self.assertRaises(RuntimeError):
            t.trainer_save_dir(NS(), NS())

    def test_epoch_callback_swallows_telemetry_errors(self):
        t.STATE = None
        t.epoch_callback(lambda trainer: 1 / 0)(object())  # must not raise


if __name__ == "__main__":
    unittest.main()
