"""Unit tests for train_crack_anomaly.py: pure helpers always; the torch scorer + ONNX export parity only when torch is installed."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from training import train_crack_anomaly as t


def cell(img, label):
    return {"imageFile": img, "label": label, "left": 0, "top": 0, "width": 40, "height": 40, "category": "spider_web_crack"}


class Helpers(unittest.TestCase):
    def test_coreset_size_determinism_and_coverage(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(0, 0.01, (500, 8)), rng.normal(5, 0.01, (3, 8))]).astype("float32")  # 3 rare outlier patches
        a, b = t.greedy_coreset(data, 20, seed=1), t.greedy_coreset(data, 20, seed=1)
        self.assertEqual(a.shape, (20, 8))
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a.mean(1) > 2).any())  # k-center keeps the rare patch that random sampling would usually drop
        self.assertIs(t.greedy_coreset(data, 10000, seed=1), data)

    def test_auroc(self):
        self.assertEqual(t.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(t.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertIsNone(t.auroc([0, 0], [0.1, 0.2]))
        # v1.4.17.1: ties count 0.5 whatever the order they arrive in
        self.assertEqual(t.auroc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(t.auroc([1, 0], [0.5, 0.5]), 0.5)
        self.assertEqual(t.auroc([0, 0, 1, 1], [1, 1, 1, 1]), 0.5)

    def test_threshold_modes(self):
        clean = np.linspace(0.0, 1.0, 1001)
        thr, mode = t.choose_threshold(clean, [], 0.99, 0.95, 100)
        self.assertEqual(mode, "clean-quantile")
        self.assertAlmostEqual(thr, 0.99, places=3)
        thr, mode = t.choose_threshold(clean[:200], [5.0, 6.0], 0.99, 0.95, 100)  # < 3 cracks: still clean-only
        self.assertEqual(mode, "clean-quantile")
        thr, mode = t.choose_threshold(np.linspace(0, 1, 200), [3.0, 4.0, 5.0], 0.99, 0.95, 100)
        self.assertNotIn(thr, (3.0, 4.0, 5.0))  # v1.4.17.1: never exactly on a validation score
        self.assertEqual(mode, "precision")
        self.assertTrue(1.0 < thr <= 3.0)  # above every clean score, at/below the lowest crack

    def test_split_falls_back_to_clean_images_when_cracks_are_scarce(self):
        grouped = {f"i{n}.png": [cell(f"i{n}.png", 0)] for n in range(5)}
        train, val = t.split_images(grouped, 0.2, seed=3)
        self.assertEqual(len(val), 1)
        self.assertEqual(sorted(train + val), sorted(grouped))
        with self.assertRaises(ValueError):
            t.split_images({"a.png": [cell("a.png", 0)], "b.png": [cell("b.png", 1)]}, 0.2, seed=3)

    def test_parse_requires_multiple_of_32(self):
        base = ["--feedback-dir", "f", "--work", "w", "--state-file", "s", "--activate", "a", "--meta-activate", "m"]
        self.assertEqual(t.parse(base).imgsz, 224)
        with self.assertRaises(SystemExit):
            t.parse(base + ["--imgsz", "100"])


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - not installed, or a broken/CUDA-only install: the scorer test needs a working torch
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class Scorer(unittest.TestCase):
    @staticmethod
    def fake_backbone():
        torch.manual_seed(0)
        strides, chans, layers = [2, 2, 1, 2, 1, 2, 1], [8, 8, 8, 16, 16, 16, 16], []
        prev = 3
        for s, c in zip(strides, chans):  # same strides as yolo11-cls layers 0..6: layer 4 = stride 8, layer 6 = stride 16
            layers.append(nn.Sequential(nn.Conv2d(prev, c, 3, s, 1), nn.ReLU()))
            prev = c
        return SimpleNamespace(model=SimpleNamespace(model=nn.Sequential(*layers)))

    def test_scores_anomaly_higher_and_onnx_matches_torch(self):
        backbone, size = self.fake_backbone(), 64
        clean = torch.rand(6, 3, size, size) * 0.2 + 0.4
        with torch.no_grad():
            probe = t.build_scorer(backbone, np.zeros((1, 1), "float32"), 2)
            bank = probe.patches(clean).flatten(0, 1).numpy()
            scorer = t.build_scorer(backbone, bank[::3], 2)
            odd = clean.clone(); odd[:, :, 20:40, 20:40] = 1.0  # a bright patch no clean crop has
            normal, anomalous = scorer.raw(clean).mean().item(), scorer.raw(odd).mean().item()
            self.assertGreater(anomalous, normal)
            scorer = t.build_scorer(backbone, bank[::3], 2, threshold=(normal + anomalous) / 2, scale=max(1e-3, (anomalous - normal) / 4))
            expected = scorer(odd)
        self.assertEqual(tuple(expected.shape), (6, 2))
        np.testing.assert_allclose(expected.sum(1).numpy(), 1.0, atol=1e-5)
        import onnxruntime as ort
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.onnx"
            t.export_onnx(scorer, size, path)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            out = session.run(None, {"images": odd.numpy()[:3]})[0]  # different batch size than the export dummy: dynamic axis
        np.testing.assert_allclose(out, expected[:3].numpy(), atol=1e-3)
        self.assertGreater(out[:, 1].min(), 0.5)


if __name__ == "__main__":
    unittest.main()
