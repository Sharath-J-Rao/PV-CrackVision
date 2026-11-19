"""Regression: the anomaly trainer must import its sibling module even when the script folder is NOT on sys.path (embedded Python with a ._pth file)."""
import subprocess, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class EmbeddedImport(unittest.TestCase):
    def test_script_runs_without_script_dir_on_path(self):
        # -I (isolated) reproduces the ._pth behaviour: no script directory, no PYTHONPATH, no cwd.
        out = subprocess.run([sys.executable, "-I", str(HERE / "train_crack_anomaly.py"), "--help"], cwd=str(HERE.parent.parent), capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--memory-bank", out.stdout)


if __name__ == "__main__":
    unittest.main()
