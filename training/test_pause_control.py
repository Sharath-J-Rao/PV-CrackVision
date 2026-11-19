"""Tests for the cooperative pause helper of the YOLO worker (v1.4.10)."""
import pathlib
import runpy
import tempfile
import unittest

WORKER = runpy.run_path(str(pathlib.Path(__file__).with_name("train_cell_yolo.py")))


class PauseControlTests(unittest.TestCase):
    def test_no_pause_file_returns_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            state = pathlib.Path(directory) / "job-state.json"
            self.assertFalse(WORKER["wait_while_paused"](state, WORKER["control_path"](state), sleep=lambda _s: None))

    def test_pauses_until_control_file_removed_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            state = pathlib.Path(directory) / "job-state.json"
            control = WORKER["control_path"](state)
            control.write_text("pause")
            ticks = []

            def fake_sleep(_seconds):
                ticks.append(WORKER["read_state"](state).get("status"))
                control.unlink()

            self.assertTrue(WORKER["wait_while_paused"](state, control, sleep=fake_sleep))
            self.assertEqual(ticks, ["paused"])
            self.assertEqual(WORKER["read_state"](state)["status"], "training")


if __name__ == "__main__":
    unittest.main()
