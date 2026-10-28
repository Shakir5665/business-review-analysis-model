"""
Test Suite: Training History Persistence, Resume Capability, and Curve Generation
"""

import json
from pathlib import Path
import tempfile
import unittest

from src.visualization.plot_curves import plot_training_curves, load_history


class TestHistoryAndCurves(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.history_file = self.temp_path / "history.json"
        self.plots_dir = self.temp_path / "plots"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_history_schema_and_resume_simulation(self):
        # 1. Simulate saving 5 epochs
        mock_history = {
            "epochs": [1, 2, 3, 4, 5],
            "train_loss_total": [0.85, 0.62, 0.48, 0.41, 0.38],
            "train_loss_sentiment": [0.45, 0.32, 0.25, 0.21, 0.19],
            "train_loss_aspect": [0.40, 0.30, 0.23, 0.20, 0.19],
            "val_loss_total": [0.80, 0.58, 0.49, 0.45, 0.46],
            "val_loss_sentiment": [0.42, 0.30, 0.26, 0.23, 0.24],
            "val_loss_aspect": [0.38, 0.28, 0.23, 0.22, 0.22],
            "val_sentiment_acc": [0.72, 0.81, 0.86, 0.89, 0.89],
            "val_sentiment_f1": [0.70, 0.80, 0.85, 0.88, 0.88],
            "val_aspect_micro_f1": [0.68, 0.77, 0.82, 0.85, 0.84],
            "val_aspect_macro_f1": [0.65, 0.74, 0.79, 0.82, 0.81],
            "is_best_epoch": [True, True, True, True, False],
            "best_val_loss_so_far": [0.80, 0.58, 0.49, 0.45, 0.45],
            "early_stopping_counter": [0, 0, 0, 0, 1],
            "early_stopping_triggered": False,
            "early_stopped_at_epoch": None,
            "best_epoch": 4,
            "best_val_loss": 0.45
        }

        with open(self.history_file, "w") as f:
            json.dump(mock_history, f, indent=2)

        # 2. Verify load_history
        loaded = load_history(self.history_file)
        self.assertEqual(len(loaded["epochs"]), 5)
        self.assertEqual(loaded["best_epoch"], 4)
        self.assertEqual(loaded["best_val_loss"], 0.45)

        # 3. Simulate resume at Epoch 6 and adding Epoch 6
        epoch_6 = 6
        loaded["epochs"].append(epoch_6)
        loaded["train_loss_total"].append(0.35)
        loaded["train_loss_sentiment"].append(0.18)
        loaded["train_loss_aspect"].append(0.17)
        loaded["val_loss_total"].append(0.44) # new best!
        loaded["val_loss_sentiment"].append(0.22)
        loaded["val_loss_aspect"].append(0.22)
        loaded["val_sentiment_acc"].append(0.91)
        loaded["val_sentiment_f1"].append(0.90)
        loaded["val_aspect_micro_f1"].append(0.87)
        loaded["val_aspect_macro_f1"].append(0.84)
        loaded["is_best_epoch"].append(True)
        loaded["best_val_loss_so_far"].append(0.44)
        loaded["early_stopping_counter"].append(0)
        loaded["best_epoch"] = 6
        loaded["best_val_loss"] = 0.44

        with open(self.history_file, "w") as f:
            json.dump(loaded, f, indent=2)

        # 4. Generate curves and verify image created
        out_img = plot_training_curves(history=self.history_file, save_dir=self.plots_dir)
        self.assertTrue(out_img.exists())
        self.assertGreater(out_img.stat().st_size, 1000)

    def test_early_stopping_curve_rendering(self):
        early_stop_history = {
            "epochs": [1, 2, 3, 4, 5, 6, 7],
            "train_loss_total": [0.85, 0.62, 0.48, 0.41, 0.38, 0.35, 0.33],
            "train_loss_sentiment": [0.45, 0.32, 0.25, 0.21, 0.19, 0.18, 0.17],
            "train_loss_aspect": [0.40, 0.30, 0.23, 0.20, 0.19, 0.17, 0.16],
            "val_loss_total": [0.80, 0.58, 0.49, 0.45, 0.46, 0.47, 0.48],
            "val_loss_sentiment": [0.42, 0.30, 0.26, 0.23, 0.24, 0.25, 0.26],
            "val_loss_aspect": [0.38, 0.28, 0.23, 0.22, 0.22, 0.22, 0.22],
            "val_sentiment_acc": [0.72, 0.81, 0.86, 0.89, 0.89, 0.88, 0.88],
            "val_sentiment_f1": [0.70, 0.80, 0.85, 0.88, 0.88, 0.87, 0.87],
            "val_aspect_micro_f1": [0.68, 0.77, 0.82, 0.85, 0.84, 0.84, 0.83],
            "val_aspect_macro_f1": [0.65, 0.74, 0.79, 0.82, 0.81, 0.81, 0.80],
            "is_best_epoch": [True, True, True, True, False, False, False],
            "best_val_loss_so_far": [0.80, 0.58, 0.49, 0.45, 0.45, 0.45, 0.45],
            "early_stopping_counter": [0, 0, 0, 0, 1, 2, 3],
            "early_stopping_triggered": True,
            "early_stopped_at_epoch": 7,
            "best_epoch": 4,
            "best_val_loss": 0.45
        }

        with open(self.history_file, "w") as f:
            json.dump(early_stop_history, f, indent=2)

        out_img = plot_training_curves(history=self.history_file, save_dir=self.plots_dir)
        self.assertTrue(out_img.exists())
        self.assertGreater(out_img.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
