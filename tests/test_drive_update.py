import os
import shutil
import tempfile
import unittest
from pathlib import Path

from drive_update import ensure_checkpoint_structure, copy_file, sync_to_gdrive, compute_md5


class TestDriveUpdate(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.local_dir = self.test_dir / "checkpoints"
        self.gdrive_ckpt_dir = self.test_dir / "gdrive_storage" / "checkpoints"

        self.local_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_ensure_checkpoint_structure_creation(self):
        self.assertFalse(self.gdrive_ckpt_dir.exists())
        status = ensure_checkpoint_structure(self.gdrive_ckpt_dir)
        
        self.assertEqual(status, "CREATED")
        self.assertTrue(self.gdrive_ckpt_dir.exists())

        # Check that calling again marks it as EXISTS
        status2 = ensure_checkpoint_structure(self.gdrive_ckpt_dir)
        self.assertEqual(status2, "EXISTS")

    def test_copy_file_and_force(self):
        src_file = self.local_dir / "best_model.pt"
        src_file.write_text("model weights version 1")
        dst_file = self.gdrive_ckpt_dir / "best_model.pt"

        # Initial copy
        status, msg = copy_file(src_file, dst_file, force=False)
        self.assertEqual(status, "COPIED")
        self.assertTrue(dst_file.exists())
        self.assertEqual(dst_file.read_text(), "model weights version 1")

        # Second copy without force -> should SKIP
        status, msg = copy_file(src_file, dst_file, force=False)
        self.assertEqual(status, "SKIPPED")

        # Third copy WITH force -> should COPY
        status, msg = copy_file(src_file, dst_file, force=True)
        self.assertEqual(status, "COPIED")

    def test_sync_to_gdrive_checkpoints(self):
        # Create test checkpoint files
        (self.local_dir / "best_model.pt").write_text("best_model_data")
        (self.local_dir / "latest_checkpoint.pt").write_text("latest_checkpoint_data")
        (self.local_dir / "history.json").write_text('{"epochs": [1, 2]}')

        results = sync_to_gdrive(
            local_checkpoint_dir=str(self.local_dir),
            gdrive_checkpoint_dir=str(self.gdrive_ckpt_dir),
            force=True
        )

        self.assertTrue((self.gdrive_ckpt_dir / "best_model.pt").exists())
        self.assertTrue((self.gdrive_ckpt_dir / "latest_checkpoint.pt").exists())
        self.assertTrue((self.gdrive_ckpt_dir / "history.json").exists())


if __name__ == "__main__":
    unittest.main()
