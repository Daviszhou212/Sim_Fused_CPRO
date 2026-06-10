import tempfile
import unittest
from pathlib import Path


class ArtifactPathsTest(unittest.TestCase):
    def test_output_path_refuses_existing_file_by_default(self):
        from MultiCell_MIMO.artifact_paths import build_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = build_output_path(tmpdir, "SLDAC", "result.mat")
            Path(path).write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                build_output_path(tmpdir, "SLDAC", "result.mat")

    def test_checkpoint_dir_can_include_run_id(self):
        from MultiCell_MIMO.artifact_paths import build_checkpoint_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            path = build_checkpoint_dir(tmpdir, "SLDAC", "tag", seed=3, run_id="run42")

        self.assertTrue(str(path).endswith(str(Path("SLDAC") / "tag" / "run42" / "seed_3")))


if __name__ == "__main__":
    unittest.main()
