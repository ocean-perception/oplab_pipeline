import unittest
from pathlib import Path
from oplab import get_processed_folders
from oplab import get_raw_folders
from oplab import check_dirs_exist


class TestFolderStructure(unittest.TestCase):
    def test_get_processed_folders_single(self):
        path = Path(__file__).parents[0]
        path_raw = path / "raw/test"
        path_processed = path / "processed/test"
        self.assertEqual(path_processed, get_processed_folders(path_raw))

    def test_get_processed_folders_list(self):
        path = Path(__file__).parents[0]
        path_raw = []
        path_raw.append(path / "raw/test0")
        path_raw.append(path / "raw/test1/")
        path_processed = []
        path_processed.append(path / "processed/test0")
        path_processed.append(path / "processed/test1/")
        self.assertEqual(path_raw, get_raw_folders(path_processed))

    def test_check_dirs_exist_folders(self):
        self.assertTrue(check_dirs_exist(Path(__file__).parents[0]))
        paths = []
        paths.append(Path(__file__).parents[0])
        paths.append(Path(__file__).parents[1])
        self.assertTrue(check_dirs_exist(paths))
        paths.append(Path(__file__).parents[0] / "non-exiting-folder/")
        self.assertFalse(check_dirs_exist(paths))

    def test_check_dirs_exist_files(self):
        self.assertFalse(check_dirs_exist(Path(__file__)))
        paths = []
        paths.append(Path(__file__))
        self.assertFalse(check_dirs_exist(paths))
