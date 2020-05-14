import unittest
from pathlib import Path
from oplab import get_processed_folders
#from oplab import get_processed_folders
from oplab.folder_structure import get_processed_folders # noqa


class TestFolderStructure(unittest.TestCase):
    def test_get_processed_folders_single(self):
        path = Path(__file__).parents[0]
        path_raw = path / 'raw/test'
        path_processed = path / 'processed/test'
        self.assertEqual(path_processed, get_processed_folders(path_raw))
        
    def test_get_processed_folders_list(self):
        path = Path(__file__).parents[0]
        path_raw = []
        path_raw.append(path / 'raw/test0')
        path_raw.append(path / 'raw/test1/')
        path_processed = []
        path_processed.append(path / 'processed/test0')
        path_processed.append(path / 'processed/test1/')
        self.assertEqual(path_raw, get_raw_folders(path_processed))
        
    def test_get_processed_folders_list(self):
        path = Path(__file__).parents[0]
        path_raw = []
        path_raw.append(path / 'raw/test0')
        path_raw.append(path / 'raw/test1')
        path_processed = []
        path_processed.append(path / 'processed/test0')
        path_processed.append(path / 'processed/test1')
        self.assertEqual(path_processed, get_processed_folders(path_raw))