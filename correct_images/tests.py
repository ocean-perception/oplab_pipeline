import os
import unittest
import imageio
import tempfile
import numpy as np
from correct_images.colour_correct import mean_per_pixel, std_per_pixel_parallel
from correct_images.utilities import generate_filelist, get_filenames, get_outpath, validate_filenames

# REQUIREMENTS
# SUPPORT 8 and 16bit
# SUPPORT 2 / 3 cameras
# SUPPORT Black and white cameras, Debayer, AND Colour
# Normally tif, could be jpeg

# Specify all of this in yaml file ^^^

# Unspecified: set some defaults and print a warning

def get_empty_test_directory():
    test_directory = os.path.join(tempfile.gettempdir(), "module_testing_correct_images")
    # Delete all files in test directory, in case there are any
    if os.path.isdir(test_directory):
        for the_file in os.listdir(test_directory):
            file_path = os.path.join(test_directory, the_file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.mkdir(test_directory)
    return test_directory


# colour_correct.py
class TestMeanPerPixelMethod(unittest.TestCase):
    def test_empty_filenames(self):
        emptyList = list()
        do_debayer = False
        filter_pattern = ''
        self.assertRaises(IndexError, mean_per_pixel, "", emptyList, do_debayer, filter_pattern)

    def test_not_filepath(self):
        do_debayer = False
        filter_pattern = ''
        self.assertRaises(IOError, mean_per_pixel, "", "not a file, oops", do_debayer, filter_pattern)

    def test_example_file(self):
        example1 = np.array([[0, 1], [2, 3]], dtype='uint8')
        example2 = np.array([[6, 8], [7, 0]], dtype='uint8')
        examplemean = np.array([[[3], [4.5]], [[4.5], [1.5]]])
        test_directory = get_empty_test_directory()
        imageio.imsave(os.path.join(test_directory, "tmppm_example1.tif"), example1)
        imageio.imsave(os.path.join(test_directory, "tmppm_example2.tif"), example2)
        do_debayer = False
        filter_pattern = ''
        self.assertTrue(np.array_equal(mean_per_pixel(test_directory, ["tmppm_example1.tif", "tmppm_example2.tif"], do_debayer, filter_pattern), examplemean))


class TestStdPerPixelParallelMethod(unittest.TestCase):

    def test_empty_path(self):
        do_debayer = False
        filter_pattern = ''
        arguments = [["", ["nonexsting_file1.tif", "nonexsting_file2.tif"], "", do_debayer, filter_pattern]]
        #arguments = []
        #arguments.append(["", ["ex1.tif", "ex2.tif"], "", do_debayer, filter_pattern])
        self.assertRaises(FileNotFoundError, std_per_pixel_parallel, arguments, 1, 2)

    def test_empty_list(self):
        emptyList = list()
        do_debayer = False
        filter_pattern = ''
        test_directory = get_empty_test_directory()
        args = [[test_directory, emptyList, "", do_debayer, filter_pattern]]
        self.assertRaises(IndexError, std_per_pixel_parallel, args, 1, 2)

    def test_example_file(self):
        example1 = np.array([[0, 1], [2, 3]], dtype='uint8')
        example2 = np.array([[6, 8], [7, 0]], dtype='uint8')
        examplemean = np.array([[[3], [4.5]], [[4.5], [1.5]]])
        examplestd = np.array([[[3], [3.5]], [[2.5], [1.5]]])
        test_directory = get_empty_test_directory()
        imageio.imsave(os.path.join(test_directory, "tsppm_example1.tif"), example1)
        imageio.imsave(os.path.join(test_directory, "tsppm_example2.tif"), example2)
        do_debayer = False
        filter_pattern = ''
        args = [[test_directory, ["tsppm_example1.tif", "tsppm_example2.tif"], examplemean, do_debayer, filter_pattern]]
        self.assertTrue(np.array_equal(std_per_pixel_parallel(args, 1, 2), examplestd))

# utilities.py
class TestGenerateFilelistMethod(unittest.TestCase):
    def test_filepath_not_valid(self):
        self.assertRaises(TypeError, generate_filelist, False)
        self.assertRaises(ValueError, generate_filelist, "/tmp/definitely_not_a_real_directory_ladidadida")

    def test_filepath_valid(self):
        test_directory = get_empty_test_directory()
        open(os.path.join(test_directory, "img1.tif"), 'w')   # Create 2 files so that there are files to test genearte_filelist()
        open(os.path.join(test_directory, "img2.tif"), 'w')
        filelist = generate_filelist(test_directory)
        filelist = open(os.path.join(test_directory, filelist), 'r')
        self.assertEqual(filelist.readlines(), ["img1.tif\n", "img2.tif"])


class TestGetFilenames(unittest.TestCase):
    def test_filepath_not_valid(self):
        self.assertRaises(TypeError, get_filenames, False, "blah")
        self.assertRaises(TypeError, get_filenames, "blah", False)
        self.assertRaises(FileNotFoundError, get_filenames, "/tmp/blah", "blah")
        self.assertRaises(TypeError, get_filenames, False, False)

    def test_filepath_valid(self):
        test_directory = get_empty_test_directory()
        file = open(os.path.join(test_directory, "flist.txt"), 'w')
        file.write("example2.tif\nexample1.tif")
        file.close()
        self.assertEqual(get_filenames(test_directory, "flist.txt"), ["example2.tif", "example1.tif"])


class TestValidateFilenames(unittest.TestCase):
    def test_invalid_filenames(self):
        self.assertRaises(ValueError, validate_filenames, '/tmp/', [""])
        self.assertRaises(ValueError, validate_filenames, '/tmp/', ["definitely_notafile_hahahahaaaa"])
        self.assertRaises(TypeError, validate_filenames, '/tmp/', [True])

    def test_valid_filenames(self):
        test_directory = get_empty_test_directory()
        file1 = open(os.path.join(test_directory, "real_file.txt"), 'w')
        file2 = open(os.path.join(test_directory, "so_real.txt"), 'w')
        file1.close()
        file2.close()
        self.assertTrue(validate_filenames(test_directory, ["real_file.txt", "so_real.txt"]))


class TestGetOutpathMethod(unittest.TestCase):

    def test_filepath_not_string(self):
        self.assertRaises(TypeError, get_outpath, 0, "dummy_method", "10", "1")
        self.assertRaises(TypeError, get_outpath, 0.5, "dummy_method", "10", "1")

    def test_empty_filepath(self):
        self.assertRaises(ValueError, get_outpath, "", "dummy_method", "10", "1")

    def test_filepath_with_spaces(self):
        self.assertRaises(ValueError, get_outpath, "example with spaces", "dummy_method", "10", "1")

    def test_valid_filepath(self):
        self.assertEqual(get_outpath("/tmp/example/raw/", "dummy_method", "10", "1"),       os.path.join("tmp", "example", "processed", "dummy_method", "mean_10_std_1"))
        self.assertEqual(get_outpath("/tmp/example/raw/raw/", "dummy_method", "10", "1"),   os.path.join("tmp", "example", "raw", "processed", "dummy_method", "mean_10_std_1"))
        self.assertEqual(get_outpath("./tmp/example/raw/", "dummy_method", "10", "1"),      os.path.join(".", "tmp", "example", "processed", "dummy_method", "mean_10_std_1"))
        self.assertEqual(get_outpath("/tmp/example/raw/straw/", "dummy_method", "10", "1"), os.path.join("tmp", "example", "processed", "straw", "dummy_method", "mean_10_std_1"))
        self.assertEqual(get_outpath("/tmp/example/straw/raw/", "dummy_method", "10", "1"), os.path.join("tmp", "example", "straw", "processed", "dummy_method", "mean_10_std_1"))
        self.assertEqual(get_outpath("D:\\tmp\\example\\raw\\", "dummy_method", "10", "1"), os.path.join("D:", "tmp", "example", "processed", "dummy_method", "mean_10_std_1"))
        self.assertEqual(get_outpath(".\\tmp\\example\\raw\\", "dummy_method", "10", "1"),  os.path.join(".", "tmp", "example", "processed", "dummy_method", "mean_10_std_1"))
        self.assertEqual(get_outpath("D:\\tmp/example\\raw/", "dummy_method", "10", "1"),   os.path.join("D:", "tmp", "example", "processed", "dummy_method", "mean_10_std_1"))

    def test_invalid_filepath(self):
        self.assertRaises(ValueError, get_outpath, "/tmp/example/notraw/", "dummy_method", "10", "1")


if __name__ == '__main__':
    unittest.main()
