import unittest
from pathlib import Path
from auv_nav.camera_system import CameraSystem


class TestCameraSystem(unittest.TestCase):
    def testInstantiation(self):
        cs = CameraSystem()
        self.assertIsInstance(cs, CameraSystem, 'CameraSystem class is not instantiable')

    def testAcfrStandard(self):
        root = Path(__file__).parents[1]
        acfr_std_camera_file = 'default_yaml/ts1/SSK17-01/camera.yaml'
        cs = CameraSystem(root / acfr_std_camera_file)
        self.assertEqual(cs.camera_system, 'acfr_standard', 'Wrong camera system')
        self.assertEqual(len(cs.cameras), 2, 'Wrong camera count')

        self.assertEqual(cs.cameras[0].name, 'LC', 'Wrong camera name')
        self.assertEqual(cs.cameras[0].type, 'grgb', 'Wrong camera type')
        self.assertEqual(cs.cameras[0].bit_depth, 12, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[0].path, 'image/i*/', 'Wrong camera path')
        self.assertEqual(cs.cameras[0].extension, 'tif', 'Wrong camera extension')
        self.assertEqual(cs.cameras[0].filename_to_date, 'xxxYYYYMMDDxhhmmssxfffxxxxx.xxx', 'Wrong camera filename_to_date')

        self.assertEqual(cs.cameras[1].name, 'RC', 'Wrong camera name')
        self.assertEqual(cs.cameras[1].type, 'grgb', 'Wrong camera type')
        self.assertEqual(cs.cameras[1].bit_depth, 12, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[1].path, 'image/i*/', 'Wrong camera path')
        self.assertEqual(cs.cameras[1].extension, 'tif', 'Wrong camera extension')
        self.assertEqual(cs.cameras[1].filename_to_date, 'xxxYYYYMMDDxhhmmssxfffxxxxx.xxx', 'Wrong camera filename_to_date')

    def checkValidColumns(self, node):
        # Count number of Y, M, D, h, m, s and f
        content = ''
        for item in node:
            self.assertIn('name', item, 'Field name is not present in timestamp file columns')
            self.assertIn('content', item, 'Field name is not present in timestamp file columns')
            content += item['content']

        self.assertEqual(content.count('Y'), 4, 'Year speficication is incorrect for timestamp file data columns')
        self.assertEqual(content.count('M'), 2, 'Month speficication is incorrect for timestamp file data columns')
        self.assertEqual(content.count('D'), 2, 'Day speficication is incorrect for timestamp file data columns')
        self.assertEqual(content.count('h'), 2, 'Hour speficication is incorrect for timestamp file data columns')
        self.assertEqual(content.count('m'), 2, 'Minute speficication is incorrect for timestamp file data columns')
        self.assertEqual(content.count('s'), 2, 'Second speficication is incorrect for timestamp file data columns')
        self.assertEqual(content.count('f'), 3, 'Millisecond speficication is incorrect for timestamp file data columns')

    def testSeaxerocks3(self):
        root = Path(__file__).parents[1]
        sx3_camera_file = 'default_yaml/ae2000/YK17-23C/camera.yaml'
        cs = CameraSystem(root / sx3_camera_file)
        self.assertEqual(cs.camera_system, 'seaxerocks_3', 'Wrong camera system')
        self.assertEqual(len(cs.cameras), 3, 'Wrong camera count')

        self.assertEqual(cs.cameras[0].name, 'Cam51707923', 'Wrong camera name')
        self.assertEqual(cs.cameras[0].type, 'rggb', 'Wrong camera type')
        self.assertEqual(cs.cameras[0].bit_depth, 18, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[0].path, 'image/SeaXerocksData*/Xviii/Cam51707923', 'Wrong camera path')
        self.assertEqual(cs.cameras[0].extension, 'raw', 'Wrong camera extension')
        self.assertEqual(cs.cameras[0].filename_to_date, 'iiiiiii.xxx', 'Wrong camera filename_to_date')
        self.assertEqual(cs.cameras[0].timestamp_file, 'image/SeaXerocksData*/Xviii/FileTime.csv', 'Wrong camera timestamp_file')
        self.assertEqual(len(cs.cameras[0].columns), 4, 'Wrong number of columns')
        self.checkValidColumns(cs.cameras[0].columns)

        self.assertEqual(cs.cameras[1].name, 'Cam51707925', 'Wrong camera name')
        self.assertEqual(cs.cameras[1].type, 'rggb', 'Wrong camera type')
        self.assertEqual(cs.cameras[1].bit_depth, 18, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[1].path, 'image/SeaXerocksData*/Xviii/Cam51707925', 'Wrong camera path')
        self.assertEqual(cs.cameras[1].extension, 'raw', 'Wrong camera extension')
        self.assertEqual(cs.cameras[1].filename_to_date, 'iiiiiii.xxx', 'Wrong camera filename_to_date')
        self.assertEqual(cs.cameras[1].timestamp_file, 'image/SeaXerocksData*/Xviii/FileTime.csv', 'Wrong camera timestamp_file')
        self.checkValidColumns(cs.cameras[1].columns)

        self.assertEqual(cs.cameras[2].name, 'LM165', 'Wrong camera name')
        self.assertEqual(cs.cameras[2].type, 'grayscale', 'Wrong camera type')
        self.assertEqual(cs.cameras[2].bit_depth, 8, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[2].path, 'image/SeaXerocksData*/LM165', 'Wrong camera path')
        self.assertEqual(cs.cameras[2].extension, 'tif', 'Wrong camera extension')
        self.assertEqual(cs.cameras[2].filename_to_date, 'iiiiiii.xxx', 'Wrong camera filename_to_date')
        self.assertEqual(cs.cameras[2].timestamp_file, 'image/SeaXerocksData*/LM165/FileTime.csv', 'Wrong camera timestamp_file')
        self.checkValidColumns(cs.cameras[2].columns)

    def testBiocam(self):
        root = Path(__file__).parents[1]
        biocam_camera_file = 'default_yaml/as6/DY109/camera.yaml'
        cs = CameraSystem(root / biocam_camera_file)
        self.assertEqual(cs.camera_system, 'biocam', 'Wrong camera system')
        self.assertEqual(len(cs.cameras), 3, 'Wrong camera count')

        self.assertEqual(cs.cameras[0].name, 'cam61003146', 'Wrong camera name')
        self.assertEqual(cs.cameras[0].type, 'rggb', 'Wrong camera type')
        self.assertEqual(cs.cameras[0].bit_depth, 16, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[0].path, 'image/cam61003146_strobed', 'Wrong camera path')
        self.assertEqual(cs.cameras[0].extension, 'tif', 'Wrong camera extension')
        self.assertEqual(cs.cameras[0].filename_to_date, 'YYYYMMDDxhhmmssxfffuuuxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxx', 'Wrong camera filename_to_date')

        self.assertEqual(cs.cameras[1].name, 'cam61004444', 'Wrong camera name')
        self.assertEqual(cs.cameras[1].type, 'grayscale', 'Wrong camera type')
        self.assertEqual(cs.cameras[1].bit_depth, 16, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[1].path, 'image/cam61004444_strobed', 'Wrong camera path')
        self.assertEqual(cs.cameras[1].extension, 'tif', 'Wrong camera extension')
        self.assertEqual(cs.cameras[1].filename_to_date, 'YYYYMMDDxhhmmssxfffuuuxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxx', 'Wrong camera filename_to_date')

        self.assertEqual(cs.cameras[2].name, 'cam61004444_laser', 'Wrong camera name')
        self.assertEqual(cs.cameras[2].type, 'grayscale', 'Wrong camera type')
        self.assertEqual(cs.cameras[2].bit_depth, 16, 'Wrong camera bit_depth')
        self.assertEqual(cs.cameras[2].path, 'image/cam61004444_laser', 'Wrong camera path')
        self.assertEqual(cs.cameras[2].extension, 'jpg', 'Wrong camera extension')
        self.assertEqual(cs.cameras[2].filename_to_date, 'YYYYMMDDxhhmmssxfffuuuxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxx', 'Wrong camera filename_to_date')
