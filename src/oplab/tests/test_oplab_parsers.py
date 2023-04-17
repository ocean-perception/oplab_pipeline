import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oplab.console import Console
from oplab.mission import CameraEntry, Mission
from oplab.vehicle import Vehicle


def get_empty_test_directory():
    # test_directory = Path('D:/temp/')
    test_directory = Path(tempfile.gettempdir())
    test_directory /= "module_testing_oplab_parsers"
    # Delete all files in test directory, in case there are any
    if test_directory.exists() and test_directory.is_dir():
        for x in test_directory.iterdir():
            if x.is_file():
                x.unlink()  # delete file
    else:
        test_directory.mkdir()
    return test_directory


class TestLoadMissionYK1723C(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= "default_yaml/ae2000/YK17-23C/mission.yaml"
        with patch.object(Console, "get_version", return_value="testing"):
            self.m = Mission(path)

    def test_LoadMission(self):
        self.assertEqual(self.m.version, 2)

        self.assertEqual(self.m.origin.latitude, 22.8778145)
        self.assertEqual(self.m.origin.longitude, 153.38397866666668)
        self.assertEqual(self.m.origin.crs, "wgs84")
        self.assertEqual(self.m.origin.date, "2018/11/19")

        self.assertEqual(self.m.velocity.format, "ae2000")
        self.assertEqual(self.m.velocity.filepath, "nav/ae_log/")
        self.assertEqual(self.m.velocity.filename, "pos181119064007.csv")
        self.assertEqual(self.m.velocity.label, 0)
        self.assertEqual(self.m.velocity.timezone, 10)
        self.assertEqual(self.m.velocity.timeoffset_s, 0.0)
        self.assertEqual(self.m.velocity.std_factor, 0.001)
        self.assertEqual(self.m.velocity.std_offset, 0.2)

        self.assertEqual(self.m.orientation.format, "ae2000")
        self.assertEqual(self.m.orientation.filepath, "nav/ae_log/")
        self.assertEqual(self.m.orientation.filename, "pos181119064007.csv")
        self.assertEqual(self.m.orientation.label, 0)
        self.assertEqual(self.m.orientation.timezone, 10)
        self.assertEqual(self.m.orientation.timeoffset_s, 0.0)
        self.assertEqual(self.m.orientation.std_factor, 0.0)
        self.assertEqual(self.m.orientation.std_offset, 0.003)

        self.assertEqual(self.m.depth.format, "ae2000")
        self.assertEqual(self.m.depth.filepath, "nav/ae_log/")
        self.assertEqual(self.m.depth.filename, "pos181119064007.csv")
        self.assertEqual(self.m.depth.label, 0)
        self.assertEqual(self.m.depth.timezone, 10)
        self.assertEqual(self.m.depth.timeoffset_s, 0.0)
        self.assertEqual(self.m.depth.std_factor, 0.0001)
        self.assertEqual(self.m.depth.std_offset, 0.0)

        self.assertEqual(self.m.altitude.format, "ae2000")
        self.assertEqual(self.m.altitude.filepath, "nav/ae_log/")
        self.assertEqual(self.m.altitude.filename, "pos181119064007.csv")
        self.assertEqual(self.m.altitude.label, 0)
        self.assertEqual(self.m.altitude.timezone, 10)
        self.assertEqual(self.m.altitude.timeoffset_s, 0.0)
        self.assertEqual(self.m.altitude.std_factor, 0.0001)
        self.assertEqual(self.m.altitude.std_offset, 0.0)

        self.assertEqual(self.m.usbl.format, "gaps")
        self.assertEqual(self.m.usbl.filepath, "nav/gaps/")
        self.assertEqual(self.m.usbl.filename, "")
        self.assertEqual(self.m.usbl.label, 4)
        self.assertEqual(self.m.usbl.timezone, 0)
        self.assertEqual(self.m.usbl.timeoffset_s, 0.0)
        self.assertEqual(self.m.usbl.std_factor, 0.01)
        self.assertEqual(self.m.usbl.std_offset, 2.0)

        self.assertEqual(self.m.tide.format, "")
        self.assertEqual(self.m.tide.filepath, "")
        self.assertEqual(self.m.tide.filename, "")
        self.assertEqual(self.m.tide.label, 0)
        self.assertEqual(self.m.tide.timezone, 0)
        self.assertEqual(self.m.tide.timeoffset_s, 0.0)
        self.assertEqual(self.m.tide.std_factor, 0.0)
        self.assertEqual(self.m.tide.std_offset, 0.0)

        self.assertEqual(self.m.image.format, "seaxerocks_3")
        self.assertEqual(self.m.image.cameras[0].name, "Cam51707923")
        self.assertEqual(self.m.image.cameras[0].origin, "Cam51707923")
        self.assertEqual(self.m.image.cameras[0].records_laser, False)
        self.assertEqual(
            self.m.image.cameras[0].path,
            "image/SeaXerocksData20181119_073812_laserCal/Xviii/Cam51707923",
        )
        self.assertEqual(self.m.image.cameras[1].name, "Cam51707925")
        self.assertEqual(self.m.image.cameras[1].origin, "Cam51707925")
        self.assertEqual(self.m.image.cameras[1].records_laser, False)
        self.assertEqual(
            self.m.image.cameras[1].path,
            "image/SeaXerocksData20181119_073812_laserCal/Xviii/Cam51707925",
        )
        self.assertEqual(self.m.image.cameras[2].name, "LM165")
        self.assertEqual(self.m.image.cameras[2].origin, "LM165")
        self.assertEqual(self.m.image.cameras[2].records_laser, True)
        self.assertEqual(
            self.m.image.cameras[2].path,
            "image/SeaXerocksData20181119_073812_laserCal/LM165",
        )
        self.assertEqual(self.m.image.timezone, 10)
        self.assertEqual(self.m.image.timeoffset_s, 0.0)


class TestLoadMissionDY109(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= "default_yaml/as6/DY109/mission.yaml"
        with patch.object(Console, "get_version", return_value="testing"):
            self.m = Mission(path)

    def test_LoadMission(self):
        self.assertEqual(self.m.version, 2)

        self.assertEqual(self.m.origin.latitude, 59.85643)
        self.assertEqual(self.m.origin.longitude, -7.15903)
        self.assertEqual(self.m.origin.crs, "wgs84")
        self.assertEqual(self.m.origin.date, "2019/09/21")

        self.assertEqual(self.m.velocity.format, "autosub")
        self.assertEqual(self.m.velocity.filepath, "nav/")
        self.assertEqual(self.m.velocity.filename, "M155.mat")
        self.assertEqual(self.m.velocity.label, 0)
        self.assertEqual(self.m.velocity.timezone, 0)
        self.assertEqual(self.m.velocity.timeoffset_s, 0.0)
        self.assertEqual(self.m.velocity.std_factor, 0.001)
        self.assertEqual(self.m.velocity.std_offset, 0.2)

        self.assertEqual(self.m.orientation.format, "autosub")
        self.assertEqual(self.m.orientation.filepath, "nav/")
        self.assertEqual(self.m.orientation.filename, "M155.mat")
        self.assertEqual(self.m.orientation.label, 0)
        self.assertEqual(self.m.orientation.timezone, 0)
        self.assertEqual(self.m.orientation.timeoffset_s, 0.0)
        self.assertEqual(self.m.orientation.std_factor, 0.0)
        self.assertEqual(self.m.orientation.std_offset, 0.003)

        self.assertEqual(self.m.depth.format, "autosub")
        self.assertEqual(self.m.depth.filepath, "nav/")
        self.assertEqual(self.m.depth.filename, "M155.mat")
        self.assertEqual(self.m.depth.label, 0)
        self.assertEqual(self.m.depth.timezone, 0)
        self.assertEqual(self.m.depth.timeoffset_s, 0.0)
        self.assertEqual(self.m.depth.std_factor, 0.001)
        self.assertEqual(self.m.depth.std_offset, 0.0)

        self.assertEqual(self.m.altitude.format, "autosub")
        self.assertEqual(self.m.altitude.filepath, "nav/")
        self.assertEqual(self.m.altitude.filename, "M155.mat")
        self.assertEqual(self.m.altitude.label, 0)
        self.assertEqual(self.m.altitude.timezone, 0)
        self.assertEqual(self.m.altitude.timeoffset_s, 0.0)
        self.assertEqual(self.m.altitude.std_factor, 0.001)
        self.assertEqual(self.m.altitude.std_offset, 0.3)

        self.assertEqual(self.m.usbl.format, "NOC_nmea")
        self.assertEqual(self.m.usbl.filepath, "nav/usbl/")
        self.assertEqual(self.m.usbl.filename, "")
        self.assertEqual(self.m.usbl.label, 13)
        self.assertEqual(self.m.usbl.timezone, 0)
        self.assertEqual(self.m.usbl.timeoffset_s, 0.0)
        self.assertEqual(self.m.usbl.std_factor, 0.01)
        self.assertEqual(self.m.usbl.std_offset, 2.0)

        self.assertEqual(self.m.tide.format, "NOC_polpred")
        self.assertEqual(self.m.tide.filepath, "tide/")
        self.assertEqual(self.m.tide.filename, "dy_108_polpred_tide_10m.txt")
        self.assertEqual(self.m.tide.label, 0)
        self.assertEqual(self.m.tide.timezone, 0)
        self.assertEqual(self.m.tide.timeoffset_s, 0.0)
        self.assertEqual(self.m.tide.std_factor, 0.0)
        self.assertEqual(self.m.tide.std_offset, 0.0)

        self.assertEqual(self.m.image.format, "biocam")
        self.assertEqual(self.m.image.cameras[0].name, "cam61003146")
        self.assertEqual(self.m.image.cameras[0].origin, "cam61003146")
        self.assertEqual(self.m.image.cameras[0].path, "image")
        self.assertEqual(self.m.image.cameras[0].records_laser, False)
        self.assertEqual(self.m.image.cameras[1].name, "cam61004444")
        self.assertEqual(self.m.image.cameras[1].origin, "cam61004444")
        self.assertEqual(self.m.image.cameras[1].path, "image")
        self.assertEqual(self.m.image.cameras[1].records_laser, True)
        self.assertEqual(self.m.image.timezone, 0)
        self.assertEqual(self.m.image.timeoffset_s, 0.0)


class TestLoadMissionSSK1701FileFormat0(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= "default_yaml/ts1/SSK17-01/legacy_vehicle_format_0/mission.yaml"
        with patch.object(Console, "get_version", return_value="testing"):
            self.m = Mission(path)

    def test_LoadMission(self):
        self.assertEqual(self.m.version, 1)

        self.assertEqual(self.m.origin.latitude, 26.674083)
        self.assertEqual(self.m.origin.longitude, 127.868054)
        self.assertEqual(self.m.origin.crs, "wgs84")
        self.assertEqual(self.m.origin.date, "2017/08/17")

        self.assertEqual(self.m.velocity.format, "phins")
        self.assertEqual(self.m.velocity.filepath, "nav/phins/")
        self.assertEqual(self.m.velocity.filename, "20170817_phins.txt")
        self.assertEqual(self.m.velocity.label, 0)
        self.assertEqual(self.m.velocity.timezone, 0)
        self.assertEqual(self.m.velocity.timeoffset_s, 0.0)
        self.assertEqual(self.m.velocity.std_factor, 0)
        self.assertEqual(self.m.velocity.std_offset, 0)

        self.assertEqual(self.m.orientation.format, "phins")
        self.assertEqual(self.m.orientation.filepath, "nav/phins/")
        self.assertEqual(self.m.orientation.filename, "20170817_phins.txt")
        self.assertEqual(self.m.orientation.label, 0)
        self.assertEqual(self.m.orientation.timezone, 0)
        self.assertEqual(self.m.orientation.timeoffset_s, 0.0)
        self.assertEqual(self.m.orientation.std_factor, 0)
        self.assertEqual(self.m.orientation.std_offset, 0)

        self.assertEqual(self.m.depth.format, "phins")
        self.assertEqual(self.m.depth.filepath, "nav/phins/")
        self.assertEqual(self.m.depth.filename, "20170817_phins.txt")
        self.assertEqual(self.m.depth.label, 0)
        self.assertEqual(self.m.depth.timezone, 0)
        self.assertEqual(self.m.depth.timeoffset_s, 0.0)
        self.assertEqual(self.m.depth.std_factor, 0)
        self.assertEqual(self.m.depth.std_offset, 0)

        self.assertEqual(self.m.altitude.format, "phins")
        self.assertEqual(self.m.altitude.filepath, "nav/phins/")
        self.assertEqual(self.m.altitude.filename, "20170817_phins.txt")
        self.assertEqual(self.m.altitude.label, 0)
        self.assertEqual(self.m.altitude.timezone, 0)
        self.assertEqual(self.m.altitude.timeoffset_s, 0.0)
        self.assertEqual(self.m.altitude.std_factor, 0)
        self.assertEqual(self.m.altitude.std_offset, 0)

        self.assertEqual(self.m.usbl.format, "gaps")
        self.assertEqual(self.m.usbl.filepath, "nav/gaps/")
        self.assertEqual(self.m.usbl.filename, "")
        self.assertEqual(self.m.usbl.label, 2)
        self.assertEqual(self.m.usbl.timezone, 0)
        self.assertEqual(self.m.usbl.timeoffset_s, 0.0)
        self.assertEqual(self.m.usbl.std_factor, 0)
        self.assertEqual(self.m.usbl.std_offset, 0)

        self.assertEqual(self.m.tide.format, "")
        self.assertEqual(self.m.tide.filepath, "")
        self.assertEqual(self.m.tide.filename, "")
        self.assertEqual(self.m.tide.label, 0)
        self.assertEqual(self.m.tide.timezone, 0)
        self.assertEqual(self.m.tide.timeoffset_s, 0.0)
        self.assertEqual(self.m.tide.std_factor, 0)
        self.assertEqual(self.m.tide.std_offset, 0)

        self.assertEqual(self.m.image.format, "acfr_standard")
        self.assertEqual(self.m.image.cameras[0].name, "camera1")
        self.assertEqual(self.m.image.cameras[0].origin, "camera1")
        self.assertEqual(self.m.image.cameras[0].records_laser, False)
        self.assertEqual(
            self.m.image.cameras[0].path,
            "image/r20170817_041459_UG117_sesoko/i20170817_041459/",
        )
        self.assertEqual(self.m.image.cameras[1].name, "camera2")
        self.assertEqual(self.m.image.cameras[1].origin, "camera2")
        self.assertEqual(self.m.image.cameras[1].records_laser, False)
        self.assertEqual(
            self.m.image.cameras[1].path,
            "image/r20170817_041459_UG117_sesoko/i20170817_041459/",
        )
        self.assertEqual(self.m.image.timezone, 9)
        self.assertEqual(self.m.image.timeoffset_s, 0.0)


class TestLoadVehicleYK1723C(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= "default_yaml/ae2000/YK17-23C/vehicle.yaml"
        self.v = Vehicle(path)

    def test_LoadVehicle(self):
        self.assertEqual(self.v.origin.surge, 0)
        self.assertEqual(self.v.origin.sway, 0)
        self.assertEqual(self.v.origin.heave, 0)
        self.assertEqual(self.v.origin.roll, 0)
        self.assertEqual(self.v.origin.pitch, 0)
        self.assertEqual(self.v.origin.yaw, 0)

        self.assertEqual(self.v.ins.surge, -0.09)
        self.assertEqual(self.v.ins.sway, 0)
        self.assertEqual(self.v.ins.heave, 0)
        self.assertEqual(self.v.ins.roll, 0)
        self.assertEqual(self.v.ins.pitch, 0)
        self.assertEqual(self.v.ins.yaw, 0.0)

        self.assertEqual(self.v.dvl.surge, -0.780625)
        self.assertEqual(self.v.dvl.sway, 0)
        self.assertEqual(self.v.dvl.heave, 0.204)
        self.assertEqual(self.v.dvl.roll, 0)
        self.assertEqual(self.v.dvl.pitch, 0)
        self.assertEqual(self.v.dvl.yaw, 0.0)

        self.assertEqual(self.v.depth.surge, 0.0)
        self.assertEqual(self.v.depth.sway, 0)
        self.assertEqual(self.v.depth.heave, 0)
        self.assertEqual(self.v.depth.roll, 0)
        self.assertEqual(self.v.depth.pitch, 0)
        self.assertEqual(self.v.depth.yaw, 0)

        self.assertEqual(self.v.usbl.surge, 0.0)
        self.assertEqual(self.v.usbl.sway, 0)
        self.assertEqual(self.v.usbl.heave, -0.289)
        self.assertEqual(self.v.usbl.roll, 0)
        self.assertEqual(self.v.usbl.pitch, 0)
        self.assertEqual(self.v.usbl.yaw, 0)

        self.assertEqual(self.v.camera1.surge, 0.262875)
        self.assertEqual(self.v.camera1.sway, 0.0)
        self.assertEqual(self.v.camera1.heave, 0.5)
        self.assertEqual(self.v.camera1.roll, 0)
        self.assertEqual(self.v.camera1.pitch, 0)
        self.assertEqual(self.v.camera1.yaw, 0)

        self.assertEqual(self.v.camera2.surge, 0.012875)
        self.assertEqual(self.v.camera2.sway, 0.0)
        self.assertEqual(self.v.camera2.heave, 0.5)
        self.assertEqual(self.v.camera2.roll, 0)
        self.assertEqual(self.v.camera2.pitch, 0)
        self.assertEqual(self.v.camera2.yaw, 0)

        self.assertEqual(self.v.camera3.surge, 0.150375)
        self.assertEqual(self.v.camera3.sway, 0)
        self.assertEqual(self.v.camera3.heave, 0.514)
        self.assertEqual(self.v.camera3.roll, 0)
        self.assertEqual(self.v.camera3.pitch, 0)
        self.assertEqual(self.v.camera3.yaw, 0)


class TestLoadVehicleDY109(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= "default_yaml/as6/DY109/vehicle.yaml"
        self.v = Vehicle(path)

    def test_LoadVehicle(self):
        self.assertEqual(self.v.origin.surge, 0)
        self.assertEqual(self.v.origin.sway, 0)
        self.assertEqual(self.v.origin.heave, 0)
        self.assertEqual(self.v.origin.roll, 0)
        self.assertEqual(self.v.origin.pitch, 0)
        self.assertEqual(self.v.origin.yaw, 0)

        self.assertEqual(self.v.ins.surge, 0)
        self.assertEqual(self.v.ins.sway, 0)
        self.assertEqual(self.v.ins.heave, 0)
        self.assertEqual(self.v.ins.roll, 0)
        self.assertEqual(self.v.ins.pitch, 0)
        self.assertEqual(self.v.ins.yaw, 0)

        self.assertEqual(self.v.dvl.surge, 0)
        self.assertEqual(self.v.dvl.sway, 0)
        self.assertEqual(self.v.dvl.heave, 0)
        self.assertEqual(self.v.dvl.roll, 0)
        self.assertEqual(self.v.dvl.pitch, 0)
        self.assertEqual(self.v.dvl.yaw, 0)

        self.assertEqual(self.v.depth.surge, -0.112)
        self.assertEqual(self.v.depth.sway, 0.230)
        self.assertEqual(self.v.depth.heave, -0.331)
        self.assertEqual(self.v.depth.roll, 0)
        self.assertEqual(self.v.depth.pitch, 0)
        self.assertEqual(self.v.depth.yaw, 0)

        self.assertEqual(self.v.usbl.surge, 2.557)
        self.assertEqual(self.v.usbl.sway, -0.120)
        self.assertEqual(self.v.usbl.heave, -0.694)
        self.assertEqual(self.v.usbl.roll, 0)
        self.assertEqual(self.v.usbl.pitch, 0)
        self.assertEqual(self.v.usbl.yaw, 0)

        self.assertEqual(self.v.camera1.surge, 1.484)
        self.assertEqual(self.v.camera1.sway, 0.0)
        self.assertEqual(self.v.camera1.heave, 0.327)
        self.assertEqual(self.v.camera1.roll, 0)
        self.assertEqual(self.v.camera1.pitch, -90)
        self.assertEqual(self.v.camera1.yaw, 0)

        self.assertEqual(self.v.camera2.surge, 1.104)
        self.assertEqual(self.v.camera2.sway, 0.0)
        self.assertEqual(self.v.camera2.heave, 0.327)
        self.assertEqual(self.v.camera2.roll, 0)
        self.assertEqual(self.v.camera2.pitch, -90)
        self.assertEqual(self.v.camera2.yaw, 0)

        self.assertEqual(self.v.camera3.surge, 0)
        self.assertEqual(self.v.camera3.sway, 0)
        self.assertEqual(self.v.camera3.heave, 0)
        self.assertEqual(self.v.camera3.roll, 0)
        self.assertEqual(self.v.camera3.pitch, 0)
        self.assertEqual(self.v.camera3.yaw, 0)


class TestLoadVehicleSSK1701FileFormat0(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= "default_yaml/ts1/SSK17-01/legacy_vehicle_format_0/vehicle.yaml"
        self.v = Vehicle(path)

    def test_LoadVehicle(self):
        self.assertEqual(self.v.origin.surge, 0)
        self.assertEqual(self.v.origin.sway, 0)
        self.assertEqual(self.v.origin.heave, 0)
        self.assertEqual(self.v.origin.roll, 0)
        self.assertEqual(self.v.origin.pitch, 0)
        self.assertEqual(self.v.origin.yaw, 0)

        self.assertEqual(self.v.ins.surge, 0.1)
        self.assertEqual(self.v.ins.sway, 0)
        self.assertEqual(self.v.ins.heave, 0)
        self.assertEqual(self.v.ins.roll, 0)
        self.assertEqual(self.v.ins.pitch, 0)
        self.assertEqual(
            self.v.ins.yaw, -45.0
        )  # read form mission.yaml (headingoffset), not from vehicle.yaml!

        self.assertEqual(self.v.dvl.surge, -0.45)
        self.assertEqual(self.v.dvl.sway, 0)
        self.assertEqual(self.v.dvl.heave, 0.45)
        self.assertEqual(self.v.dvl.roll, 0)
        self.assertEqual(self.v.dvl.pitch, 0)
        self.assertEqual(
            self.v.dvl.yaw, -45.0
        )  # read form mission.yaml (headingoffset), not from vehicle.yaml!

        self.assertEqual(self.v.depth.surge, 0.16)
        self.assertEqual(self.v.depth.sway, 0)
        self.assertEqual(self.v.depth.heave, 0)
        self.assertEqual(self.v.depth.roll, 0)
        self.assertEqual(self.v.depth.pitch, 0)
        self.assertEqual(self.v.depth.yaw, 0)

        self.assertEqual(self.v.usbl.surge, 0.1)
        self.assertEqual(self.v.usbl.sway, 0)
        self.assertEqual(self.v.usbl.heave, -0.5)
        self.assertEqual(self.v.usbl.roll, 0)
        self.assertEqual(self.v.usbl.pitch, 0)
        self.assertEqual(self.v.usbl.yaw, 0)

        self.assertEqual(self.v.camera1.surge, -0.05)
        self.assertEqual(self.v.camera1.sway, -0.3)
        self.assertEqual(self.v.camera1.heave, 0.18)
        self.assertEqual(self.v.camera1.roll, 0)
        self.assertEqual(self.v.camera1.pitch, 0)
        self.assertEqual(self.v.camera1.yaw, 0)

        self.assertEqual(self.v.camera2.surge, -0.05)
        self.assertEqual(self.v.camera2.sway, -0.1)
        self.assertEqual(self.v.camera2.heave, 0.18)
        self.assertEqual(self.v.camera2.roll, 0)
        self.assertEqual(self.v.camera2.pitch, 0)
        self.assertEqual(self.v.camera2.yaw, 0)

        self.assertEqual(self.v.camera3.surge, 0)
        self.assertEqual(self.v.camera3.sway, 0)
        self.assertEqual(self.v.camera3.heave, 0)
        self.assertEqual(self.v.camera3.roll, 0)
        self.assertEqual(self.v.camera3.pitch, 0)
        self.assertEqual(self.v.camera3.yaw, 0)


class TestWriteMissionDY109(unittest.TestCase):
    def setUp(self):
        directory = get_empty_test_directory()
        mission_path = directory / "mission.yaml"
        vehicle_path = directory / "vehicle.yaml"
        # Loading mission.yaml requires the corresponding vehicle.yaml file to
        # be present, but there is currently no function to write entire
        # vehicle.yaml files, therefore an existing file has to be used
        vehicle_path_dy109 = Path(__file__).parents[1]
        vehicle_path_dy109 /= "default_yaml/as6/DY109/vehicle.yaml"
        shutil.copy(str(vehicle_path_dy109), str(vehicle_path))
        self.v1 = Vehicle(vehicle_path_dy109)
        with patch.object(Console, "get_version", return_value="testing"):
            self.m1 = Mission()

            self.m1.version = 1

            self.m1.origin.latitude = 59.85643
            self.m1.origin.longitude = -7.15903
            self.m1.origin.crs = "wgs84"
            self.m1.origin.date = "2019/09/21"

            self.m1.velocity.format = "autosub"
            self.m1.velocity.filepath = "nav/"
            self.m1.velocity.filename = "M155.mat"
            self.m1.velocity.label = 0
            self.m1.velocity.timezone = 0
            self.m1.velocity.timeoffset_s = 0.0
            self.m1.velocity.std_factor = 0.001
            self.m1.velocity.std_offset = 0.2
            self.m1.velocity._empty = False

            self.m1.orientation.format = "autosub"
            self.m1.orientation.filepath = "nav/"
            self.m1.orientation.filename = "M155.mat"
            self.m1.orientation.label = 0
            self.m1.orientation.timezone = 0
            self.m1.orientation.timeoffset_s = 0.0
            self.m1.orientation.std_factor = 0.0
            self.m1.orientation.std_offset = 0.003
            self.m1.orientation._empty = False

            self.m1.depth.format = "autosub"
            self.m1.depth.filepath = "nav/"
            self.m1.depth.filename = "M155.mat"
            self.m1.depth.label = 0
            self.m1.depth.timezone = 0
            self.m1.depth.timeoffset_s = 0.0
            self.m1.depth.std_factor = 0.001
            self.m1.depth.std_offset = 0.0
            self.m1.depth._empty = False

            self.m1.altitude.format = "autosub"
            self.m1.altitude.filepath = "nav/"
            self.m1.altitude.filename = "M155.mat"
            self.m1.altitude.label = 0
            self.m1.altitude.timezone = 0
            self.m1.altitude.timeoffset_s = 0.0
            self.m1.altitude.std_factor = 0.001
            self.m1.altitude.std_offset = 0.3
            self.m1.altitude._empty = False

            self.m1.usbl.format = "NOC_nmea"
            self.m1.usbl.filepath = "nav/usbl/"
            self.m1.usbl.filename = ""
            self.m1.usbl.label = 13
            self.m1.usbl.timezone = 0
            self.m1.usbl.timeoffset_s = 0.0
            self.m1.usbl.std_factor = 0.01
            self.m1.usbl.std_offset = 2.0
            self.m1.usbl._empty = False

            self.m1.tide.format = "NOC_polpred"
            self.m1.tide.filepath = "tide/"
            self.m1.tide.filename = "dy_108_polpred_tide_10m.txt"
            self.m1.tide.label = 0
            self.m1.tide.timezone = 0
            self.m1.tide.timeoffset_s = 0.0
            self.m1.tide.std_factor = 0.0
            self.m1.tide.std_offset = 0.0
            self.m1.tide._empty = False

            self.m1.image.format = "biocam"
            self.m1.image.cameras.append(CameraEntry())
            self.m1.image.cameras[0].name = "cam61003146"
            self.m1.image.cameras[0].origin = "cam61003146"
            self.m1.image.cameras[0].records_laser = False
            self.m1.image.cameras[0].path = "image"
            self.m1.image.cameras.append(CameraEntry())
            self.m1.image.cameras[1].name = "cam61004444"
            self.m1.image.cameras[1].origin = "cam61004444"
            self.m1.image.cameras[1].records_laser = True
            self.m1.image.cameras[1].path = "image"
            self.m1.image.timezone = 0
            self.m1.image.timeoffset_s = 0.0
            self.m1.image._empty = False

            self.m1.write(mission_path)
            self.m2 = Mission(mission_path)

    def test_WriteMission(self):
        self.assertEqual(self.m1.version, self.m2.version)

        self.assertEqual(self.m1.origin.latitude, self.m2.origin.latitude)
        self.assertEqual(self.m1.origin.longitude, self.m2.origin.longitude)
        self.assertEqual(self.m1.origin.crs, self.m2.origin.crs)
        self.assertEqual(self.m1.origin.date, self.m2.origin.date)

        self.assertEqual(self.m1.velocity.format, self.m2.velocity.format)
        self.assertEqual(self.m1.velocity.filepath, self.m2.velocity.filepath)
        self.assertEqual(self.m1.velocity.filename, self.m2.velocity.filename)
        self.assertEqual(self.m1.velocity.label, self.m2.velocity.label)
        self.assertEqual(self.m1.velocity.timezone, self.m2.velocity.timezone)
        self.assertEqual(self.m1.velocity.timeoffset_s, self.m2.velocity.timeoffset_s)
        self.assertEqual(self.m1.velocity.std_factor, self.m2.velocity.std_factor)
        self.assertEqual(self.m1.velocity.std_offset, self.m2.velocity.std_offset)

        self.assertEqual(self.m1.orientation.format, self.m2.orientation.format)
        self.assertEqual(self.m1.orientation.filepath, self.m2.orientation.filepath)
        self.assertEqual(self.m1.orientation.filename, self.m2.orientation.filename)
        self.assertEqual(self.m1.orientation.label, self.m2.orientation.label)
        self.assertEqual(self.m1.orientation.timezone, self.m2.orientation.timezone)
        self.assertEqual(
            self.m1.orientation.timeoffset_s, self.m2.orientation.timeoffset_s
        )
        self.assertEqual(self.m1.orientation.std_factor, self.m2.orientation.std_factor)
        self.assertEqual(self.m1.orientation.std_offset, self.m2.orientation.std_offset)

        self.assertEqual(self.m1.depth.format, self.m2.depth.format)
        self.assertEqual(self.m1.depth.filepath, self.m2.depth.filepath)
        self.assertEqual(self.m1.depth.filename, self.m2.depth.filename)
        self.assertEqual(self.m1.depth.label, self.m2.depth.label)
        self.assertEqual(self.m1.depth.timezone, self.m2.depth.timezone)
        self.assertEqual(self.m1.depth.timeoffset_s, self.m2.depth.timeoffset_s)
        self.assertEqual(self.m1.depth.std_factor, self.m2.depth.std_factor)
        self.assertEqual(self.m1.depth.std_offset, self.m2.depth.std_offset)

        self.assertEqual(self.m1.altitude.format, self.m2.altitude.format)
        self.assertEqual(self.m1.altitude.filepath, self.m2.altitude.filepath)
        self.assertEqual(self.m1.altitude.filename, self.m2.altitude.filename)
        self.assertEqual(self.m1.altitude.label, self.m2.altitude.label)
        self.assertEqual(self.m1.altitude.timezone, self.m2.altitude.timezone)
        self.assertEqual(self.m1.altitude.timeoffset_s, self.m2.altitude.timeoffset_s)
        self.assertEqual(self.m1.altitude.std_factor, self.m2.altitude.std_factor)
        self.assertEqual(self.m1.altitude.std_offset, self.m2.altitude.std_offset)

        self.assertEqual(self.m1.usbl.format, self.m2.usbl.format)
        self.assertEqual(self.m1.usbl.filepath, self.m2.usbl.filepath)
        self.assertEqual(self.m1.usbl.filename, self.m2.usbl.filename)
        self.assertEqual(self.m1.usbl.label, self.m2.usbl.label)
        self.assertEqual(self.m1.usbl.timezone, self.m2.usbl.timezone)
        self.assertEqual(self.m1.usbl.timeoffset_s, self.m2.usbl.timeoffset_s)
        self.assertEqual(self.m1.usbl.std_factor, self.m2.usbl.std_factor)
        self.assertEqual(self.m1.usbl.std_offset, self.m2.usbl.std_offset)

        self.assertEqual(self.m1.tide.format, self.m2.tide.format)
        self.assertEqual(self.m1.tide.filepath, self.m2.tide.filepath)
        self.assertEqual(self.m1.tide.filename, self.m2.tide.filename)
        self.assertEqual(self.m1.tide.label, self.m2.tide.label)
        self.assertEqual(self.m1.tide.timezone, self.m2.tide.timezone)
        self.assertEqual(self.m1.tide.timeoffset_s, self.m2.tide.timeoffset_s)
        self.assertEqual(self.m1.tide.std_factor, self.m2.tide.std_factor)
        self.assertEqual(self.m1.tide.std_offset, self.m2.tide.std_offset)

        self.assertEqual(self.m1.image.format, self.m2.image.format)
        self.assertEqual(self.m1.image.cameras[0].name, self.m2.image.cameras[0].name)
        self.assertEqual(
            self.m1.image.cameras[0].origin, self.m2.image.cameras[0].origin
        )
        self.assertEqual(
            self.m1.image.cameras[0].records_laser,
            self.m2.image.cameras[0].records_laser,
        )
        self.assertEqual(self.m1.image.cameras[0].path, self.m2.image.cameras[0].path)
        self.assertEqual(self.m1.image.cameras[1].name, self.m2.image.cameras[1].name)
        self.assertEqual(
            self.m1.image.cameras[1].origin, self.m2.image.cameras[1].origin
        )
        self.assertEqual(
            self.m1.image.cameras[1].records_laser,
            self.m2.image.cameras[1].records_laser,
        )
        self.assertEqual(self.m1.image.cameras[1].path, self.m2.image.cameras[1].path)
        self.assertEqual(self.m1.image.timezone, self.m2.image.timezone)
        self.assertEqual(self.m1.image.timeoffset_s, self.m2.image.timeoffset_s)
