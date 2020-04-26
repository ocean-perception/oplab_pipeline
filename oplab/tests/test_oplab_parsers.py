import unittest
from pathlib import Path
from oplab.mission import Mission
from oplab.vehicle import Vehicle


class TestLoadMissionYK1723C(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= 'default_yaml/ae2000/YK17-23C/mission.yaml'
        self.m = Mission(path)

    def test_LoadMission(self):
        self.assertEqual(self.m.version, 1)

        self.assertEqual(self.m.origin.latitude, 22.8778145)
        self.assertEqual(self.m.origin.longitude, 153.38397866666668)
        self.assertEqual(self.m.origin.crs, 'wgs84')
        self.assertEqual(self.m.origin.date, '2018/11/19')

        self.assertEqual(self.m.velocity.format, 'ae2000')
        self.assertEqual(self.m.velocity.filepath, 'nav/ae_log/')
        self.assertEqual(self.m.velocity.filename, 'pos181119064007.csv')
        self.assertEqual(self.m.velocity.label, 0)
        self.assertEqual(self.m.velocity.timezone, 10)
        self.assertEqual(self.m.velocity.timeoffset, 0.0)
        self.assertEqual(self.m.velocity.std_factor, 0.001)
        self.assertEqual(self.m.velocity.std_offset, 0.2)

        self.assertEqual(self.m.orientation.format, 'ae2000')
        self.assertEqual(self.m.orientation.filepath, 'nav/ae_log/')
        self.assertEqual(self.m.orientation.filename, 'pos181119064007.csv')
        self.assertEqual(self.m.orientation.label, 0)
        self.assertEqual(self.m.orientation.timezone, 10)
        self.assertEqual(self.m.orientation.timeoffset, 0.0)
        self.assertEqual(self.m.orientation.std_factor, 0.0)
        self.assertEqual(self.m.orientation.std_offset, 0.003)

        self.assertEqual(self.m.depth.format, 'ae2000')
        self.assertEqual(self.m.depth.filepath, 'nav/ae_log/')
        self.assertEqual(self.m.depth.filename, 'pos181119064007.csv')
        self.assertEqual(self.m.depth.label, 0)
        self.assertEqual(self.m.depth.timezone, 10)
        self.assertEqual(self.m.depth.timeoffset, 0.0)
        self.assertEqual(self.m.depth.std_factor, 0.0001)
        self.assertEqual(self.m.depth.std_offset, 0.0)

        self.assertEqual(self.m.altitude.format, 'ae2000')
        self.assertEqual(self.m.altitude.filepath, 'nav/ae_log/')
        self.assertEqual(self.m.altitude.filename, 'pos181119064007.csv')
        self.assertEqual(self.m.altitude.label, 0)
        self.assertEqual(self.m.altitude.timezone, 10)
        self.assertEqual(self.m.altitude.timeoffset, 0.0)
        self.assertEqual(self.m.altitude.std_factor, 0.0001)
        self.assertEqual(self.m.altitude.std_offset, 0.0)

        self.assertEqual(self.m.usbl.format, 'gaps')
        self.assertEqual(self.m.usbl.filepath, 'nav/gaps/')
        self.assertEqual(self.m.usbl.filename, '')
        self.assertEqual(self.m.usbl.label, 4)
        self.assertEqual(self.m.usbl.timezone, 0)
        self.assertEqual(self.m.usbl.timeoffset, 0.0)
        self.assertEqual(self.m.usbl.std_factor, 0.01)
        self.assertEqual(self.m.usbl.std_offset, 2.0)

        self.assertEqual(self.m.tide.format, '')
        self.assertEqual(self.m.tide.filepath, '')
        self.assertEqual(self.m.tide.filename, '')
        self.assertEqual(self.m.tide.label, 0)
        self.assertEqual(self.m.tide.timezone, 0)
        self.assertEqual(self.m.tide.timeoffset, 0.0)
        self.assertEqual(self.m.tide.std_factor, 0.0)
        self.assertEqual(self.m.tide.std_offset, 0.0)

        self.assertEqual(self.m.image.format, 'seaxerocks_3')
        self.assertEqual(self.m.image.cameras[0].name, 'fore')
        self.assertEqual(self.m.image.cameras[0].origin, 'fore')
        self.assertEqual(self.m.image.cameras[0].type, 'bayer_rggb')
        #self.assertEqual(self.m.image.cameras[0].bit_depth, 12) # this does not get loaded
        self.assertEqual(self.m.image.cameras[0].path, 'image/SeaXerocksData20181119_073812_laserCal/Xviii/Cam51707923')
        self.assertEqual(self.m.image.cameras[1].name, 'aft')
        self.assertEqual(self.m.image.cameras[1].origin, 'aft')
        self.assertEqual(self.m.image.cameras[1].type, 'bayer_rggb')
        #self.assertEqual(self.m.image.cameras[1].bit_depth, 12) # this does not get loaded
        self.assertEqual(self.m.image.cameras[1].path, 'image/SeaXerocksData20181119_073812_laserCal/Xviii/Cam51707925')
        self.assertEqual(self.m.image.cameras[2].name, 'laser')
        self.assertEqual(self.m.image.cameras[2].origin, 'laser')
        self.assertEqual(self.m.image.cameras[2].type, 'grayscale')
        #self.assertEqual(self.m.image.cameras[2].bit_depth, 12) # this does not get loaded
        self.assertEqual(self.m.image.cameras[2].path, 'image/SeaXerocksData20181119_073812_laserCal/LM165')
        self.assertEqual(self.m.image.timezone, 10)
        self.assertEqual(self.m.image.timeoffset, 0.0)


class TestLoadMissionDY109(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= 'default_yaml/as6/DY109/mission.yaml'
        self.m = Mission(path)

    def test_LoadMission(self):
        self.assertEqual(self.m.version, 1)

        self.assertEqual(self.m.origin.latitude, 59.85643)
        self.assertEqual(self.m.origin.longitude, -7.15903)
        self.assertEqual(self.m.origin.crs, 'wgs84')
        self.assertEqual(self.m.origin.date, '2019/09/21')

        self.assertEqual(self.m.velocity.format, 'autosub')
        self.assertEqual(self.m.velocity.filepath, 'nav/')
        self.assertEqual(self.m.velocity.filename, 'M155.mat')
        self.assertEqual(self.m.velocity.label, 0)
        self.assertEqual(self.m.velocity.timezone, 0)
        self.assertEqual(self.m.velocity.timeoffset, 0.0)
        self.assertEqual(self.m.velocity.std_factor, 0.001)
        self.assertEqual(self.m.velocity.std_offset, 0.2)

        self.assertEqual(self.m.orientation.format, 'autosub')
        self.assertEqual(self.m.orientation.filepath, 'nav/' )
        self.assertEqual(self.m.orientation.filename, 'M155.mat')
        self.assertEqual(self.m.orientation.label, 0)
        self.assertEqual(self.m.orientation.timezone, 0)
        self.assertEqual(self.m.orientation.timeoffset, 0.0)
        self.assertEqual(self.m.orientation.std_factor, 0.0)
        self.assertEqual(self.m.orientation.std_offset, 0.003)

        self.assertEqual(self.m.depth.format, 'autosub')
        self.assertEqual(self.m.depth.filepath, 'nav/')
        self.assertEqual(self.m.depth.filename, 'M155.mat')
        self.assertEqual(self.m.depth.label, 0)
        self.assertEqual(self.m.depth.timezone, 0)
        self.assertEqual(self.m.depth.timeoffset, 0.0)
        self.assertEqual(self.m.depth.std_factor, 0.001)
        self.assertEqual(self.m.depth.std_offset, 0.0)

        self.assertEqual(self.m.altitude.format, 'autosub')
        self.assertEqual(self.m.altitude.filepath, 'nav/')
        self.assertEqual(self.m.altitude.filename, 'M155.mat')
        self.assertEqual(self.m.altitude.label, 0)
        self.assertEqual(self.m.altitude.timezone, 0)
        self.assertEqual(self.m.altitude.timeoffset, 0.0)
        self.assertEqual(self.m.altitude.std_factor, 0.001)
        self.assertEqual(self.m.altitude.std_offset, 0.3)

        self.assertEqual(self.m.usbl.format, 'NOC_nmea')
        self.assertEqual(self.m.usbl.filepath, 'nav/usbl/')
        self.assertEqual(self.m.usbl.filename, '')
        self.assertEqual(self.m.usbl.label, 13)
        self.assertEqual(self.m.usbl.timezone, 0)
        self.assertEqual(self.m.usbl.timeoffset, 0.0)
        self.assertEqual(self.m.usbl.std_factor, 0.01)
        self.assertEqual(self.m.usbl.std_offset, 2.0)

        self.assertEqual(self.m.tide.format, 'NOC_polpred')
        self.assertEqual(self.m.tide.filepath, 'tide/')
        self.assertEqual(self.m.tide.filename, 'dy_108_polpred_tide_10m.txt')
        self.assertEqual(self.m.tide.label, 0)
        self.assertEqual(self.m.tide.timezone, 0)
        self.assertEqual(self.m.tide.timeoffset, 0.0)
        self.assertEqual(self.m.tide.std_factor, 0.0)
        self.assertEqual(self.m.tide.std_offset, 0.0)

        self.assertEqual(self.m.image.format, 'biocam')
        self.assertEqual(self.m.image.cameras[0].name, 'cam61003146')
        self.assertEqual(self.m.image.cameras[0].origin, 'cam61003146')
        self.assertEqual(self.m.image.cameras[0].type, 'bayer_rggb')
        #self.assertEqual(self.m.image.cameras[0].bit_depth, 16) # this does not get loaded
        self.assertEqual(self.m.image.cameras[0].path, 'image')
        self.assertEqual(self.m.image.cameras[1].name, 'cam61004444')
        self.assertEqual(self.m.image.cameras[1].origin, 'cam61004444')
        self.assertEqual(self.m.image.cameras[1].type, 'grayscale')
        #self.assertEqual(self.m.image.cameras[1].bit_depth, 16) # this does not get loaded
        self.assertEqual(self.m.image.cameras[1].path, 'image')
        self.assertEqual(self.m.image.timezone, 0)
        self.assertEqual(self.m.image.timeoffset, 0.0)

class TestLoadMissionSSK1701FileFormat0(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= 'default_yaml/ts1/SSK17-01/legacy_vehicle_format_0/mission.yaml'
        self.m = Mission(path)

    def test_LoadMission(self):
        self.assertEqual(self.m.version, 1)

        self.assertEqual(self.m.origin.latitude, 26.674083)
        self.assertEqual(self.m.origin.longitude, 127.868054)
        self.assertEqual(self.m.origin.crs, 'wgs84')
        self.assertEqual(self.m.origin.date, '2017/08/17')

        self.assertEqual(self.m.velocity.format, 'phins')
        self.assertEqual(self.m.velocity.filepath, 'nav/phins/')
        self.assertEqual(self.m.velocity.filename, '20170817_phins.txt')
        self.assertEqual(self.m.velocity.label, 0)
        self.assertEqual(self.m.velocity.timezone, 0)
        self.assertEqual(self.m.velocity.timeoffset, 0.0)
        self.assertEqual(self.m.velocity.std_factor, 0)
        self.assertEqual(self.m.velocity.std_offset, 0)

        self.assertEqual(self.m.orientation.format, 'phins')
        self.assertEqual(self.m.orientation.filepath, 'nav/phins/' )
        self.assertEqual(self.m.orientation.filename, '20170817_phins.txt')
        self.assertEqual(self.m.orientation.label, 0)
        self.assertEqual(self.m.orientation.timezone, 0)
        self.assertEqual(self.m.orientation.timeoffset, 0.0)
        self.assertEqual(self.m.orientation.std_factor, 0)
        self.assertEqual(self.m.orientation.std_offset, 0)

        self.assertEqual(self.m.depth.format, 'phins')
        self.assertEqual(self.m.depth.filepath, 'nav/phins/')
        self.assertEqual(self.m.depth.filename, '20170817_phins.txt')
        self.assertEqual(self.m.depth.label, 0)
        self.assertEqual(self.m.depth.timezone, 0)
        self.assertEqual(self.m.depth.timeoffset, 0.0)
        self.assertEqual(self.m.depth.std_factor, 0)
        self.assertEqual(self.m.depth.std_offset, 0)

        self.assertEqual(self.m.altitude.format, 'phins')
        self.assertEqual(self.m.altitude.filepath, 'nav/phins/')
        self.assertEqual(self.m.altitude.filename, '20170817_phins.txt')
        self.assertEqual(self.m.altitude.label, 0)
        self.assertEqual(self.m.altitude.timezone, 0)
        self.assertEqual(self.m.altitude.timeoffset, 0.0)
        self.assertEqual(self.m.altitude.std_factor, 0)
        self.assertEqual(self.m.altitude.std_offset, 0)

        self.assertEqual(self.m.usbl.format, 'gaps')
        self.assertEqual(self.m.usbl.filepath, 'nav/gaps/')
        self.assertEqual(self.m.usbl.filename, '')
        self.assertEqual(self.m.usbl.label, 2)
        self.assertEqual(self.m.usbl.timezone, 0)
        self.assertEqual(self.m.usbl.timeoffset, 0.0)
        self.assertEqual(self.m.usbl.std_factor, 0)
        self.assertEqual(self.m.usbl.std_offset, 0)

        self.assertEqual(self.m.tide.format, '')
        self.assertEqual(self.m.tide.filepath, '')
        self.assertEqual(self.m.tide.filename, '')
        self.assertEqual(self.m.tide.label, 0)
        self.assertEqual(self.m.tide.timezone, 0)
        self.assertEqual(self.m.tide.timeoffset, 0.0)
        self.assertEqual(self.m.tide.std_factor, 0)
        self.assertEqual(self.m.tide.std_offset, 0)

        self.assertEqual(self.m.image.format, 'acfr_standard')
        self.assertEqual(self.m.image.cameras[0].name, 'camera1')
        self.assertEqual(self.m.image.cameras[0].origin, 'camera1')
        self.assertEqual(self.m.image.cameras[0].type, 'bayer_rggb')
        #self.assertEqual(self.m.image.cameras[0].bit_depth, 0) # this does not get loaded
        self.assertEqual(self.m.image.cameras[0].path, 'image/r20170817_041459_UG117_sesoko/i20170817_041459/')
        self.assertEqual(self.m.image.cameras[1].name, 'camera2')
        self.assertEqual(self.m.image.cameras[1].origin, 'camera2')
        self.assertEqual(self.m.image.cameras[1].type, 'bayer_rggb')
        #self.assertEqual(self.m.image.cameras[1].bit_depth, 0) # this does not get loaded
        self.assertEqual(self.m.image.cameras[1].path, 'image/r20170817_041459_UG117_sesoko/i20170817_041459/')
        self.assertEqual(self.m.image.timezone, 9)
        self.assertEqual(self.m.image.timeoffset, 0.0)

class TestLoadVehicleYK1723C(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= 'default_yaml/ae2000/YK17-23C/vehicle.yaml'
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

        self.assertEqual(self.v.depth.surge, 0.)
        self.assertEqual(self.v.depth.sway, 0)
        self.assertEqual(self.v.depth.heave, 0)
        self.assertEqual(self.v.depth.roll, 0)
        self.assertEqual(self.v.depth.pitch, 0)
        self.assertEqual(self.v.depth.yaw, 0)

        self.assertEqual(self.v.usbl.surge, 0.)
        self.assertEqual(self.v.usbl.sway, 0)
        self.assertEqual(self.v.usbl.heave, -0.289)
        self.assertEqual(self.v.usbl.roll, 0)
        self.assertEqual(self.v.usbl.pitch, 0)
        self.assertEqual(self.v.usbl.yaw, 0)

        self.assertEqual(self.v.camera1.surge, 0.262875)
        self.assertEqual(self.v.camera1.sway, 0.)
        self.assertEqual(self.v.camera1.heave, 0.5)
        self.assertEqual(self.v.camera1.roll, 0)
        self.assertEqual(self.v.camera1.pitch, 0)
        self.assertEqual(self.v.camera1.yaw, 0)

        self.assertEqual(self.v.camera2.surge, 0.012875)
        self.assertEqual(self.v.camera2.sway, 0.)
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

        self.assertEqual(self.v.chemical.surge, 0.4)
        self.assertEqual(self.v.chemical.sway, 0)
        self.assertEqual(self.v.chemical.heave, -0.5)
        self.assertEqual(self.v.chemical.roll, 0)
        self.assertEqual(self.v.chemical.pitch, 0)
        self.assertEqual(self.v.chemical.yaw, 0)


class TestLoadVehicleDY109(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= 'default_yaml/as6/DY109/vehicle.yaml'
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
        self.assertEqual(self.v.camera1.sway, 0.)
        self.assertEqual(self.v.camera1.heave, 0.327)
        self.assertEqual(self.v.camera1.roll, 0)
        self.assertEqual(self.v.camera1.pitch, -90)
        self.assertEqual(self.v.camera1.yaw, 0)

        self.assertEqual(self.v.camera2.surge, 1.104)
        self.assertEqual(self.v.camera2.sway, 0.)
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

        self.assertEqual(self.v.chemical.surge, 0)
        self.assertEqual(self.v.chemical.sway, 0)
        self.assertEqual(self.v.chemical.heave, 0)
        self.assertEqual(self.v.chemical.roll, 0)
        self.assertEqual(self.v.chemical.pitch, 0)
        self.assertEqual(self.v.chemical.yaw, 0)


class TestLoadVehicleSSK1701FileFormat0(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]
        path /= 'default_yaml/ts1/SSK17-01/legacy_vehicle_format_0/vehicle.yaml'
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
        self.assertEqual(self.v.ins.yaw, -45.0) # read form mission.yaml (headingoffset), not from vehicle.yaml!

        self.assertEqual(self.v.dvl.surge, -0.45)
        self.assertEqual(self.v.dvl.sway, 0)
        self.assertEqual(self.v.dvl.heave, 0.45)
        self.assertEqual(self.v.dvl.roll, 0)
        self.assertEqual(self.v.dvl.pitch, 0)
        self.assertEqual(self.v.dvl.yaw, -45.0) # read form mission.yaml (headingoffset), not from vehicle.yaml!

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

        self.assertEqual(self.v.chemical.surge, 0)
        self.assertEqual(self.v.chemical.sway, 0)
        self.assertEqual(self.v.chemical.heave, 0)
        self.assertEqual(self.v.chemical.roll, 0)
        self.assertEqual(self.v.chemical.pitch, 0)
        self.assertEqual(self.v.chemical.yaw, 0)