# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import unittest
from auv_nav.sensors import OutputFormat
from auv_nav.sensors import BodyVelocity
from auv_nav.sensors import InertialVelocity
from auv_nav.sensors import Orientation
from auv_nav.sensors import Depth
from auv_nav.sensors import Altitude
from auv_nav.sensors import Usbl
from auv_nav.sensors import Camera
from auv_nav.sensors import Tide
from auv_nav.sensors import Other
from auv_nav.sensors import SyncedOrientationBodyVelocity


class TestOutputFormat(unittest.TestCase):
    def setUp(self):
        self.of = OutputFormat()

    def testComparison(self):
        other = OutputFormat()
        other.epoch_timestamp = 100
        self.of.epoch_timestamp = 10
        self.assertTrue(self.of < other)


class TestBodyVelocity(unittest.TestCase):
    def setUp(self):
        self.bv = BodyVelocity()

    def test_BodyVelocityFromAutosub(self):
        self.bv.clear()
        self.assertFalse(self.bv.valid())

        autosub_data = {
            "eTime": [1574950320],
            "Vnorth0": [10.0],  # mm/s
            "Veast0": [10.0],
            "Vdown0": [10.0],
            "Verr0": [0.01],
            "Verr0": [0.01],
            "Verr0": [0.01],
        }
        self.bv.from_autosub(autosub_data, 0)
        self.assertEqual(self.bv.x_velocity, -0.01, "incorrect forward speed")
        self.assertTrue(self.bv.valid())

    def test_BodyVelocityFromPhins(self):
        self.bv.clear()
        self.assertFalse(self.bv.valid())

        phins_data = ["SPEED_", "", "0.2", "0.03", "0.1", "", "083015.23"]
        self.bv.from_phins(phins_data)
        self.assertTrue(self.bv.valid)
        self.assertEqual(self.bv.x_velocity, 0.2, "incorrect forward speed")
        self.assertEqual(self.bv.y_velocity, -0.03, "incorrect forward speed")
        self.assertEqual(self.bv.z_velocity, -0.1, "incorrect forward speed")


class TestInertialVelocity(unittest.TestCase):
    def setUp(self):
        self.iv = InertialVelocity()

    def test_InertialVelocity(self):
        self.iv.clear()
        self.assertFalse(self.iv.valid())


class TestOrientation(unittest.TestCase):
    def setUp(self):
        self.ori = Orientation()

    def test_Orientation(self):
        self.ori.clear()
        self.assertFalse(self.ori.valid())


class TestDepth(unittest.TestCase):
    def setUp(self):
        self.depth = Depth()

    def test_Depth(self):
        self.depth.clear()
        self.assertFalse(self.depth.valid())


class TestAltitude(unittest.TestCase):
    def setUp(self):
        self.altitude = Altitude()

    def test_Altitude(self):
        self.altitude.clear()
        self.assertFalse(self.altitude.valid())


class TestUsbl(unittest.TestCase):
    def setUp(self):
        self.usbl = Usbl()

    def test_Usbl(self):
        self.usbl.clear()
        self.assertFalse(self.usbl.valid())


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.camera = Camera()

    def test_Camera(self):
        json_data = {"epoch_timestamp": 10, "filename": "test.jpg"}
        self.camera.from_json(json_data, "cam_name")
        self.assertEqual(self.camera.epoch_timestamp, 10)
        self.assertEqual(self.camera.filename, "test.jpg")


class TestTide(unittest.TestCase):
    def setUp(self):
        self.tide = Tide()

    def test_Tide(self):
        self.tide.clear()
        self.assertFalse(self.tide.valid())


class TestOther(unittest.TestCase):
    def setUp(self):
        self.other = Other()

    def test_Other(self):
        json_data = {"epoch_timestamp": 10, "data": "test"}
        self.other.from_json(json_data)
        self.assertEqual(self.other.epoch_timestamp, 10)


class TestSyncedOrientationBodyVelocity(unittest.TestCase):
    def setUp(self):
        self.sobv = SyncedOrientationBodyVelocity()

    def testComparison(self):
        other = SyncedOrientationBodyVelocity()
        other.epoch_timestamp = 100
        self.sobv.epoch_timestamp = 10
        self.assertTrue(self.sobv < other)


if __name__ == "__main__":
    unittest.main()
