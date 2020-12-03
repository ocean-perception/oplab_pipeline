# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import unittest
import os
import math
from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.body_to_inertial import body_to_inertial
from oplab import Console
from auv_nav.tools.displayable_path import DisplayablePath
from auv_nav.tools.latlon_wgs84 import latlon_to_metres
from auv_nav.tools.latlon_wgs84 import metres_to_latlon


class TestTools(unittest.TestCase):
    def test_interpolate(self):
        x_query = 150.0
        x_lower = 100.0
        x_upper = 200.0
        y_lower = 100.0
        y_upper = 200.0
        y_query = interpolate(x_query, x_lower, x_upper, y_lower, y_upper)
        assert y_query == 150.0

    def test_body_to_inertial(self):
        x, y, z = body_to_inertial(0, 0, 0, 0, 0, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_console(self):
        Console.warn("This is a warning")
        Console.error("This is an error message")
        Console.info("This is an informative message")
        Console.banner()

        Console.get_username()
        Console.get_hostname()
        Console.get_date()
        Console.get_version()
        for i in range(1, 10):
            Console.progress(i, 10)

    def test_DisplayablePath(self):
        cwd = os.getcwd()
        DisplayablePath.show_tree(cwd)

    def test_latlon_wgs84(self):
        # Boldrewood
        lat_p = 50.936501
        lon_p = -1.404266

        # Highfield B35
        lat_ref = 50.936870
        lon_ref = -1.396295

        x, theta = latlon_to_metres(lat_p, lon_p, lat_ref, lon_ref)
        self.assertLess(x, 600.0)
        self.assertGreater(x, 500.0)
        self.assertLess(theta, 95.0)
        self.assertGreater(theta, 85.0)

        easting = x * math.sin(math.radians(theta))
        northing = x * math.cos(math.radians(theta))
        lat, lon = metres_to_latlon(lat_ref, lon_ref, easting, northing)

        self.assertAlmostEqual(lat, lat_p, places=2)
        self.assertAlmostEqual(lon, lon_p, places=2)


if __name__ == "__main__":
    unittest.main()
