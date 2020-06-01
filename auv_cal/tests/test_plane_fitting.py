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
import numpy as np
from unittest.mock import patch
from auv_cal.plane_fitting import Plane

fit_points = np.array(
    [
        [-1.35292, -6.46749, 12.57575],
        [-1.34562, -5.09195, 11.92035],
        [-1.46393, 4.62872, 8.63609],
        [-1.46479, -4.10360, 7.33406],
        [-1.45952, 2.29114, 5.88340],
        [-1.29415, -3.36200, 13.33667],
        [-1.34015, -2.42965, 10.89580],
        [-1.38715, -6.13391, 12.82764],
        [-1.45565, -1.94665, 6.03601],
        [-1.50653, -2.89860, 6.32823],
        [-1.44928, 0.75775, 7.31810],
        [-1.39079, 2.47775, 10.64742],
        [-1.46938, -3.04145, 6.17961],
        [-1.39957, -1.00616, 8.45533],
        [-1.37203, -2.61745, 10.27539],
        [-1.31279, -1.99647, 14.74841],
        [-1.44082, -1.24051, 7.84737],
        [-1.35129, 3.85482, 13.73590],
        [-1.32345, -0.34059, 13.78970],
        [-1.41388, 1.05672, 7.63525],
        [-1.44007, 2.14432, 8.08693],
        [-1.40560, 0.13610, 9.07367],
        [-1.42701, 2.09095, 8.40117],
        [-1.41056, 0.42012, 9.39658],
        [-1.33025, 5.45814, 14.59224],
        [-1.43639, 4.39703, 9.03938],
        [-1.34129, -1.49182, 12.31207],
        [-1.47170, 2.39499, 5.58877],
        [-1.38624, -1.60548, 9.84260],
        [-1.28817, 4.74794, 14.85527],
        [-1.38331, -3.06664, 9.86038],
        [-1.43759, 1.74048, 8.62229],
        [-1.48691, -3.41055, 6.33874],
        [-1.42086, -2.90020, 7.83595],
        [-1.40738, -0.80426, 7.90800],
        [-1.42859, 5.03922, 9.76992],
        [-1.48634, -3.07216, 5.58952],
        [-1.45261, -1.01513, 6.90907],
        [-1.38489, 5.77858, 14.30126],
        [-1.48529, -1.90349, 5.71702],
        [-1.31858, 0.53275, 13.23734],
        [-1.40689, 0.27441, 7.37660],
        [-1.45094, -0.46826, 6.88955],
        [-1.44304, -1.46464, 7.34428],
        [-1.44625, -1.54639, 7.11190],
        [-1.39295, 0.55372, 8.47721],
        [-1.32614, 4.78954, 14.78375],
        [-1.32915, 1.72572, 12.38124],
        [-1.43647, 2.34614, 8.97639],
        [-1.42988, 0.90346, 8.35417],
        [-1.46668, 2.01715, 6.00727],
        [-1.34223, -4.72912, 11.96327],
        [-1.33287, -2.54866, 13.76460],
        [-1.44201, -2.10486, 7.82797],
        [-1.34262, -2.07976, 12.46099],
        [-1.32351, 4.24009, 14.56942],
        [-1.41931, -2.56828, 7.16972],
        [-1.38570, -6.17525, 11.81367],
        [-1.37186, 5.07133, 13.78215],
        [-1.39605, 8.24112, 14.70244],
        [-1.42994, -3.99940, 8.15429],
        [-1.45284, -4.38974, 8.98742],
        [-1.31768, -5.06777, 13.93658],
        [-1.43516, 1.29999, 6.94971],
        [-1.93259, 6.41687, 11.18917],
        [-1.37221, 6.33847, 13.76111],
        [-1.49383, -2.44321, 6.00687],
        [-1.41119, -4.89676, 9.83043],
        [-1.30333, -0.41119, 13.25385],
    ]
)


class TestPlaneFitting(unittest.TestCase):
    def test_distance(self):
        p = Plane([1, 0, 0, 0])  # x = 0

        point = np.array([0, 0, 0])
        d = p.distance(point)
        self.assertAlmostEqual(d, 0)

        point = np.array([0, 1, 0])
        d = p.distance(point)
        self.assertAlmostEqual(d, 0.0)

        point = np.array([1, 0, 1])
        d = p.distance(point)
        self.assertAlmostEqual(d, 1.0)

        point = np.array([1, 1, 0])
        d = p.distance(point)
        self.assertAlmostEqual(d, 1.0)

        point = np.array([2, 2, 0])
        d = p.distance(point)
        self.assertAlmostEqual(d, 2.0)

        point = np.array([2, 0, 2])
        d = p.distance(point)
        self.assertAlmostEqual(d, 2.0)

        point = np.array([2, 2, 0])
        d = p.distance(point)
        self.assertAlmostEqual(d, 2.0)

        p = Plane([1, -1, 0, 0])  # x = y
        d = p.distance(point)
        self.assertAlmostEqual(d, 0.0)

        p = Plane([1, 0, 0, -10])  # x = 10
        point = np.array([1, 0, 0])
        d = p.distance(point)
        self.assertAlmostEqual(d, 9.0)

    def test_fitting(self):
        coeffs = [1, 0, 0, 1.5]
        p = Plane(coeffs)
        m, _ = p.fit(fit_points, 0.1)
        self.assertAlmostEqual(m[0], 0.99, places=1)
        self.assertAlmostEqual(m[1], 0.0, places=1)
        self.assertAlmostEqual(m[2], 0.0, places=1)
        self.assertAlmostEqual(m[3], 1.58, places=1)

    @patch("matplotlib.pyplot.figure")
    def test_plot(self, mock_fig):
        coeffs = [1, 0, 0, -1.5849]
        p = Plane(coeffs)
        p.plot()
        mock_fig.assert_called()
