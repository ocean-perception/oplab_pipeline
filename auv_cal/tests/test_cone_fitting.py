import unittest
import os
import math
import numpy as np
from auv_cal.cone_fitting import Paraboloid


class TestConeFitting(unittest.TestCase):
    def test_fit_paraboloid (self):
        p = Paraboloid(1, 2, 3, 4, 5, -1)
        z = list()
        for x in range(-2, 3) :
            for y in range(-2, 3) :
                z.append((x, y, p.image(x, y)))
        q = Paraboloid()
        q.fit(* z)
        print("original   =", p.coef)
        print("estimation =", q.coef)	
        for i, j in zip(q.coef, p.coef):
            self.assertAlmostEqual(i, j)