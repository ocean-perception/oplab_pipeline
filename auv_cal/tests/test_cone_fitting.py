import unittest
import os
import math
import numpy as np
from auv_cal.cone_fitting import Paraboloid
from auv_cal.cone_fitting import Circle
from auv_cal.cone_fitting import CircularCone
from auv_cal.cone_fitting import rotation_matrix
from auv_cal.cone_fitting import generate_circle_by_angles


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

    def test_rotation_matrix(self):
        v = [3, 5, 0]
        axis = [4, 4, 1]
        theta = 1.2 

        res = np.dot(rotation_matrix(axis, theta), v)
        self.assertAlmostEqual(res[0], 2.74911638)
        self.assertAlmostEqual(res[1], 4.77180932)
        self.assertAlmostEqual(res[2], 1.91629719)

    def test_point_cone_distance(self):
        c = CircularCone()
        c.apex = np.array([0, 0, 0])
        c.axis = np.array([1, 0, 0])
        c.half_angle = math.radians(45.0)

        point = np.array([0, 0, 0])
        d = c.distanceTo(point)
        self.assertAlmostEqual(d, 0)

        point = np.array([0, 1, 0])
        d = c.distanceTo(point)
        self.assertAlmostEqual(d, math.sqrt(2.)/2.)

        c.axis = np.array([0, 0, 1])
        d = c.distanceTo(point)
        self.assertAlmostEqual(d, math.sqrt(2.)/2.)

        c.axis = np.array([0, 1, 0])
        d = c.distanceTo(point)
        self.assertAlmostEqual(d, math.sqrt(2.)/2.)

    def test_circle_fitting(self):
        # Generating circle
        r = 2.5               # Radius
        C = np.array([3,3,4])    # Center
        theta = 45/180*np.pi     # Azimuth
        phi   = -30/180*np.pi    # Zenith 

        t = np.linspace(0, 2*np.pi, 100)
        P_gen = generate_circle_by_angles(t, C, r, theta, phi)

        #-------------------------------------------------------------------------------
        # Cluster of points
        #-------------------------------------------------------------------------------
        t = np.linspace(-np.pi, -0.25*np.pi, 100)
        n = len(t)
        P = generate_circle_by_angles(t, C, r, theta, phi)

        # Add some random noise to the points
        P += np.random.normal(size=P.shape) * 0.01

        c = Circle()
        centre, radius = c.fit(P)
        print('centre:', centre)
        print('radius:', radius)
        self.assertAlmostEqual(radius, 2.5, places=0)
        self.assertAlmostEqual(centre[0], C[0], places=2)
        self.assertAlmostEqual(centre[1], C[1], places=2)
        self.assertAlmostEqual(centre[2], C[2], places=2)
