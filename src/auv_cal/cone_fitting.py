# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa
from numpy.linalg import norm
from scipy.optimize import least_squares


def rotation_matrix(axis, theta):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def build_matrix(v1, v2, v3):
    """Returns the rotation matrix associated with the vector basis provided

    Parameters
    ----------
    v1 : np.ndarray
        Three-dimensional vector
    v2 : np.ndarray
        Three-dimensional vector
    v3 : np.ndarray
        Three-dimensional vector

    Returns
    np.ndarray
        3x3 Rotation matrix
    -------
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v3 = np.asarray(v3)
    v1 = v1 / norm(v1)
    v2 = v2 / norm(v2)
    v3 = v3 / norm(v3)
    return np.array(
        [[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]]
    )


class CircularCone:
    apex = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    theta = 0.0

    def __init__(self, coeffs=None):
        if coeffs is not None:
            self.from_coeffs(coeffs)

    def from_coeffs(self, coeffs):
        coeffs = np.array(coeffs, dtype=np.float64)
        self.apex = np.array(coeffs[0:3])
        self.axis = np.array(coeffs[3:6])
        self.theta = coeffs[6]
        self.axis /= np.linalg.norm(self.axis)
        if self.axis[0] < 0:
            self.axis *= -1.0

    def distance(self, point):
        """Compute distance from point to modelled cone

        The distance from a point :math:`p_i` to a cone with apex :math:`Ap`
        and basis :math:`[\\vec{u}, \\vec{n}_1, \\vec{n}_2]`, where
        :math:`\\vec{u}` is the cone asis is defined by:

        .. math::
            (p_i - Ap) = \\begin{bmatrix} \\vec{u}, \\vec{n}_1, \\vec{n}_2
            \\end{bmatrix} \\begin{bmatrix} \\alpha \\\\ \\beta \\\\ \\gamma
            \\end{bmatrix}

        being :math:`\\gamma` the signed distance between the point and the
        cone.
        """
        w = point - self.apex
        if norm(w) == 0:
            # The point is in the apex
            return 0
        w = w / norm(w)
        v = self.axis
        wxv = np.cross(w, v)
        if norm(wxv) == 0:
            # The point lies in the axis
            d = norm(point - self.apex)
            return d * np.cos(self.theta)
        n1 = wxv / norm(wxv)
        R = rotation_matrix(n1, -self.theta)
        u = np.dot(R, v)
        uxn1 = np.cross(u, n1)
        n2 = uxn1 / norm(uxn1)
        basis = build_matrix(u, n1, n2)
        abc = basis.T @ w
        signed_distance = abc[2]
        return np.abs(signed_distance)

    def residuals(self, coeffs, points):
        residuals = np.zeros(len(points))
        self.from_coeffs(coeffs)
        for i, p in enumerate(points):
            residuals[i] = self.distance(p)
        return residuals

    def fit(self, points, verbose=True):
        # Coeffs: apex(x, y, z), axis(x, y, z) and theta
        coefficients = np.array([0, 0, 0, 0, 0, 1, np.pi / 2], dtype=float)
        bounds = (
            [-2.0, -0.2, -0.5, -1.0, -1.0, -1.0, 0.0],
            [2.0, 0.2, 0.5, 1.0, 1.0, 1.0, np.pi / 2.0],
        )

        if verbose:
            verb_level = 2
        else:
            verb_level = 0

        ret = least_squares(
            self.residuals,
            coefficients,
            bounds=bounds,
            args=([points]),
            ftol=None,
            xtol=1e-9,
            loss="soft_l1",
            verbose=verb_level,
            max_nfev=5000,
        )
        self.from_coeffs(ret.x)
        print("Fitted cone with:")
        print("\t Apex:", self.apex)
        print("\t Axis:", self.axis)
        print("\t Half angle:", self.theta)

    def ray_intersection(self, ray_point, ray_vec):
        # Source http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/  # noqa
        cos2 = np.cos(self.theta) * np.cos(self.theta)
        co = ray_point - self.apex

        ray_vec /= np.linalg.norm(ray_vec)

        d_v = np.dot(ray_vec, self.axis)
        co_v = np.dot(co, self.axis)
        d_co = np.dot(ray_vec, co)
        co_co = np.dot(co, co)

        a = d_v**2 - cos2
        b = 2 * (d_v * co_v - d_co * cos2)
        c = co_v**2 - co_co * cos2
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None, None
        elif discriminant == 0:
            t = -b / (2 * a)
            point = ray_point + t * ray_vec
            return point, None
        elif discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            point1 = ray_point + t1 * ray_vec
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)
            point2 = ray_point + t2 * ray_vec
            return point1, point2

    def plot(self, length=10, cloud=None, points=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        R0 = 0
        R1 = length * np.tan(self.theta)
        # unit vector in direction of axis
        v = self.axis
        # make some vector not in the same direction as v
        not_v = np.array([1, 1, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        # make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # normalize n1
        n1 /= np.linalg.norm(n1)
        # make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)

        # Rot matrix
        R = build_matrix(v, n1, n2)

        # surface ranges over t from 0 to length of axis and 0 to 2*pi
        n = 800
        t = np.linspace(0, length, n)
        theta = np.linspace(0, 2 * np.pi, n)
        # use meshgrid to make 2d arrays
        t, theta = np.meshgrid(t, theta)
        R = np.linspace(R0, R1, n)
        # generate coordinates for surface
        X, Y, Z = [
            self.apex[i]
            + v[i] * t
            + R * np.sin(theta) * n1[i]
            + R * np.cos(theta) * n2[i]
            for i in [0, 1, 2]
        ]
        ax.plot_wireframe(X, Y, Z)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.invert_zaxis()
        if cloud is not None:
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c="red")
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="green")
        plt.show()

    def image_triangulation(
        self, cam_origin, image_width, image_height, fx, fy, cx, cy
    ):
        """Debug function to draw an entire image intersection with the fitted
        cone

        Parameters
        ----------
        cam_origin : np.ndarray
            Camera origin
        image_width : int
            Width of the image in pixels
        image_height : int
            Heignt of the image in pixels
        fx : float
        fy : float
        cx : float
        cy : float

        Returns
        -------
        np.ndarray
            Triangulated pixels rays to the surface
        """
        points = []
        for i in range(0, image_width, 20):
            for j in range(0, image_height, 10):
                cam_ray = np.array([(cy - j) / fy, (i - cx) / fx, 1.0])
                cam_ray /= np.linalg.norm(cam_ray)
                p1, p2 = self.ray_intersection(cam_origin, cam_ray)
                if p1 is not None and p2 is not None:
                    k1 = np.linalg.norm(p1)
                    k2 = np.linalg.norm(p2)
                    if k1 > k2 and k1 < 20.0 and p1[2] > 0.0 and p1[2] < 20.0:
                        points.append(p1)
                    elif k2 < 20.0 and p2[2] > 0.0 and p2[2] < 20.0:
                        points.append(p2)
                else:
                    if p1 is not None:
                        if p1[2] > 0.0 and p1[2] < 20.0:
                            points.append(p1)
                    elif p2 is not None:
                        if p2[2] > 0.0 and p2[2] < 20.0:
                            points.append(p2)
        return np.array(points)
