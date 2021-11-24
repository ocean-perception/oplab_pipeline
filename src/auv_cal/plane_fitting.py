# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from math import atan2, pi

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.optimize import least_squares


class Plane:
    coeffs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    point = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def __init__(self, coeffs=None):
        if coeffs is not None:
            self.from_coeffs(coeffs)

    def from_coeffs(self, coeffs):
        self.coeffs = np.array(coeffs, dtype=np.float64)
        self.normal = self.coeffs[0:3]
        norm_normal = norm(self.normal)
        self.normal /= norm_normal
        self.coeffs /= norm_normal
        self.point = np.array([0.0, 1.0, 10.0], dtype=np.float64)
        self.point[0] = -(
            (
                self.point[1] * self.coeffs[1]
                + self.point[2] * self.coeffs[2]
                + self.coeffs[3]
            )
            / self.coeffs[0]
        )

    def distance(self, point):
        """Compute distance from point to plane"""
        d = np.abs(np.dot(self.normal, point) + self.coeffs[3])
        return d

    def residuals(self, coeffs, points):
        residuals = np.zeros(len(points))
        self.from_coeffs(coeffs)
        for i, p in enumerate(points):
            residuals[i] = self.distance(p)
        return residuals

    def fit(self, points, min_distance_inliers, verbose=True, output_inliers=True):
        """Fit plane to points

        Parameters
        ----------
        points : ndarray
            (n x 3) array containing points of point cloud
        min_distance_inliers : float
            Points with located within this distance from the plane are kept
        verbose : bool, optional
            Enable or disable verbose output. Default: True
        output_inliers : bool, optional
            Enable or disable outputting of inliers. Default: True

        Returns
        -------
        ndarray (float64)
            (vector of length 4) Coefficients of plane
        list of ndarray of length 3
            (list of ndarray vectors of length 3) Inlier points
        """

        # Coeffs: apex(x, y, z), axis(x, y, z) and theta
        coefficients = np.array([1, 0, 0, -1.5], dtype=np.float64)
        bounds = ([-1.0, -1.0, -1.0, -np.inf], [1.0, 1.0, 1.0, np.inf])
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

        inliers = None
        if output_inliers:
            inliers = []
            for p in points:
                if self.distance(p) < min_distance_inliers:
                    inliers.append(p)

        if verbose:
            print("Fitted plane with:")
            print("\t Coefficients:", self.coeffs)
            print("\t With", len(inliers), "inliers")
            pitch_offset_deg = atan2(self.coeffs[2], self.coeffs[0]) * 180 / pi
            print("\t Pitch offset from x-axis: ", pitch_offset_deg, "°")
            yaw_offset_deg = atan2(self.coeffs[1], self.coeffs[0]) * 180 / pi
            print("\t Yaw offset from x-axis:   ", yaw_offset_deg, "°")
        return self.coeffs, inliers

    def fit_non_robust(self, points):
        """Fit plane to points

        Parameters
        ----------
        points : list of ndarray
            Each array is a vector containing the x,y,z coordinates of 1 point

        Returns
        -------
        np.ndarray (float64)
            (vector of length 4) Coefficients of plane
        """

        axyz = np.ones((len(points), 4))
        axyz[:, :3] = points
        m = np.linalg.svd(axyz)[-1][-1, :]
        if m[0] < 0:
            m = m * (-1)
        self.from_coeffs(m)
        return self.coeffs

    def ray_intersection(self, ray_point, ray_vec):
        ray_vec /= norm(ray_vec)
        if np.dot(self.normal, ray_vec) == 0:
            return None
        num = np.dot(self.normal, self.point) - np.dot(self.normal, ray_point)
        den = np.dot(self.normal, ray_vec)
        t = num / den
        return ray_point + ray_vec * t

    def plot(self, cloud=None, points=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        yy, zz = np.mgrid[-5:5, 0:15]
        xx = (
            -self.coeffs[3] - self.coeffs[2] * zz - self.coeffs[1] * yy
        ) / self.coeffs[0]

        ax.plot_wireframe(xx, yy, zz)
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
        cam_origin : np.array
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
        np.array
            Triangulated pixels rays to the surface
        """
        points = []
        for i in range(0, image_width, 20):
            for j in range(0, image_height, 10):
                cam_ray = np.array([(cy - j) / fy, (i - cx) / fx, 1.0])
                cam_ray /= norm(cam_ray)
                p1, p2 = self.ray_intersection(cam_origin, cam_ray)
                if p1 is not None and p2 is not None:
                    k1 = norm(p1)
                    k2 = norm(p2)
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
