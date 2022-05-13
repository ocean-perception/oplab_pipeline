# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path

import cv2
import numpy as np

from .console import Console  # noqa


class MonoCamera:
    """Monocular camera using OpenCV functions and parameters.
    Reads and writes calibration.yaml files
    """

    def __init__(self, filename=None):
        self.K = np.zeros((3, 3))
        self.d = np.zeros((5, 1))
        self.R = np.eye(3)
        self.P = np.zeros((3, 4))
        self.image_width = 0
        self.image_height = 0
        self.name = ""

        if filename is not None:
            filename = Path(filename)
            fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
            self.from_node(fs)

    @property
    def aspect_ratio(self):
        return float(self.image_width) / float(self.image_height)

    @property
    def rectification_maps(self):
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.intrinsics,
            self.distortion,
            self.R,
            self.P,
            self.size,
            cv2.CV_32FC1,
        )
        return mapx, mapy

    @property
    def size(self):
        return (self.image_width, self.image_height)

    @size.setter
    def size(self, s):
        (self.image_width, self.image_height) = s

    def from_node(self, fs):
        self.name = fs.getNode("camera_name").string()
        self.image_width = int(fs.getNode("image_width").real())
        self.image_height = int(fs.getNode("image_height").real())
        self.K = fs.getNode("camera_matrix").mat()
        self.d = fs.getNode("distortion_coefficients").mat()
        self.R = fs.getNode("rectification_matrix").mat()
        self.P = fs.getNode("projection_matrix").mat()

    def to_str(
        self,
        num_images=None,
        error=None,
        write_metadata=True,
        write_header=True,
        nest=False,
    ):
        msg = ""
        t = ""
        if write_header:
            msg = "%YAML:1.0\n"
        if nest:
            t = "  "
        msg += (
            t
            + "image_width: "
            + str(self.size[0])
            + "\n"
            + t
            + "image_height: "
            + str(self.size[1])
            + "\n"
            + t
            + "camera_name: "
            + self.name
            + "\n"
            + t
            + "camera_matrix: !!opencv-matrix\n"
            + t
            + "  rows: 3\n"
            + t
            + "  cols: 3\n"
            + t
            + "  dt: d\n"
            + t
            + "  data: ["
            + ", ".join(["%8f" % i for i in self.K.reshape(1, 9)[0]])
            + "]\n"
            + t
            + "distortion_model: "
            + ("rational_polynomial" if self.d.size > 5 else "plumb_bob")
            + "\n"
            + t
            + "distortion_coefficients: !!opencv-matrix\n"
            + t
            + "  rows: 1\n"
            + t
            + "  cols: 5\n"
            + t
            + "  dt: d\n"
            + t
            + "  data: ["
            + ", ".join(["%8f" % self.d[i, 0] for i in range(self.d.shape[0])])
            + "]\n"
            + t
            + "rectification_matrix: !!opencv-matrix\n"
            + t
            + "  rows: 3\n"
            + t
            + "  cols: 3\n"
            + t
            + "  dt: d\n"
            + t
            + "  data: ["
            + ", ".join(["%8f" % i for i in self.R.reshape(1, 9)[0]])
            + "]\n"
            + t
            + "projection_matrix: !!opencv-matrix\n"
            + t
            + "  rows: 3\n"
            + t
            + "  cols: 4\n"
            + t
            + "  dt: d\n"
            + t
            + "  data: ["
            + ", ".join(["%8f" % i for i in self.P.reshape(1, 12)[0]])
            + "]\n"
        )
        if write_metadata:
            msg += Console.write_metadata()
        if num_images is not None:
            msg += t + "number_of_images: " + str(num_images) + "\n"
        if error is not None:
            msg += t + "avg_reprojection_error: " + str(error) + "\n"
        return msg

    def return_valid(self, p):
        px = p[0]
        py = p[1]
        v = px > 0 and px < self.image_width and py > 0 and py < self.image_height
        if v:
            return p
        else:
            return None

    def distort_point(self, p):
        fx = self.K[0, 0]
        cx = self.K[0, 2]
        fy = self.K[1, 1]
        cy = self.K[1, 2]
        px = p[0]
        py = p[1]
        ztemp = np.array([0, 0, 0], dtype="float32")
        p_unif = np.array([[[(px - cx) / fx, (py - cy) / fy, 1]]], dtype=np.float)
        out_p = []
        out_p = cv2.projectPoints(p_unif, ztemp, ztemp, self.K, self.d)[0][0][0]
        return self.return_valid(out_p)

    def undistort_point(self, p):
        # Undistorts points
        p = np.array([p], dtype=np.float).reshape(1, 2)
        # If matrix P is identity or omitted, dst will contain
        # normalized point coordinates
        dst = cv2.undistortPoints(p, self.K, self.d)[0][0]
        px = dst[0]
        py = dst[1]

        # To normalize them again, we need a new camera matrix as if
        # no distortion was present
        alpha = -1
        self.size = (self.image_width, self.image_height)
        cv2.getOptimalNewCameraMatrix(self.K, self.d, self.size, alpha)

        fx = self.K[0, 0]
        cx = self.K[0, 2]
        fy = self.K[1, 1]
        cy = self.K[1, 2]

        p = [(px * fx + cx), (py * fy + cy)]
        return self.return_valid(p)

    def undistort_and_rectify_point(self, p):
        p_und = self.undistort_point(p)
        if p_und is None:
            return None
        p_unif = np.array([p_und[0], p_und[1], 1], dtype=np.float).reshape(3, 1)
        p_und_rec = self.R @ p_unif
        p_und_rec = p_und_rec[0:2, 0]
        return self.return_valid(p_und_rec)

    def unrectify_and_distort_point(self, p):
        p_unif = np.array([p[0], p[1], 1], dtype=np.float).reshape(3, 1)
        p_unrec = self.R.T @ p_unif
        return self.distort_point(p_unrec[0:2])


class StereoCamera:
    """Stereo camera model using OpenCV functions and parameters.
    Reads and writes calibration yaml files
    """

    def __init__(self, filename=None, left=None, right=None):
        self.left = MonoCamera(left)
        self.right = MonoCamera(right)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.E = np.zeros((3, 3))
        self.F = np.zeros((3, 3))

        if filename is not None:
            filename = Path(filename)
            fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
            self.left.from_node(fs.getNode("left"))
            self.right.from_node(fs.getNode("right"))
            self.from_node(fs.getNode("extrinsics"))

        self.different_resolution = False
        self.different_aspect_ratio = False
        if (
            self.left.image_width != self.right.image_width
            or self.left.image_height != self.right.image_height
        ):
            self.different_resolution = True
        if self.left.aspect_ratio != self.right.aspect_ratio:
            self.different_aspect_ratio = True

    def from_node(self, fs):
        self.R = fs.getNode("rotation_matrix").mat()
        self.t = fs.getNode("translation_vector").mat()
        if not fs.getNode("fundamental_matrix").empty():
            self.F = fs.getNode("fundamental_matrix").mat()
            self.E = fs.getNode("essential_matrix").mat()

    def to_str(self):
        msg = "%YAML:1.0\n"
        msg += "left:\n"
        msg += self.left.to_str(write_metadata=False, write_header=False, nest=True)
        msg += "right:\n"
        msg += self.right.to_str(write_metadata=False, write_header=False, nest=True)
        msg += (
            "extrinsics:"
            + "  rotation_matrix:\n"
            + "    rows: 3\n"
            + "    cols: 3\n"
            + "    data: ["
            + ", ".join(["%8f" % i for i in self.R.reshape(1, 9)[0]])
            + "]\n"
            + "  translation_vector:\n"
            + "    rows: 1\n"
            + "    cols: 3\n"
            + "    data: ["
            + ", ".join(["%8f" % self.t[i, 0] for i in range(self.t.shape[0])])
            + "]\n"
            + "  fundamental_matrix:\n"
            + "    rows: 3\n"
            + "    cols: 3\n"
            + "    data: ["
            + ", ".join(["%8f" % i for i in self.F.reshape(1, 9)[0]])
            + "]\n"
            + "  essential_matrix:\n"
            + "    rows: 3\n"
            + "    cols: 3\n"
            + "    data: ["
            + ", ".join(["%8f" % i for i in self.E.reshape(1, 9)[0]])
            + "]\n"
        )
        msg += Console.write_metadata()
        return msg

    def triangulate_point(self, left_uv, right_uv):
        """Point pair triangulation from
        least squares solution."""
        M = np.zeros((6, 6))
        M[:3, :4] = self.left.P
        M[3:, :4] = self.right.P
        M[:3, 4] = -np.array([left_uv[0], left_uv[1], 1.0])
        M[3:, 5] = -np.array([right_uv[0], right_uv[1], 1.0])
        U, S, V = np.linalg.svd(M)
        X = V[-1, :3] / V[-1, 3]
        return X

    def triangulate_point_undistorted(self, left_uv, right_uv):
        p1 = self.left.undistort_and_rectify_point(left_uv)
        p2 = self.right.undistort_and_rectify_point(right_uv)
        return self.triangulate_point(p1, p2)

    def project_point(self, point3d):
        """Projects a 3D point into the rectified frame
        point3d a list of three elements [x, y, z]
        """
        X = np.ones((4, 1), dtype=np.float)
        X[:3, 0] = point3d
        left_uv = self.left.P @ X
        left_uv /= left_uv[2]
        left_uv = left_uv[0:2, 0]
        right_uv = self.right.P @ X
        right_uv /= right_uv[2]
        right_uv = right_uv[0:2, 0]
        return (
            self.left.return_valid(left_uv),
            self.right.return_valid(right_uv),
        )

    def project_point_undistorted(self, point3d):
        """Projects a 3D point into the undistorted frame
        point3d a list of three elements [x, y, z]
        """
        p1, p2 = self.project_point(point3d)
        if p1 is None:
            return None, None
        p1d = self.left.unrectify_and_distort_point(p1)
        p2d = self.right.unrectify_and_distort_point(p2)
        return p1d, p2d
