import numpy as np
from pathlib import Path
import cv2
from oplab.console import Console  # noqa


class MonoCamera():
    def __init__(self, filename=None):
        self.K = np.zeros((3, 3))
        self.d = np.zeros((5, 1))
        self.R = np.eye(3)
        self.P = np.zeros((3, 4))
        self.image_width = 0
        self.image_height = 0
        self.name = ''

        if filename is not None:
            filename = Path(filename)
            fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
            self.from_node(fs)

    @property
    def rectification_maps(self):
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.intrinsics,
            self.distortion,
            self.R,
            self.P,
            self.size,
            cv2.CV_32FC1)
        return mapx, mapy

    @property
    def size(self):
        return (self.image_width, self.image_height)

    @size.setter
    def size(self, s):
        (self.image_width, self.image_height) = s

    def from_node(self, fs):
        self.name = fs.getNode('camera_name').string()
        self.image_width = int(fs.getNode('image_width').real())
        self.image_height = int(fs.getNode('image_height').real())
        self.K = fs.getNode('camera_matrix').mat()
        self.d = fs.getNode('distortion_coefficients').mat()
        self.R = fs.getNode('rectification_matrix').mat()
        self.P = fs.getNode('projection_matrix').mat()

    def to_str(self, num_images=None, error=None,
               write_metadata=True, write_header=True, nest=False):
        msg = ''
        t = ''
        if write_header:
            msg = "%YAML:1.0\n"
        if nest:
            t = '  '
        msg += (
            t + "image_width: " + str(self.size[0]) + "\n"
            + t + "image_height: " + str(self.size[1]) + "\n"
            + t + "camera_name: " + self.name + "\n"
            + t + "camera_matrix: !!opencv-matrix\n"
            + t + "  rows: 3\n" + t + "  cols: 3\n" + t
            + "  dt: d\n" + t + "  data: [" + ", ".join(
                ["%8f" % i for i in self.K.reshape(1, 9)[0]]) + "]\n"
            + t + "distortion_model: "
            + ("rational_polynomial" if self.d.size > 5 else "plumb_bob")
            + "\n" + t + "distortion_coefficients: !!opencv-matrix\n"
            + t + "  rows: 1\n" + t + "  cols: 5\n"
            + t + "  dt: d\n" + t + "  data: [" + ", ".join(
                ["%8f" % self.d[i, 0] for i in range(self.d.shape[0])])
            + "]\n" + t + "rectification_matrix: !!opencv-matrix\n"
            + t + "  rows: 3\n" + t + "  cols: 3\n"
            + t + "  dt: d\n" + t + "  data: [" + ", ".join(
                ["%8f" % i for i in self.R.reshape(1, 9)[0]])
            + "]\n" + t + "projection_matrix: !!opencv-matrix\n"
            + t + "  rows: 3\n" + t + "  cols: 4\n"
            + t + "  dt: d\n" + t + "  data: [" + ", ".join(
                ["%8f" % i for i in self.P.reshape(1, 12)[0]])
            + "]\n")
        if write_metadata:
            msg += Console.write_metadata()
        if num_images is not None:
            msg += t + "number_of_images: " + str(num_images) + "\n"
        if error is not None:
            msg += t + "avg_reprojection_error: " + str(error) + "\n"
        return msg

    def distort_point(self, p):
        fx = self.K[0, 0]
        cx = self.K[0, 2]
        fy = self.K[1, 1]
        cy = self.K[1, 2]
        px = p[0]
        py = p[1]
        ztemp = np.array([0, 0, 0], dtype='float32')
        p_unif = np.array([[[(px-cx)/fx, (py-cy)/fy, 1]]], dtype=np.float)
        out_p = []
        out_p = cv2.projectPoints(
            p_unif, ztemp, ztemp, self.K, self.d)[0][0][0]
        return out_p

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

        return [(px*fx+cx), (py*fy+cy)]

    def undistort_and_rectify_point(self, p):
        p_und = self.undistort_point(self, p)
        p_unif = np.array([p_und[0], p_und[1], 1],
                          dtype=np.float).reshape(3, 1)
        p_und_rec = self.R @ p_unif
        p_und_rec = p_und_rec[0:2, 0]
        return p_und_rec

    def unrectify_and_distort_point(self, p):
        p_unif = np.array([p[0], p[1], 1], dtype=np.float).reshape(3, 1)
        p_unrec = self.R.T @ p_unif
        return self.distort_point(self, p_unrec[0:2])


class StereoCamera():
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
            self.left.from_node(fs.getNode('left'))
            self.right.from_node(fs.getNode('right'))
            self.from_node(fs.getNode('extrinsics'))

    def from_node(self, fs):
        self.R = fs.getNode('rotation_matrix').mat()
        self.t = fs.getNode('translation_vector').mat()
        if not fs.getNode('fundamental_matrix').empty():
            self.F = fs.getNode('fundamental_matrix').mat()
            self.E = fs.getNode('essential_matrix').mat()

    def to_str(self):
        msg = "%YAML:1.0\n"
        msg += 'left:\n'
        msg += self.left.to_str(write_metadata=False,
                                write_header=False, nest=True)
        msg += 'right:\n'
        msg += self.right.to_str(write_metadata=False,
                                 write_header=False, nest=True)
        msg += (
            "extrinsics:"
            + "  rotation_matrix:\n" + "    rows: 3\n"
            + "    cols: 3\n" + "    data: [" + ", ".join(
                ["%8f" % i for i in self.R.reshape(1, 9)[0]]) + "]\n"
            + "  translation_vector:\n" + "    rows: 1\n"
            + "    cols: 3\n" + "    data: [" + ", ".join(
                ["%8f" % self.t[i, 0] for i in range(self.t.shape[0])]) + "]\n"
            + "  fundamental_matrix:\n" + "    rows: 3\n"
            + "    cols: 3\n" + "    data: [" + ", ".join(
                ["%8f" % i for i in self.F.reshape(1, 9)[0]]) + "]\n"
            + "  essential_matrix:\n" + "    rows: 3\n"
            + "    cols: 3\n" + "    data: [" + ", ".join(
                ["%8f" % i for i in self.E.reshape(1, 9)[0]]) + "]\n")
        msg += Console.write_metadata()
        return msg

    def triangulate_point(self, left_uv, right_uv):
        """Find 3D coordinate using SVD triangulation

        Implements a linear triangulation method to find a 3D
        point. For example, see Hartley & Zisserman section 12.2
        (p.312).
        """
        A = []
        A.append(float(left_uv[0])*self.left.P[2, :] - self.left.P[0, :])
        A.append(float(left_uv[1])*self.left.P[2, :] - self.left.P[1, :])
        A.append(float(right_uv[0])*self.right.P[2, :] - self.right.P[0, :])
        A.append(float(right_uv[1])*self.right.P[2, :] - self.right.P[1, :])
        A = np.array(A)
        u, s, vt = np.linalg.svd(A)
        X = vt[-1, 0:3]/vt[-1, 3]  # normalize
        return X

    def project_point(self, point3d):
        X = np.ones((4, 1), dtype=np.float)
        X[:3, 0] = point3d
        left_uv = self.left.P @ X
        left_uv /= left_uv[2]
        left_uv = left_uv[0:2, 0]
        right_uv = self.right.P @ X
        right_uv /= right_uv[2]
        right_uv = right_uv[0:2, 0]
        return left_uv, right_uv
