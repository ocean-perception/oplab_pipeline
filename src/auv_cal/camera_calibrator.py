# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import json
import math
from pathlib import Path

import cv2
import joblib
import numpy as np
import numpy.linalg

from auv_nav.parsers.parse_acfr_images import acfr_timestamp_from_filename
from auv_nav.parsers.parse_biocam_images import biocam_timestamp_from_filename
from oplab import Console


# Supported calibration patterns
class Patterns:
    Chessboard, Circles, ACircles = list(range(3))


class CalibrationException(Exception):
    pass


# TODO: Make pattern per-board?


class ChessboardInfo(object):
    def __init__(self, n_cols=0, n_rows=0, dim=0.0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.dim = dim

    def tolist(self):
        return (self.n_cols, self.n_rows, self.dim)

    def fromlist(self, d):
        self.n_cols = d[0]
        self.n_rows = d[1]
        self.dim = d[2]


# Make all private!!!!!


def lmin(seq1, seq2):
    """Pairwise minimum of two sequences"""
    return [min(a, b) for (a, b) in zip(seq1, seq2)]


def lmax(seq1, seq2):
    """Pairwise maximum of two sequences"""
    return [max(a, b) for (a, b) in zip(seq1, seq2)]


def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def _get_outside_corners(corners, board):
    """
    Return the four corners of the board as a whole, as
    (up_left, up_right, down_right, down_left).
    """
    xdim = board.n_cols
    ydim = board.n_rows

    if corners.shape[1] * corners.shape[0] != xdim * ydim:
        raise Exception(
            "Invalid number of corners! %d corners. X: %d, Y: %d"
            % (corners.shape[1] * corners.shape[0], xdim, ydim)
        )

    up_left = corners[0, 0]
    up_right = corners[xdim - 1, 0]
    down_right = corners[-1, 0]
    down_left = corners[-xdim, 0]

    return (up_left, up_right, down_right, down_left)


def _get_skew(corners, board):
    """
    Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from
    90 degrees.
    """
    # TODO Using three nearby interior corners might be more robust,
    # outside corners occasionally
    # get mis-detected
    up_left, up_right, down_right, _ = _get_outside_corners(corners, board)

    def angle(a, b, c):
        """
        Return angle between lines ab, bc
        """
        ab = a - b
        cb = c - b
        return math.acos(
            numpy.dot(ab, cb) / (numpy.linalg.norm(ab) * numpy.linalg.norm(cb))
        )

    skew = min(1.0, 2.0 * abs((math.pi / 2.0) - angle(up_left, up_right, down_right)))
    return skew


def _get_area(corners, board):
    """
    Get 2d image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral,
    and the area computed as
    |p X q|/2; see http://mathworld.wolfram.com/Quadrilateral.html.
    """
    (up_left, up_right, down_right, down_left) = _get_outside_corners(corners, board)
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0] * q[1] - p[1] * q[0]) / 2.0


def _get_corners(img, board, refine=True, checkerboard_flags=0):
    """
    Get corners for a particular chessboard for an image
    """
    h = img.shape[0]
    w = img.shape[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mono = img
    (ok, corners) = cv2.findChessboardCorners(
        mono,
        (board.n_cols, board.n_rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | checkerboard_flags,
    )
    if not ok:
        return (ok, corners)

    # If any corners are within BORDER pixels of the screen edge, reject the
    # detection by setting ok to false
    # NOTE: This may cause problems with very low-resolution cameras, where
    # 8 pixels is a non-negligible fraction
    # of the image size. See http://answers.ros.org/question/3155/how-can-i-calibrate-low-resolution-cameras # noqa
    BORDER = 8
    if not all(
        [
            (BORDER < corners[i, 0, 0] < (w - BORDER))
            and (BORDER < corners[i, 0, 1] < (h - BORDER))
            for i in range(corners.shape[0])
        ]
    ):
        ok = False

    # Ensure that all corner-arrays are going from top to bottom.
    if board.n_rows != board.n_cols:
        if corners[0, 0, 1] > corners[-1, 0, 1]:
            corners = numpy.copy(numpy.flipud(corners))
    else:
        direction_corners = (corners[-1] - corners[0]) >= numpy.array([[0.0, 0.0]])

        if not numpy.all(direction_corners):
            if not numpy.any(direction_corners):
                corners = numpy.copy(numpy.flipud(corners))
            elif direction_corners[0][0]:
                corners = numpy.rot90(
                    corners.reshape(board.n_rows, board.n_cols, 2)
                ).reshape(board.n_cols * board.n_rows, 1, 2)
            else:
                corners = numpy.rot90(
                    corners.reshape(board.n_rows, board.n_cols, 2), 3
                ).reshape(board.n_cols * board.n_rows, 1, 2)

    if refine and ok:
        # Use a radius of half the minimum distance between corners.
        # This should be large enough to snap to the
        # correct corner, but not so large as to include a wrong corner in the
        # search window.
        min_distance = float("inf")
        for row in range(board.n_rows):
            for col in range(board.n_cols - 1):
                index = row * board.n_rows + col
                min_distance = min(
                    min_distance,
                    _pdist(corners[index, 0], corners[index + 1, 0]),
                )
        for row in range(board.n_rows - 1):
            for col in range(board.n_cols):
                index = row * board.n_rows + col
                min_distance = min(
                    min_distance,
                    _pdist(corners[index, 0], corners[index + board.n_cols, 0]),
                )
        radius = int(math.ceil(min_distance * 0.5))
        cv2.cornerSubPix(
            mono,
            corners,
            (radius, radius),
            (-1, -1),
            (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6),
        )

    return (ok, corners)


def _is_sharp(img, corners, board):
    # Crop image
    up_left, up_right, down_right, down_left = _get_outside_corners(corners, board)
    x1 = up_left[0]
    x2 = up_right[0]
    x3 = down_right[0]
    x4 = down_left[0]
    y1 = up_left[1]
    y2 = up_right[1]
    y3 = down_right[1]
    y4 = down_left[1]

    min_x = int(min(x1, x2, x3, x4))
    max_x = int(max(x1, x2, x3, x4))
    min_y = int(min(y1, y2, y3, y4))
    max_y = int(max(y1, y2, y3, y4))

    img_roi = img[min_y:max_y, min_x:max_x]
    # cv2.imshow("ROI", img_roi)
    lap_var = cv2.Laplacian(img_roi, cv2.CV_64F).var()
    # TODO: what is a good value here?
    # source: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ # noqa
    return lap_var > 0.26


def _get_circles(img, board, pattern, invert=False):
    """
    Get circle centers for a symmetric or asymmetric grid
    """
    if img is None:
        return False, []
    if len(img.shape) == 3 and img.shape[2] == 3:
        mono = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mono = img

    if invert:
        (thresh, mono) = cv2.threshold(
            mono, 140, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        # cv2.imshow('Test', mono)
        # cv2.waitKey(3)

    flag = cv2.CALIB_CB_SYMMETRIC_GRID
    if pattern == Patterns.ACircles:
        flag = cv2.CALIB_CB_ASYMMETRIC_GRID
    mono_arr = numpy.array(mono)

    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 15
    params.minDistBetweenBlobs = 7

    detector = cv2.SimpleBlobDetector_create(params)

    (ok, corners) = cv2.findCirclesGrid(
        mono_arr,
        (board.n_cols, board.n_rows),
        flags=flag,
        blobDetector=detector,
    )

    # In symmetric case, findCirclesGrid does not detect the target if it's
    # turned sideways. So we try again with dimensions swapped - not so
    # efficient.
    # TODO Better to add as second board? Corner ordering will change.
    if not ok and pattern == Patterns.Circles:
        (ok, corners) = cv2.findCirclesGrid(
            mono_arr, (board.n_cols, board.n_rows), flags=flag
        )
    return (ok, corners)


# TODO self.size needs to come from CameraInfo, full resolution
class Calibrator(object):
    """
    Base class for calibration system
    """

    def __init__(
        self,
        boards,
        flags=0,
        pattern=Patterns.Chessboard,
        name="",
        checkerboard_flags=cv2.CALIB_CB_FAST_CHECK,
        max_chessboard_speed=-1.0,
        invert=False,
    ):
        # Ordering the dimensions for the different detectors is actually a
        # minefield...
        if pattern == Patterns.Chessboard:
            # Make sure n_cols > n_rows to agree with OpenCV CB detector output
            self._boards = [
                ChessboardInfo(max(i.n_cols, i.n_rows), min(i.n_cols, i.n_rows), i.dim)
                for i in boards
            ]
        elif pattern == Patterns.ACircles:
            # 7x4 and 4x7 are actually different patterns. Assume square-ish
            # pattern, so n_rows > n_cols.
            self._boards = [
                ChessboardInfo(min(i.n_cols, i.n_rows), max(i.n_cols, i.n_rows), i.dim)
                for i in boards
            ]
        elif pattern == Patterns.Circles:
            # We end up having to check both ways anyway
            self._boards = boards

        self.invert = invert

        # Set to true after we perform calibration
        self.calibrated = False
        self.debug = False
        self.calib_flags = flags
        self.checkerboard_flags = checkerboard_flags
        self.pattern = pattern

        # self.db is list of (parameters, image) samples for use in
        # calibration. parameters has form (X, Y, size, skew) all normalized
        # to [0,1], to keep track of what sort of samples we've taken and
        # ensure enough variety.
        self.db = []
        # For each db sample, we also record the detected corners.
        self.good_corners = []
        # Set to true when we have sufficiently varied samples to calibrate
        self.name = name
        self.last_frame_corners = None
        self.max_chessboard_speed = max_chessboard_speed

    def get_parameters(self, corners, board, size):
        """
        Return list of parameters [X, Y, size, skew] describing the
        checkerboard view.
        """
        (width, height) = size
        Xs = corners[:, :, 0]
        Ys = corners[:, :, 1]
        area = _get_area(corners, board)
        border = math.sqrt(area)
        # For X and Y, we "shrink" the image all around by approx. half the
        # board size. Otherwise large boards are penalized because you can't
        # get much X/Y variation.
        p_x = min(1.0, max(0.0, (numpy.mean(Xs) - border / 2) / (width - border)))
        p_y = min(1.0, max(0.0, (numpy.mean(Ys) - border / 2) / (height - border)))
        p_size = math.sqrt(area / (width * height))
        skew = _get_skew(corners, board)
        params = [p_x, p_y, p_size, skew]
        return params

    def is_slow_moving(self, corners, last_frame_corners):
        """
        Returns true if the motion of the checkerboard is sufficiently low
        between this and the previous frame.
        """
        # If we don't have previous frame corners, we can't accept the sample
        if last_frame_corners is None:
            return False
        num_corners = len(corners)
        corner_deltas = (corners - last_frame_corners).reshape(num_corners, 2)
        # Average distance travelled overall for all corners
        average_motion = numpy.average(numpy.linalg.norm(corner_deltas, axis=1))
        return average_motion <= self.max_chessboard_speed

    def is_good_sample(self, params, corners, last_frame_corners=None, threshold=0.12):
        """
        Returns true if the checkerboard detection described by params should
        be added to the database.
        """
        if not self.db:
            return True

        def param_distance(p1, p2):
            return sum([abs(a - b) for (a, b) in zip(p1, p2)])

        db_params = [sample[0] for sample in self.db]
        d = min([param_distance(params, p) for p in db_params])
        # print("d = %.3f" % d)  # DEBUG
        # TODO What's a good threshold here? Should it be configurable?
        return d > threshold

    def mk_object_points(self, boards):
        opts = []
        for _, b in enumerate(boards):
            num_pts = b.n_cols * b.n_rows
            opts_loc = numpy.zeros((num_pts, 1, 3), numpy.float32)

            for i in range(b.n_rows):
                for j in range(b.n_cols):
                    if self.pattern == Patterns.ACircles:
                        opts_loc[j + i * b.n_cols, 0, 0] = (2 * j + i % 2) * b.dim
                        opts_loc[j + i * b.n_cols, 0, 1] = i * b.dim
                    else:
                        opts_loc[j + i * b.n_cols, 0, 0] = j * b.dim
                        opts_loc[j + i * b.n_cols, 0, 1] = i * b.dim
                opts_loc[j, 0, 2] = 0
            opts.append(opts_loc)
        return opts

    def get_corners(self, img, refine=True):
        """
        Use cvFindChessboardCorners to find corners of chessboard in image.

        Check all boards. Return corners for first chessboard that it detects
        if given multiple size chessboards.

        Returns (ok, corners, board)
        """

        for b in self._boards:
            if self.pattern == Patterns.Chessboard:
                (ok, corners) = _get_corners(img, b, refine, self.checkerboard_flags)
            else:
                (ok, corners) = _get_circles(img, b, self.pattern, self.invert)

            """
            show_img = img.copy()
            cv2.namedWindow('Corners ' + self.name, 0)
            if ok:
                show_img = cv2.drawChessboardCorners(
                    show_img,
                    (b.n_cols, b.n_rows),
                    corners,
                    ok)
                cv2.imshow('Corners ' + self.name, show_img)
                cv2.waitKey(3)
            else:
                cv2.namedWindow('Corners ' + self.name, 0)
                cv2.imshow('Corners ' + self.name, show_img)
                cv2.waitKey(3)
            """

            if ok:
                return (ok, corners, b)
        return (False, None, None)

    def lrreport(self, d, k, r, p):
        print("D = ", numpy.ravel(d).tolist())
        print("K = ", numpy.ravel(k).tolist())
        print("R = ", numpy.ravel(r).tolist())
        print("P = ", numpy.ravel(p).tolist())


def resize_with_padding(img, size):
    width, height = size
    o_height, o_width = img.shape[0], img.shape[1]

    ar = float(width) / float(height)
    o_ar = float(o_width) / float(o_height)

    left, right, top, bottom = 0, 0, 0, 0

    if ar < o_ar:
        # Image needs padding on top/bottom
        top = int((float(o_width) / ar - o_height) / 2.0)
        bottom = top
    elif ar > o_ar:
        # Image needs padding on left/right
        left = int((ar * float(o_height) - o_width) / 2.0)
        right = left

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    if img.shape[0] != height:
        # needs scaling
        img = cv2.resize(img, size)

    return img


class MonoCalibrator(Calibrator):
    """
    Calibration class for monocular cameras::

        images = [cv2.imread("mono%d.png") for i in range(8)]
        mc = MonoCalibrator()
        mc.cal(images)
        print mc.as_message()
    """

    is_mono = True  # TODO Could get rid of is_mono

    def __init__(self, target_width=None, target_height=None, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "narrow_stereo/left"
        self.resize = False
        if target_width is not None and target_height is not None:
            self.resize = True
            self.target_height = target_height
            self.target_width = target_width
        super(MonoCalibrator, self).__init__(*args, **kwargs)
        self.R = numpy.eye(3, dtype=numpy.float64)
        self.P = numpy.zeros((3, 4), dtype=numpy.float64)
        self.max_iterations = 5
        self.calibrations = [None] * self.max_iterations

    def cal(self, images_list):
        """
        Calibrate camera from given images
        """
        self.collect_corners(images_list)
        self.cal_fromcorners()
        self.calibrated = True

    def cal_from_json(self, json_file, images_list):
        self.good_corners = []
        test_image = cv2.imread(str(images_list[0]))
        self.size = (test_image.shape[1], test_image.shape[0])
        if self.resize:
            self.size = self.target_width, self.target_height
        with json_file.open("r") as f:
            data = json.load(f)
            for lf in data:
                lboard = ChessboardInfo()
                lboard.fromlist(lf["board"])
                lcorners = np.array(lf["corners"], dtype=np.float32)
                params = self.get_parameters(lcorners, lboard, lf["size"])
                if self.is_good_sample(params, lcorners, self.last_frame_corners):
                    lgray = cv2.imread(lf["file"])
                    print(
                        "*** Added sample p_x = %.3f, p_y = %.3f, p_size = %.3f, skew = %.3f"  # noqa
                        % tuple(params)
                    )
                    # lgray2 = lgray.copy()
                    # limg = cv2.drawChessboardCorners(
                    #    lgray2, (lboard.n_cols, lboard.n_rows), lcorners, True
                    # )
                    # cv2.namedWindow("Mono calibration1", 0)
                    # cv2.imshow("Mono calibration1", limg)
                    # cv2.waitKey(3)
                    self.last_frame_corners = lcorners
                    self.db.append((params, lgray))
                    self.good_corners.append((lcorners, lboard))
            print("Using " + str(len(self.good_corners)) + " inlier files!")
            self.cal_fromcorners()

    def collect_corners(self, images_list):
        """
        :param images: source images containing chessboards
        :type images: list of :class:`cvMat`

        Find chessboards in all images.

        Return [ (corners, ChessboardInfo) ]
        """
        test_image = cv2.imread(str(images_list[0]))
        self.size = (test_image.shape[1], test_image.shape[0])

        if self.resize:
            self.size = self.target_width, self.target_height

        self.good_corners = []
        self.db = []
        self.json = []
        good_corners_names = []

        def get_image_corners(i):
            gray = cv2.imread(str(i))

            if gray is None:
                print(
                    str(i.name)
                    + " could not be loaded. File might not be an image or "
                    + "file might be corrupted. Skipping file."
                )
                return

            if self.resize:
                gray = resize_with_padding(gray, self.size)

            ok, corners, board = self.get_corners(gray)
            if not ok:
                print("Chessboard NOT detected in " + str(i.name))
                return
            # Add sample to database only if it's sufficiently different from
            # any previous sample.
            params = self.get_parameters(corners, board, self.size)
            if _is_sharp(gray, corners, board):
                print("Chessboard detected! in " + str(i.name))
                if self.debug:
                    gray_copy = gray.copy()
                    img = cv2.drawChessboardCorners(
                        gray_copy, (board.n_cols, board.n_rows), corners, ok
                    )
                    # cv2.imshow("Monocular calibration", img)
                    # cv2.waitKey(3)
                    name = Path(i).stem
                    filename = Path("/tmp/" + self.name + "/" + name + "_corners.png")
                    if self.resize:
                        filename = Path(
                            "/tmp/" + self.name + "/resized_" + name + "_corners.png"
                        )
                    if not filename.parents[0].exists():
                        filename.parents[0].mkdir(parents=True)
                    print("Writing debug image to " + str(filename))
                    cv2.imwrite(str(filename), img)

                return [(params, gray), (corners, board), i]

            else:
                print("Image " + str(i.name) + " is blurry, discarded.")

        result = joblib.Parallel(n_jobs=-2)(
            [joblib.delayed(get_image_corners)(i) for i in images_list]
        )

        name = ""
        good_corners_names = []
        all_detections = None
        for i in result:
            if i is not None:
                if len(i) > 2:
                    (params, gray) = i[0]
                    (corners, board) = i[1]
                    if all_detections is None:
                        all_detections = gray.copy()
                    self.json.append(
                        {
                            "file": str(i[2]),
                            "corners": corners.tolist(),
                            "board": board.tolist(),
                            "size": self.size,
                        }
                    )
                    if self.is_good_sample(params, corners, self.last_frame_corners):
                        print(
                            "*** Added sample p_x = %.3f, p_y = %.3f, p_size = %.3f, skew = %.3f"  # noqa
                            % tuple(params)
                        )
                        self.last_frame_corners = corners
                        self.db.append(i[0])
                        self.good_corners.append(i[1])
                        good_corners_names.append(str(i[2]))
                        all_detections = cv2.drawChessboardCorners(
                            all_detections,
                            (board.n_cols, board.n_rows),
                            corners,
                            True,
                        )

                        if name == "":
                            name = i[2].stem
        filename = Path("/tmp/" + self.name + "_corners.png")
        if self.resize:
            filename = Path("/tmp/resized_" + self.name + "_corners.png")
        if not filename.parents[0].exists():
            filename.parents[0].mkdir(parents=True)
        print("Writing corners image to " + str(filename))
        cv2.imwrite(str(filename), all_detections)

        if not self.good_corners:
            raise CalibrationException("No corners found in images!")
        print("Using " + str(len(self.good_corners)) + " inlier files!")
        filename = Path("/tmp/" + self.name + "/" + name + "_inliers.txt")
        if self.resize:
            filename = Path("/tmp/" + self.name + "/resized_" + name + "_inliers.txt")
        if not filename.parents[0].exists():
            filename.parents[0].mkdir(parents=True)
        with filename.open("w") as f:
            for n in good_corners_names:
                f.write("%s\n" % n)

    def cal_fromcorners(self, iteration=0):
        """
        Compute intrinsic camera calibrtion parameters

        Given a set of chessboard corners / calibration dot locations (in
        self.good_corners), this function computes the intrisic camera
        calibrtion parameters. It detects calibration patterns with high
        reprojection errors, eliminates these and calls itself again (i.e.
        recursively) with the reduced set of calibration patterns until all
        reprojection errors are within given bounds or the maximum number of
        iterations has been reached.

        :param iteration: Indicating number of (recursive) function call
        :type iteration: int
        """
        if iteration == 0:
            Console.info("Calibrating monocular...")

        boards = [b for (_, b) in self.good_corners]
        Console.info(
            "Iteration",
            iteration,
            "using",
            len(boards),
            "views of the calibration board",
        )

        ipts = [points for (points, _) in self.good_corners]
        opts = self.mk_object_points(boards)

        self.intrinsics = numpy.zeros((3, 3), numpy.float64)
        if self.calib_flags & cv2.CALIB_RATIONAL_MODEL:
            self.distortion = numpy.zeros((8, 1), numpy.float64)  # rational polynomial
        else:
            self.distortion = numpy.zeros((5, 1), numpy.float64)  # plumb bob

        self.num_images = len(self.good_corners)

        [
            self.avg_reprojection_error,
            _,
            _,
            _,
            _,
            self.stdev_intrinsics,
            _,
            per_view_errors,
        ] = cv2.calibrateCameraExtended(
            opts, ipts, self.size, self.intrinsics, self.distortion
        )

        Console.info("Per view reprojection error:\n" + str(per_view_errors))
        Console.info("Average reprojection error: " + str(self.avg_reprojection_error))
        Console.info("Stdev intrinsics: " + str(self.stdev_intrinsics[:, 0].T))

        # R is identity matrix for monocular calibration
        self.R = numpy.eye(3, dtype=numpy.float64)
        self.P = numpy.zeros((3, 4), dtype=numpy.float64)
        self.set_alpha(1.0)

        calibration = {
            "name": self.name,
            "distortion": self.distortion,
            "intrinsics": self.intrinsics,
            "stdev_intrinsics": self.stdev_intrinsics,
            "R": self.R,
            "P": self.P,
            "num_images": self.num_images,
            "avg_reprojection_error": self.avg_reprojection_error,
        }

        # Save all calibrations and pick the best
        self.calibrations[iteration] = calibration

        reproj_err_std = np.std(per_view_errors)
        acceptable_reproj_err = max(
            self.avg_reprojection_error + 2 * reproj_err_std, 0.25
        )
        bad_corners = per_view_errors > acceptable_reproj_err
        if sum(bad_corners) > 0:
            Console.info(
                "Number of views exceeding reprojection error threshold ("
                + str(acceptable_reproj_err)
                + "): "
                + str(sum(bad_corners)[0])
                + " out of "
                + str(len(per_view_errors))
                + " calibration board views"
            )
            if iteration == self.max_iterations - 1:
                Console.info("but maximum number of iterations has been reached")
            else:
                self.good_corners = [
                    c
                    for c, e in zip(self.good_corners, per_view_errors)
                    if e <= acceptable_reproj_err
                ]
                self.cal_fromcorners(iteration=iteration + 1)
        else:
            Console.info("All reprojection errors below throshold.")

        # Find best calibration
        min_reprojection_error = np.inf
        best_idx = None
        for i, cal in enumerate(self.calibrations):
            if cal is None:
                continue
            if cal["avg_reprojection_error"] < min_reprojection_error:
                best_idx = i
                min_reprojection_error = cal["avg_reprojection_error"]

        if best_idx is not None:
            # print(
            #     "Calibration with 0-based index", best_idx, "out of",
            #     len(self.calibrations),
            #     "calibrations was chosen as best calibration"
            # )  # Usually last one
            self.distortion = self.calibrations[best_idx]["distortion"]
            self.intrinsics = self.calibrations[best_idx]["intrinsics"]
            self.stdev_intrinsics = self.calibrations[best_idx]["stdev_intrinsics"]
            self.R = self.calibrations[best_idx]["R"]
            self.P = self.calibrations[best_idx]["P"]
            self.num_images = self.calibrations[best_idx]["num_images"]
            self.avg_reprojection_error = self.calibrations[best_idx][
                "avg_reprojection_error"
            ]

        if self.debug:
            for i, (params, img) in enumerate(self.db):
                # img = self.db[0][1]
                img = self.remap(img)

                filename = Path(
                    "/tmp/"
                    + self.name
                    + "_remap/"
                    + str(i)
                    + "_"
                    + self.name
                    + "_remap.png"
                )
                if not filename.parents[0].exists():
                    filename.parents[0].mkdir(parents=True)
                Console.info("Writing debug remap image to " + str(filename))
                cv2.imwrite(str(filename), img)

    def set_alpha(self, a):
        """
        Set the alpha value for the calibrated camera solution.  The alpha
        value is a zoom, and ranges from 0 (zoomed in, all pixels in
        calibrated image are valid) to 1 (zoomed out, all pixels in
        original image are in calibrated image).
        """

        ncm, _ = cv2.getOptimalNewCameraMatrix(
            self.intrinsics, self.distortion, self.size, a
        )
        for j in range(3):
            for i in range(3):
                self.P[j, i] = ncm[j, i]
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.intrinsics,
            self.distortion,
            self.R,
            ncm,
            self.size,
            cv2.CV_32FC1,
        )

    def remap(self, src):
        """
        :param src: source image
        :type src: :class:`cvMat`

        Apply the post-calibration undistortion to the source image
        """
        return cv2.remap(src, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_points(self, src):
        """
        :param src: N source pixel points (u,v) as an Nx2 matrix
        :type src: :class:`cvMat`

        Apply the post-calibration undistortion to the source points
        """

        return cv2.undistortPoints(
            src, self.intrinsics, self.distortion, R=self.R, P=self.P
        )

    def report(self):
        self.lrreport(self.distortion, self.intrinsics, self.R, self.P)

    def yaml(self):
        name = self.name
        d = self.distortion
        k = self.intrinsics
        stdev_intrisics = self.stdev_intrinsics
        r = self.R
        p = self.P
        num_images = self.num_images
        error = self.avg_reprojection_error

        calmessage = (
            "%YAML:1.0\n"
            + "image_width: "
            + str(self.size[0])
            + "\n"
            + "image_height: "
            + str(self.size[1])
            + "\n"
            + "camera_name: "
            + name
            + "\n"
            + "camera_matrix: !!opencv-matrix\n"
            + "  rows: 3\n"
            + "  cols: 3\n"
            + "  dt: d\n"
            + "  data: ["
            + ", ".join(["%8f" % i for i in k.reshape(1, 9)[0]])
            + "]\n"
            + "distortion_model: "
            + ("rational_polynomial" if d.size > 5 else "plumb_bob")
            + "\n"
            + "distortion_coefficients: !!opencv-matrix\n"
            + "  rows: 1\n"
            + "  cols: 5\n"
            + "  dt: d\n"
            + "  data: ["
            + ", ".join(["%8f" % d[i, 0] for i in range(d.shape[0])])
            + "]\n"
            + "rectification_matrix: !!opencv-matrix\n"
            + "  rows: 3\n"
            + "  cols: 3\n"
            + "  dt: d\n"
            + "  data: ["
            + ", ".join(["%8f" % i for i in r.reshape(1, 9)[0]])
            + "]\n"
            + "projection_matrix: !!opencv-matrix\n"
            + "  rows: 3\n"
            + "  cols: 4\n"
            + "  dt: d\n"
            + "  data: ["
            + ", ".join(["%8f" % i for i in p.reshape(1, 12)[0]])
            + "]\n"
            + "stdev_intrinsics: !!opencv-matrix\n"
            + "  rows: 1\n"
            + "  cols: 9\n"
            + "  dt: d\n"
            + "  data: ["  # Output elements 0 to 8 as 9 to 17 are 0
            + ", ".join(["%8f" % stdev_intrisics[i, 0] for i in range(9)])
            + "]\n"
            + 'date: "'
            + Console.get_date()
            + '" \n'
            + 'user: "'
            + Console.get_username()
            + '" \n'
            + 'host: "'
            + Console.get_hostname()
            + '" \n'
            + 'version: "'
            + Console.get_version()
            + '" \n'
            + "number_of_images: "
            + str(num_images)
            + "\n"
            + "avg_reprojection_error: "
            + str(error)
            + "\n"
            + ""
        )
        return calmessage


def find_common_filenames(list1, list2):
    result = []

    lf = [Path(d["file"]).stem for d in list1]
    rf = [Path(d["file"]).stem for d in list2]

    if (
        lf[0][23:25] == "LC"
        or lf[0][23:25] == "RC"
        or lf[0][23:25] == "FC"
        or lf[0][23:25] == "AC"
    ):
        camera_format = "acfr"
    else:
        if lf[0][0:5] == "image":
            lf = [Path(d["file"]).stem[5:] for d in list1]
        if rf[0][0:5] == "image":
            rf = [Path(d["file"]).stem[5:] for d in list2]
        camera_format = "seaxerocks3"
        if len(lf[0]) > 26:
            camera_format = "biocam"

    if camera_format == "seaxerocks3":
        for i, lname in enumerate(lf):
            for j, rname in enumerate(rf):
                if lname == rname:
                    # Console.info(lname, rname)
                    # Console.info(list1[i]['file'], list2[j]['file'])
                    result.append((list1[i], list2[j]))
    elif camera_format == "biocam":
        stamp_pc1 = []
        stamp_cam1 = []
        stamp_pc2 = []
        stamp_cam2 = []
        for i in range(len(lf)):
            t1, tc1 = biocam_timestamp_from_filename(lf[i], 0, 0)
            stamp_pc1.append(float(t1))
            stamp_cam1.append(float(tc1))
        for i in range(len(rf)):
            t1, tc1 = biocam_timestamp_from_filename(rf[i], 0, 0)
            stamp_pc2.append(float(t1))
            stamp_cam2.append(float(tc1))

        tolerance = 0.05  # stereo pair must be within 50ms of each other

        for i in range(len(lf)):
            values = []
            for j in range(len(rf)):
                values.append(abs(stamp_pc1[i] - stamp_pc2[j]))

            (sync_difference, sync_pair) = min((v, k) for k, v in enumerate(values))
            if sync_difference < tolerance:
                # Console.info(lf[i] + ' syncs with ' + rf[sync_pair] + ' with dif ' + str(sync_difference)) # noqa
                result.append((list1[i], list2[sync_pair]))
    elif camera_format == "acfr":
        stamp_cam1 = []
        stamp_cam2 = []
        for i in range(len(lf)):
            tc1 = acfr_timestamp_from_filename(lf[i], 0, 0)
            stamp_cam1.append(float(tc1))
        for i in range(len(rf)):
            tc1 = acfr_timestamp_from_filename(rf[i], 0, 0)
            stamp_cam2.append(float(tc1))

        tolerance = 0.05  # stereo pair must be within 50ms of each other

        for i in range(len(lf)):
            values = []
            for j in range(len(rf)):
                values.append(abs(stamp_cam1[i] - stamp_cam2[j]))

            (sync_difference, sync_pair) = min((v, k) for k, v in enumerate(values))
            if sync_difference < tolerance:
                # print(lf[i] + ' syncs with ' + rf[sync_pair] + ' with dif ' + str(sync_difference)) # noqa
                result.append((list1[i], list2[sync_pair]))
    else:
        Console.error("Stereo format not known, calibration will not work")
    return result


# TODO Replicate MonoCalibrator improvements in stereo
class StereoCalibrator(Calibrator):
    """
    Calibration class for stereo cameras::

        limages = [cv2.imread("left%d.png") for i in range(8)]
        rimages = [cv2.imread("right%d.png") for i in range(8)]
        sc = StereoCalibrator()
        sc.cal(limages, rimages)
        Console.info sc.as_message()
    """

    is_mono = False

    def __init__(self, stereo_camera_model, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "camera1"
        super(StereoCalibrator, self).__init__(*args, **kwargs)
        self.l = MonoCalibrator(*args, **kwargs)  # noqa E741
        self.r = MonoCalibrator(*args, **kwargs)
        # Collecting from two cameras in a horizontal stereo rig, can't get
        # full X range in the left camera.
        self.inliers = []

        self.l.name = stereo_camera_model.left.name
        self.l.size = (
            stereo_camera_model.left.image_width,
            stereo_camera_model.left.image_height,
        )
        self.l.intrinsics = stereo_camera_model.left.K
        self.l.distortion = stereo_camera_model.left.d
        self.r.name = stereo_camera_model.right.name
        self.r.size = (
            stereo_camera_model.right.image_width,
            stereo_camera_model.right.image_height,
        )
        self.r.intrinsics = stereo_camera_model.right.K
        self.r.distortion = stereo_camera_model.right.d

        self.resize_right = False

        self.max_iterations = 5
        self.calibrations = [None] * self.max_iterations

    def test_same_image_size(self, test_left, test_right):
        Console.info(test_left, test_right)
        test_image_left = cv2.imread(str(test_left))
        test_image_right = cv2.imread(str(test_right))
        size_left = (test_image_left.shape[1], test_image_left.shape[0])
        size_right = (test_image_right.shape[1], test_image_right.shape[0])
        Console.info("Images sizes are", size_left, size_right)
        if size_left != size_right:
            Console.warn("Calibrating stereo pair with different image sizes")
            return True
        return False

    def cal_from_json(self, left_json, right_json):
        common = find_common_filenames(left_json, right_json)

        test_left, test_right = common[0]
        self.resize_right = self.test_same_image_size(
            test_left["file"], test_right["file"]
        )

        Console.info("Found " + str(len(common)) + " common files!")
        self.good_corners = []
        for lf, rf in common:
            lboard = ChessboardInfo()
            lboard.fromlist(lf["board"])
            lcorners = np.array(lf["corners"], dtype=np.float32)
            rcorners = np.array(rf["corners"], dtype=np.float32)
            lparams = self.get_parameters(lcorners, lboard, self.l.size)
            # rparams = self.get_parameters(rcorners, lboard, self.r.size)
            if self.is_good_sample(lparams, lcorners):
                lgray = cv2.imread(lf["file"])
                rgray = cv2.imread(rf["file"])
                if self.resize_right:
                    rgray = resize_with_padding(rgray, self.l.size)
                Console.info(
                    (
                        "*** Added sample %d, p_x = %.3f, p_y = %.3f, p_size = %.3f, skew = %.3f"  # noqa
                        % tuple([len(self.db)] + lparams)
                    )
                )
                # lgray_copy = lgray.copy()
                # rgray_copy = rgray.copy()
                # limg = cv2.drawChessboardCorners(
                #     lgray_copy, (lboard.n_cols, lboard.n_rows),
                #     lcorners, True
                # )
                # rimg = cv2.drawChessboardCorners(
                #     rgray_copy, (lboard.n_cols, lboard.n_rows),
                #     rcorners, True
                # )
                # vis = np.concatenate((limg, rimg), axis=1)
                # cv2.namedWindow("Stereo calibration", 0)
                # cv2.imshow("Stereo calibration", vis)
                # cv2.waitKey(3)
                # Console.info(lf['file'], rf['file'])
                self.db.append((lparams, lgray, rgray))
                self.good_corners.append((lcorners, rcorners, lboard))
        Console.info("Using " + str(len(self.good_corners)) + " inlier files!")
        self.cal_fromcorners()

    def cal(self, limages_list, rimages_list):
        """
        :param limages: source left images containing chessboards
        :type limages: list of :class:`cvMat`
        :param rimages: source right images containing chessboards
        :type rimages: list of :class:`cvMat`

        Find chessboards in images, and runs the OpenCV calibration solver.
        """

        self.resize_right = self.test_same_image_size(limages_list[0], rimages_list[0])
        self.collect_corners(limages_list, rimages_list)
        self.cal_fromcorners()
        self.calibrated = True

    def collect_corners(self, limages_list, rimages_list):
        """
        For a sequence of left and right images, find pairs of images where
        both left and right have a chessboard, and return  their corners as a
        list of pairs.
        """
        self.good_corners = []
        for (i, j) in zip(limages_list, rimages_list):
            lgray = cv2.imread(str(i))
            rgray = cv2.imread(str(j))
            if self.resize_right:
                rgray = resize_with_padding(rgray, self.l.size)
            lok, lcorners, lboard = self.get_corners(lgray)
            rok, rcorners, rboard = self.get_corners(rgray)
            if (not lok) or (not rok):
                continue
            # Add sample to database only if it's sufficiently different from
            # any previous sample.
            params = self.get_parameters(lcorners, lboard, self.l.size)
            if self.is_good_sample(params, lcorners, self.last_frame_corners):
                if _is_sharp(lgray, lcorners, lboard) and _is_sharp(
                    rgray, rcorners, rboard
                ):
                    self.db.append((params, lgray, rgray))
                    self.good_corners.append((lcorners, rcorners, lboard))
                    # Console.info(("*** Added sample %d, p_x = %.3f, p_y = %.3f, p_size = %.3f, skew = %.3f" % # noqa
                    #        tuple([len(self.db)] + params)))

                    self.inliers.append(
                        (lgray, rgray, lcorners, lboard, rcorners, rboard)
                    )

                    Console.info("Writing debug images to /tmp/")
                    if self.debug:
                        limg = cv2.drawChessboardCorners(
                            lgray,
                            (lboard.n_cols, lboard.n_rows),
                            lcorners,
                            lok,
                        )
                        rimg = cv2.drawChessboardCorners(
                            rgray,
                            (rboard.n_cols, rboard.n_rows),
                            rcorners,
                            rok,
                        )
                        lname = Path(i).stem
                        rname = Path(j).stem

                        lfilename = Path(
                            "/tmp/stereo_"
                            + self.name
                            + "/"
                            + lname
                            + "_left_corners.png"
                        )
                        rfilename = Path(
                            "/tmp/stereo_"
                            + self.name
                            + "/"
                            + rname
                            + "_right_corners.png"
                        )
                        if not lfilename.parents[0].exists():
                            lfilename.parents[0].mkdir(parents=True)
                        if not rfilename.parents[0].exists():
                            rfilename.parents[0].mkdir(parents=True)
                        cv2.imwrite(str(lfilename), limg)
                        cv2.imwrite(str(rfilename), rimg)
                elif self.debug:
                    Console.info(
                        "Image "
                        + str(i)
                        + " or "
                        + str(j)
                        + " are blurry, pair discarded."
                    )
            self.last_frame_corners = lcorners
        if len(self.good_corners) == 0:
            raise CalibrationException("No corners found in images!")

    def cal_fromcorners(self, iteration=0):
        lipts = [l for (l, _, _) in self.good_corners]
        ripts = [r for (_, r, _) in self.good_corners]
        boards = [b for (_, _, b) in self.good_corners]

        opts = self.mk_object_points(boards)

        flags = cv2.CALIB_FIX_INTRINSIC
        # flags = cv2.CALIB_USE_INTRINSIC_GUESS
        self.T = numpy.zeros((3, 1), dtype=numpy.float64)
        self.R = numpy.eye(3, dtype=numpy.float64)

        Console.info(
            "Calibrating stereo with " + str(len(self.good_corners)) + " image pairs..."
        )
        if len(self.good_corners) < 8:
            Console.quit("Too few stereo calibration images!")

        self.num_images = len(self.good_corners)
        (
            self.avg_reprojection_error,
            self.l.intrinsics,
            self.l.distortion,
            self.r.intrinsics,
            self.r.distortion,
            self.R,
            self.T,
            self.E,
            self.F,
        ) = cv2.stereoCalibrate(
            opts,
            lipts,
            ripts,
            self.l.intrinsics,
            self.l.distortion,
            self.r.intrinsics,
            self.r.distortion,
            self.l.size,
            flags=flags,
        )
        Console.info(
            "Calibrated with RMS = {}".format(str(self.avg_reprojection_error))
        )

        # print(self.l.intrinsics)
        # print(self.l.distortion)
        # print(self.R)
        # print(self.T)

        self.set_alpha(1.0)

        calibration = {
            "name": self.name,
            "R": self.R,
            "T": self.T,
            "E": self.E,
            "F": self.F,
            "num_images": self.num_images,
            "avg_reprojection_error": self.avg_reprojection_error,
        }

        # Save all calibrations and pick the best
        print(iteration, "iteration")
        self.calibrations[iteration] = calibration

        epipolar_error = []
        for (params, lgray, rgray) in self.db:
            error = self.epipolar_error_from_images(lgray, rgray)
            print("stereo error:", error)
            if error is not None:
                epipolar_error.append(error)
            else:
                epipolar_error.append(1e3)

        if iteration < self.max_iterations - 1 and self.avg_reprojection_error > 0.6:
            epi_error_mean = np.mean(np.array(epipolar_error))
            epi_error_std = np.std(np.array(epipolar_error))
            self.good_corners = [
                c
                for c, e in zip(self.good_corners, epipolar_error)
                if e < epi_error_mean + 2 * epi_error_std
            ]
            Console.info(
                "Using "
                + str(len(self.good_corners))
                + " inlier files after filtering!"
            )
            self.cal_fromcorners(iteration=iteration + 1)

        # Use best calibration
        min_reprojection_error = np.inf
        best_idx = None
        for i, cal in enumerate(self.calibrations):
            if cal is None:
                continue
            if cal["avg_reprojection_error"] < min_reprojection_error:
                best_idx = i
                min_reprojection_error = cal["avg_reprojection_error"]

        if best_idx is not None:
            self.R = self.calibrations[best_idx]["R"]
            self.T = self.calibrations[best_idx]["T"]
            self.E = self.calibrations[best_idx]["E"]
            self.F = self.calibrations[best_idx]["F"]
            self.num_images = self.calibrations[best_idx]["num_images"]
            self.avg_reprojection_error = self.calibrations[best_idx][
                "avg_reprojection_error"
            ]

        if self.debug:
            errors = []
            Console.info("Writing debug images to /tmp/")
            for i, (params, limg, rimg) in enumerate(self.db):
                error = self.epipolar_error_from_images(limg, rimg)

                limg = self.l.remap(limg)
                rimg = self.r.remap(rimg)

                lh, lw, lc = limg.shape
                rh, rw, rc = rimg.shape

                # find the max width of all the images
                max_width = lw
                if rw > max_width:
                    max_width = rw
                # the total height of the images (vertical stacking)
                total_height = lh + rh
                # create a new array with a size large enough to contain all
                # the images
                final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)
                final_image[:lh, :lw, :] = limg
                final_image[lh : lh + rh, :rw, :] = rimg  # noqa

                # cv2.imshow('Final images', final_image)
                # cv2.waitKey(3)

                filename = Path(
                    "/tmp/stereo_"
                    + self.name
                    + "_remap/stereo_"
                    + str(i)
                    + "_"
                    + self.l.name
                    + "_"
                    + self.r.name
                    + "_remap.png"
                )
                if not filename.parents[0].exists():
                    filename.parents[0].mkdir(parents=True)

                cv2.imwrite(str(filename), final_image)
                if error is not None:
                    errors.append(error)
            # Console.info(errors)

    def set_alpha(self, a):
        """
        Set the alpha value for the calibrated camera solution. The
        alpha value is a zoom, and ranges from 0 (zoomed in, all pixels
        in calibrated image are valid) to 1 (zoomed out, all pixels in
        original image are in calibrated image).
        """
        Console.info("Rectifying stereo...")
        self.l.R, self.r.R, self.l.P, self.r.P = cv2.stereoRectify(
            self.l.intrinsics,
            self.l.distortion,
            self.r.intrinsics,
            self.r.distortion,
            self.l.size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=a,
        )[0:4]

        print(self.l.P, "\n", self.r.P)

        self.l.mapx, self.l.mapy = cv2.initUndistortRectifyMap(
            self.l.intrinsics,
            self.l.distortion,
            self.l.R,
            self.l.P,
            self.l.size,
            cv2.CV_32FC1,
        )
        self.r.mapx, self.r.mapy = cv2.initUndistortRectifyMap(
            self.r.intrinsics,
            self.r.distortion,
            self.r.R,
            self.r.P,
            self.r.size,
            cv2.CV_32FC1,
        )

    def report(self):
        Console.info("\nLeft:")
        self.lrreport(self.l.distortion, self.l.intrinsics, self.l.R, self.l.P)
        Console.info("\nRight:")
        self.lrreport(self.r.distortion, self.r.intrinsics, self.r.R, self.r.P)
        Console.info("self.T ", numpy.ravel(self.T).tolist())
        Console.info("self.R ", numpy.ravel(self.R).tolist())

    def yaml(self):
        d1 = self.l.distortion
        k1 = self.l.intrinsics
        r1 = self.l.R
        p1 = self.l.P

        d2 = self.r.distortion
        k2 = self.r.intrinsics
        r2 = self.r.R
        p2 = self.r.P
        Console.info(d1.shape)
        num_images = self.num_images
        error = self.avg_reprojection_error

        calmessage = (
            "%YAML:1.0\n"
            "left:\n"
            "  image_width: " + str(self.l.size[0]) + "\n"
            "  image_height: " + str(self.l.size[1]) + "\n"
            "  camera_name: " + self.l.name + "\n"
            "  camera_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 3\n"
            "    dt: d\n"
            "    data: [" + ", ".join(["%8f" % i for i in k1.reshape(1, 9)[0]]) + "]\n"
            "  distortion_model: "
            + ("rational_polynomial" if d1.size > 5 else "plumb_bob")
            + "\n"
            "  distortion_coefficients: !!opencv-matrix\n"
            "    rows: 1\n"
            "    cols: 5\n"
            "    dt: d\n"
            "    data: ["
            + ", ".join(["%8f" % d1[0, i] for i in range(d1.shape[1])])
            + "]\n"
            "  rectification_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 3\n"
            "    dt: d\n"
            "    data: [" + ", ".join(["%8f" % i for i in r1.reshape(1, 9)[0]]) + "]\n"
            "  projection_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 4\n"
            "    dt: d\n"
            "    data: [" + ", ".join(["%8f" % i for i in p1.reshape(1, 12)[0]]) + "]\n"
            "right:\n"
            "  image_width: " + str(self.r.size[0]) + "\n"
            "  image_height: " + str(self.r.size[1]) + "\n"
            "  camera_name: " + self.r.name + "\n"
            "  camera_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 3\n"
            "    dt: d\n"
            "    data: [" + ", ".join(["%8f" % i for i in k2.reshape(1, 9)[0]]) + "]\n"
            "  distortion_model: "
            + ("rational_polynomial" if d2.size > 5 else "plumb_bob")
            + "\n"
            "  distortion_coefficients: !!opencv-matrix\n"
            "    rows: 1\n"
            "    cols: 5\n"
            "    dt: d\n"
            "    data: ["
            + ", ".join(["%8f" % d2[0, i] for i in range(d2.shape[1])])
            + "]\n"
            "  rectification_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 3\n"
            "    dt: d\n"
            "    data: [" + ", ".join(["%8f" % i for i in r2.reshape(1, 9)[0]]) + "]\n"
            "  projection_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 4\n"
            "    dt: d\n"
            "    data: [" + ", ".join(["%8f" % i for i in p2.reshape(1, 12)[0]]) + "]\n"
            "extrinsics:\n"
            "  rotation_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 3\n"
            "    dt: d\n"
            "    data: ["
            + ", ".join(["%8f" % i for i in self.R.reshape(1, 9)[0]])
            + "]\n"
            "  translation_vector: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 1\n"
            "    dt: d\n"
            "    data: ["
            + ", ".join(["%8f" % i for i in self.T.reshape(1, 3)[0]])
            + "]\n"
            "  fundamental_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 3\n"
            "    dt: d\n"
            "    data: ["
            + ", ".join(["%8f" % i for i in self.F.reshape(1, 9)[0]])
            + "]\n"  # troublemaker
            "  essential_matrix: !!opencv-matrix\n"
            "    rows: 3\n"
            "    cols: 3\n"
            "    dt: d\n"
            "    data: ["
            + ", ".join(["%8f" % i for i in self.E.reshape(1, 9)[0]])
            + "]\n"
            'date: "' + Console.get_date() + '" \n'
            'user: "' + Console.get_username() + '" \n'
            'host: "' + Console.get_hostname() + '" \n'
            'version: "' + Console.get_version() + '" \n'
            "number_of_images: " + str(num_images) + "\n"
            "avg_reprojection_error: " + str(error) + "\n"
            ""
        )
        return calmessage

    # TODO Get rid of "from_images" versions of these, instead have function
    # to get undistorted corners
    def epipolar_error_from_images(self, limage, rimage):
        """
        Detect the checkerboard in both images and compute the epipolar error.
        Mainly for use in tests.
        """
        lok, lcorners, lboard = self.get_corners(limage)
        rok, rcorners, rboard = self.get_corners(rimage)

        # vis = np.concatenate((limage, rimage), axis=1)
        # cv2.namedWindow("Stereo error", 0)
        # cv2.imshow("Stereo error", vis)
        # cv2.waitKey(3)

        if lcorners is None or rcorners is None:
            # Console.warn("Cannot find the calibration pattern again when
            # computing the epipolar error.")
            return None

        lundistorted = self.l.undistort_points(lcorners)
        rundistorted = self.r.undistort_points(rcorners)

        # print('L', lcorners[0, 0, 0], lcorners[0, 0, 1], 'to',
        #       lundistorted[0, 0, 0], lundistorted[0, 0, 1])
        # print('R', rcorners[0, 0, 0], rcorners[0, 0, 1], 'to',
        #       rundistorted[0, 0, 0], rundistorted[0, 0, 1])

        return self.epipolar_error(lundistorted, rundistorted)

    def epipolar_error(self, lcorners, rcorners):
        """
        Compute the epipolar error from two sets of matching undistorted points
        """
        d2 = numpy.abs(lcorners[:, :, 1] - rcorners[:, :, 1])
        d1 = numpy.abs(lcorners[:, :, 0] - rcorners[:, :, 0])
        d = numpy.minimum(d1, d2)
        error = numpy.sqrt(numpy.square(d).sum() / d.size)
        return error
