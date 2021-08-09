# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
import yaml

# RANSAC
from ransac import plane_fitting_ransac

from oplab import StereoCamera


def projectPoint(sc, point3d):
    p = np.array([point3d]).reshape(-1, 1, 3)
    z = np.zeros((1, 3), dtype=float)

    left_image_points = cv2.projectPoints(
        p, z, z, sc.left.P[0:3, 0:3], sc.left.d
    )[0]

    rvec, _ = cv2.Rodrigues(sc.R.T)
    tvec = sc.t
    right_image_points = cv2.projectPoints(
        p, rvec, tvec, sc.right.P[0:3, 0:3], sc.right.d
    )[0]

    w = float(sc.left.image_width)
    h = float(sc.left.image_height)

    lp = left_image_points[0]
    rp = right_image_points[0]

    lx = int(round(lp[0, 0]))
    ly = int(round(lp[0, 1]))
    rx = int(round(rp[0, 0]))
    ry = int(round(rp[0, 1]))

    if (
        lx < w
        and rx < w
        and ly < h
        and ry < h
        and lx > 0
        and rx > 0
        and ly > 0
        and ry > 0
    ):
        return lx, ly, rx, ry
    else:
        return None, None, None, None


def drawGaussian(img, row, col_float):
    col_ceil = int(ceil(col_float))
    col_floor = int(floor(col_float))

    floor_gain = col_ceil - col_float
    ceil_gain = col_float - col_floor

    if row < img.shape[1]:
        img[col_ceil, row] = int(ceil_gain * 255)
        img[col_floor, row] = int(floor_gain * 255)

    return img


yaml_config_path = "laser_plane_virtual_image_generator.yaml"
with open(yaml_config_path, "r") as f:
    config = yaml.safe_load(f)
sc = StereoCamera(config["camera_calibration_file"])
laser_plane = config["laser_plane"]
amax = config["altitude"]["max"]
amin = config["altitude"]["min"]
n = config["altitude"]["step"]
n = int((amax - amin) / n) + 1
altitude = np.linspace(amin, amax, n)


output_folder = config.get("output_folder", "./virtual_images")
output_folder = Path(output_folder)
left_output = output_folder / "left"
right_output = output_folder / "right"
if not left_output.exists():
    left_output.mkdir(parents=True, exist_ok=True)
if not right_output.exists():
    right_output.mkdir(parents=True, exist_ok=True)


count = 0
xyzs = []
for alt in altitude:
    print("Altitude: ", alt)
    x_swath = 0.75 * alt
    x = np.linspace(-x_swath, x_swath, int(1.5 * sc.left.image_width))
    limg = np.zeros(
        (sc.left.image_height, sc.left.image_width), dtype=np.uint8
    )
    rimg = np.zeros(
        (sc.left.image_height, sc.left.image_width), dtype=np.uint8
    )
    for i in range(len(x)):
        point3d = [x[i], -laser_plane[3], alt]
        xyzs.append(point3d)
        # print(point3d)
        lp, rp = sc.project_point(point3d)
        # lx, ly, rx, ry = projectPoint(sc, point3d)
        if lp is not None and rp is not None:
            lx, ly = lp
            rx, ry = rp
            print(lx, ly, rx, ry)
            lx = int(round(lx))
            rx = int(round(rx))
            if lx >= sc.left.image_width or rx >= sc.right.image_width:
                continue
            limg = drawGaussian(limg, lx, ly)
            rimg = drawGaussian(rimg, rx, ry)
    # print(point2dl)
    # print(point2dl - point2dr)
    name = "{:07d}.png".format(count)
    lout = str(left_output / name)
    rout = str(right_output / name)
    cv2.imwrite(lout, limg)
    cv2.imwrite(rout, rimg)
    count += 1


xyzs = np.array(xyzs)
m, inliers = plane_fitting_ransac(
    xyzs, 0.01, 3, len(xyzs) * 0.8, 100, plot=False
)
scale = 1.0 / m[1]
print(np.array(m) * scale)
print("Inliers: ", len(inliers))

print("Finished!")
