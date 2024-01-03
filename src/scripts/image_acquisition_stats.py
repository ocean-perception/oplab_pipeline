"""Compute statistics on image acquisition near seafloor from nav file

Usage:
image_acquisition_stats.py [-h] [-t ALTITUDE_THRESHOLD] nav_file_path

positional arguments:
  nav_file_path

optional arguments:
  -h, --help            show this help message and exit
  -t ALTITUDE_THRESHOLD, --altitude_threshold ALTITUDE_THRESHOLD

Copyright (c) 2024, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=(
        "Compute statistics on image acquisition near seafloor from nav file, "
        "including number of images taken and bottom time."
    ),
)
parser.add_argument("nav_file_path", type=str, help="Navigation file path")
parser.add_argument(
    "-t",
    "--altitude_threshold_m",
    type=float,
    default=6,
    help="Altitude (m) below which images are considered to be near seafloor",
)
args = parser.parse_args()

if not Path(args.nav_file_path).exists():
    print(f"ERROR: Navigation file at {args.nav_file_path} does not exist")
    exit()

df = pd.read_csv(args.nav_file_path)

df_filtered = df[df["altitude [m]"] < args.altitude_threshold_m]
count = len(df_filtered)
average_altitude_m = df_filtered["altitude [m]"].mean()
bottom_time_s = df_filtered["timestamp [s]"].max() - df_filtered["timestamp [s]"].min()

start_bottom_time_s = df_filtered["timestamp [s]"].min()
end_bottom_time_s = df_filtered["timestamp [s]"].max()
images_during_bottom_time = len(
    df[
        (df["timestamp [s]"] >= start_bottom_time_s)
        & (df["timestamp [s]"] <= end_bottom_time_s)
    ]
)

print(f"Stats for {args.nav_file_path}")
print(f"Number of images at alt < {args.altitude_threshold_m}m: {count}")
print(
    f"Average altitude when at bottom (alt < {args.altitude_threshold_m}m): "
    f"{average_altitude_m:.3f}m"
)
print(
    f"Images between start (first time alt < {args.altitude_threshold_m}m) and "
    f"end (last time alt < {args.altitude_threshold_m}m) "
    f"of bottom time: {images_during_bottom_time}"
)
m, s = divmod(bottom_time_s, 60)
h, m = divmod(m, 60)
print(
    f"Bottom time (first time to last time alt < {args.altitude_threshold_m}m): "
    f"{h:.0f}h {m:.0f}m {s:.3f}s"
)
print(f"Average period: {(end_bottom_time_s - start_bottom_time_s)/count:.3f}s")
