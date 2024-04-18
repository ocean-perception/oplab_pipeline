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
import math
from pathlib import Path

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=(
        "Compute statistics on image acquisition near seafloor from nav file, "
        "including number of images taken, bottom time and distance and surface area "
        "covered."
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
parser.add_argument("-s", "--start_time", type=float, help="Start time (epoch, s).")
parser.add_argument("-e", "--end_time", type=float, help="End time (epoch, s).")
parser.add_argument(
    "-a", "--alpha", type=float, default=70.34, help="Camera field of view width (deg)."
)
args = parser.parse_args()

if not Path(args.nav_file_path).exists():
    print(f"ERROR: Navigation file at {args.nav_file_path} does not exist")
    exit()

df = pd.read_csv(args.nav_file_path)
if len(df) == 0:
    print(f"ERROR: Navigation file at {args.nav_file_path} is empty")
    exit()

if args.start_time is not None:
    df = df[df["timestamp [s]"] >= args.start_time]
if args.end_time is not None:
    df = df[df["timestamp [s]"] <= args.end_time]

if len(df) == 0:
    print("ERROR: No images between indicated start and end time")
    exit()

df_filtered = df[df["altitude [m]"] < args.altitude_threshold_m]
count = len(df_filtered)
if count == 0:
    print(f"ERROR: No images at alt < {args.altitude_threshold_m}m")
    exit()
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

df_filtered["delta_northing [m]"] = df_filtered["northing [m]"].diff()
df_filtered["delta_easting [m]"] = df_filtered["easting [m]"].diff()
df_filtered["delta_distance [m]"] = (
    df_filtered["delta_northing [m]"] ** 2 + df_filtered["delta_easting [m]"] ** 2
) ** 0.5
df_filtered["delta_surface [m2]"] = (
    df_filtered["delta_distance [m]"]
    * df_filtered["altitude [m]"]
    * math.tan(math.radians(args.alpha / 2))
    * 2
)

start_time_str = ""
if args.start_time is not None:
    start_time = pd.to_datetime(args.start_time, unit="s")
    start_time_str = start_time.strftime(" from %Y-%m-%d %H:%M:%S")
end_time_str = ""
if args.end_time is not None:
    end_time = pd.to_datetime(args.end_time, unit="s")
    end_time_str = end_time.strftime(" up to %Y-%m-%d %H:%M:%S")

print(f"Stats for {args.nav_file_path}{start_time_str}{end_time_str}")
print(
    f"Number of images at alt < {args.altitude_threshold_m}m:                    {count}"
)
print(
    f"Average altitude when at bottom (alt < {args.altitude_threshold_m}m):      "
    f"{average_altitude_m:.3f}m"
)
print(
    f"Total distance travelled at bottom (alt < {args.altitude_threshold_m}m):   "
    f"{df_filtered['delta_distance [m]'].sum():.0f}m"
)
print(
    f"Total surface area covered at bottom (alt < {args.altitude_threshold_m}m): "
    f"{df_filtered['delta_surface [m2]'].sum():.0f}m2"
)
m, s = divmod(bottom_time_s, 60)
h, m = divmod(m, 60)
print(
    f"Bottom time (first time to last time alt < {args.altitude_threshold_m}m):  "
    f"{h:.0f}h {m:.0f}min {s:.0f}s"
)
print(f"Number of images during bottom time:             {images_during_bottom_time}\n")
