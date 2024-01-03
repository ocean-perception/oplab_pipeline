"""Simplify nav files and filter to images in image folder

This script generates a simplified navigation file with only the columns that are
relevant for typical users.
The output navigation file only covers the images that are in the image folder.

Usage:
simplify_nav_file.py [-h] [-o OUTPUT] [-f] nav_file image_folder

positional arguments:
  nav_file              Navigation file (input; e.g. EKF based nav)
  image_folder          Path to image folder

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: navigation_data.csv)
  -f, --force           Overwrite output file if it exists (default: False)


Copyright (c) 2023, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import argparse
from pathlib import Path

import pandas as pd


def simplify_nav_file(nav_file, image_folder, output, force):
    if Path(output).exists() and not force:
        print(f"Output file {output} exists. Use -f to overwrite.")
        return

    # Read navigation file
    print("Reading navigation file")
    if not Path(nav_file).exists():
        print(f"ERROR: Navigation file at {nav_file} does not exist")
        return
    df = pd.read_csv(nav_file)
    if len(df) == 0:
        print(f"ERROR: Navigation file at {nav_file} is empty")
        return
    print(f"Number of entries in navigation file: {len(df)}")

    # Check filename extension
    print("Checking filename extension")
    fn_extension = None
    if len([p for p in Path(image_folder).glob("*.png")]) > 0:
        print("Image folder contains png files")
        fn_extension = ".png"
    if len([p for p in Path(image_folder).glob("*.jpg")]) > 0:
        print("Image folder contains jpg files")
        if fn_extension is not None:
            print("ERROR: Image folder contains both png and jpg files")
            return
        fn_extension = ".jpg"
    if not fn_extension:
        print("ERROR: Image folder does not contain png or jpg files")
        return

    # Get file list for the image folder
    print("Gathering image file list")
    image_files = [f for f in Path(image_folder).glob("*" + fn_extension)]

    # Only keep the stem from the column relative_path
    print("Generating filename column")
    df["filename"] = df["relative_path"].apply(lambda x: Path(x).stem + fn_extension)

    # Filter out the images that are not in the image folder
    print("Filtering out images that are not in the image folder")
    df = df[(df["filename"]).isin([f.name for f in image_files])]
    if len(df) == 0:
        print("ERROR: No images of input navigation file are in the image folder")
        return

    print("Sorting dataframe")
    # Sort the dataframe by timestamp
    df = df.sort_values(by="timestamp [s]")

    # Write output file
    print("Writing simplified and filtered nav file to disk")
    df.to_csv(
        output,
        index=False,
        columns=[
            "filename",
            "northing [m]",
            "easting [m]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "heading [deg]",
            "altitude [m]",
            "timestamp [s]",
            "latitude [deg]",
            "longitude [deg]",
        ],
    )

    # Sanity check data
    print("Sanity check data by comparing number of image files to entries in nav file")
    print(f"Number of images in generated nav file: {len(df)}")
    print(f"Number of images in image folder:       {len(image_files)}")
    if len(df) != len(image_files):
        print(
            "WARNING: Number of images in nav file does not match number of images in "
            "image folder"
        )

    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simplify navigation file. Based on DR or EKF nav file, but only outputting"
            " relevant columns, and only for images that are in the image folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("nav_file", help="Navigation file (input; e.g. EKF based nav)")
    parser.add_argument("image_folder", help="Path to image folder")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path",
        default="navigation_data.csv",
    )
    parser.add_argument(
        "-f", "--force", help="Overwrite output file if it exists", action="store_true"
    )
    args = parser.parse_args()

    simplify_nav_file(args.nav_file, args.image_folder, args.output, args.force)
