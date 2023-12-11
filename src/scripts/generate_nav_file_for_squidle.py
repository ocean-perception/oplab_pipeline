"""Generate nav file formatted for Squidle

This script generates a navigation file formatted for Squidle from a navigation file
in the oplab-pipeline format and a folder of images and thumbnails, if available.
The output navigation file only covers the images that are in the image folder.

Usage:
generate_nav_file_for_squidle.py [-h] [-o OUTPUT] [-f] nav_file image_folder

positional arguments:
  nav_file              Navigation file (e.g. EKF based nav)
  image_folder          Path to image folder

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path. Squidle expects the filename to be navdata.csv
                        (default: navdata.csv)
  -f, --force           Overwrite output file if it exists (default: False)

  
Definition of file format for Squidle:
https://squidle.org/wiki/general_information/datasources_and_ingesting_from_cloud_repositories

Copyright (c) 2023, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import argparse
from pathlib import Path

import pandas as pd


def generate_nav_file_for_squidle(nav_file, image_folder, output, force):
    if Path(output).exists() and not force:
        print(f"Output file {output} exists. Use -f to overwrite.")
        return

    # Read navigation file
    print("Reading navigation file")
    if not Path(nav_file).exists():
        print(f"ERROR: Navigatino file at {nav_file} does not exist")
        return
    df = pd.read_csv(nav_file)
    if len(df) == 0:
        print(f"ERROR: Navigation file at {nav_file} is empty")
        return

    # Only keep the stem from the column relative_path
    print("Generating keys (filename stem) column")
    df["key"] = df["relative_path"].apply(lambda x: Path(x).stem)

    # Get file list for the image folder
    print("Gathering image file list")
    image_files = [f for f in Path(image_folder).glob("*.png")]

    # Filter out the images that are not in the image folder
    print("Filtering out images that are not in the image folder")
    df = df[(df["key"] + ".png").isin([f.name for f in image_files])]
    if len(df) == 0:
        print("ERROR: No images in navigation file are in the image folder")
        return

    print("Sorting dataframe")
    # Sort the dataframe by timestamp
    df = df.sort_values(by="timestamp [s]")

    # Convert timestamps to datetime objects to enable ouput as
    # YYYY-MM-DD HH:MM:SS.ssssss
    print(
        "Converting epoch timestamps to datetime objects for conversion to human "
        "readable format. Check that timestamps in ouput file are indeed in UTC!"
    )
    df["timestamp_start"] = pd.to_datetime(df["timestamp [s]"], unit="s")

    # Rename columns
    print("Renaming columns")
    df = df.rename(
        columns={
            "northing [m]": "pose.data.northing_m",
            "easting [m]": "pose.data.easting_m",
            "depth [m]": "pose.dep",
            "roll [deg]": "pose.data.roll",
            "pitch [deg]": "pose.data.pitch",
            "heading [deg]": "pose.data.heading",
            "altitude [m]": "pose.alt",
            "latitude [deg]": "pose.lat",
            "longitude [deg]": "pose.lon",
        }
    )

    # Write output file
    print("Writing output nav file for Squidle to disk")
    df.to_csv(
        output,
        index=False,
        columns=[
            "key",
            "pose.lat",
            "pose.lon",
            "pose.dep",
            "pose.alt",
            "timestamp_start",
            "pose.data.northing_m",
            "pose.data.easting_m",
            "pose.data.roll",
            "pose.data.pitch",
            "pose.data.heading",
        ],
    )

    # Sanity check data
    print(
        "Sanity check data by comparing number of image files, thumbnails and entries "
        "in nav file"
    )
    print(f"Number of images in generated nav file: {len(df)}")
    print(f"Number of images in image folder:       {len(image_files)}")
    if len(df) != len(image_files):
        print(
            "WARNING: Number of images in nav file does not match number of images in "
            "image folder"
        )

    if Path(image_folder + "_thumbnails").exists():
        thumb_files = [f for f in Path(image_folder + "_thumbnails").glob("*.jpg")]
        print(f"Number of images in thumbnails folder:  {len(thumb_files)}")
        if len(thumb_files) != len(image_files):
            print(
                "WARNING: Number of thumbnails does not match number of images in "
                "image folder"
            )
        thumbnail_files_stems = [Path(f).stem for f in thumb_files]
        image_files_stems = [Path(f).stem for f in image_files]
        print("Checking if there are images without thumbnails...")
        images_without_thumbnail = 0
        for f in image_files_stems:
            if f + "_THM" not in thumbnail_files_stems:
                images_without_thumbnail += 1
                if images_without_thumbnail < 10:
                    print(f"WARNING: Image {f} does not have a thumbnail")
        if images_without_thumbnail == 0:
            print("...all images have thumbnails")
        else:
            print(f"WARNING: {images_without_thumbnail} images do not have a thumbnail")
    else:
        print(
            "Thumbnail folder does not exist. Create thumbnails first to include them "
            "in sanity check."
        )

    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("nav_file", help="Navigation file (e.g. EKF based nav)")
    parser.add_argument("image_folder", help="Path to image folder")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. Squidle expects the filename to be navdata.csv",
        default="navdata.csv",
    )
    parser.add_argument(
        "-f", "--force", help="Overwrite output file if it exists", action="store_true"
    )
    args = parser.parse_args()

    generate_nav_file_for_squidle(
        args.nav_file, args.image_folder, args.output, args.force
    )
