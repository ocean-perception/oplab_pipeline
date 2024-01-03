"""Compute max brightness for each image in a folder

Usage:
compute_max_brightness.py [-h] image_folder [output]

positional arguments:
  image_folder  Folder with images to be analysed
  output        Path of CSV file with max brightness for each image to be written
                (default: image_brightness.csv)

optional arguments:
  -h, --help    show this help message and exit

Copyright (c) 2024, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import argparse
from pathlib import Path

import imageio
import pandas as pd
import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=(
        "Compute max brightness for each image in a folder. This can be used to find "
        "images with very bright spots, e.g. fish that were close to the strobes and "
        "cameras. These can then be excluded from the training of attenuation "
        "parameters, to avoid introducing artefacts. Use this in combination with "
        "find_images_with_artefacts.py. To avoid getting many false positives, apply "
        "this on images that have been corrected with a low mean value (e.g. 5)."
    ),
)
parser.add_argument("image_folder", type=str, help="Folder with images to be analysed")
parser.add_argument(
    "output",
    type=str,
    nargs="?",
    default="image_brightness.csv",
    help="Path of CSV file with max brightness for each image to be written",
)
args = parser.parse_args()

image_files = [f for f in Path(args.image_folder).glob("*.jpg")]
image_files.extend([f for f in Path(args.image_folder).glob("*.png")])

max_brightnesses = []
print(f"Computing max brightness for {len(image_files)} images")
for image_file in tqdm.tqdm(image_files, desc="Compute max brightness for each image"):
    img = imageio.v2.imread(image_file)
    # Convert to grayscale
    img = img.mean(axis=2)
    max_brightnesses.append(img.max())

df = pd.DataFrame({"path": image_files, "max_brightness": max_brightnesses})
df.to_csv(args.output, index=False)
