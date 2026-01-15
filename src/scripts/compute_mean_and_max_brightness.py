"""Compute mean and max brightness for each image in a folder

Usage:
compute_max_brightness.py [-h] [-f] image_folder [output]

Compute mean and max brightness for each image in a folder.
This can be used to find images with very bright spots, e.g. fish that were close to the
strobes and cameras, or dark images, e.g. where the strobes did not trigger.
These can then be excluded from the training of attenuation parameters, to avoid 
introducing artefacts. Use this in combination with find_images_with_artefacts.py.
Note: To avoid getting many false positives for too bright images, apply this on images
that have been corrected by correct_images with a low mean value (e.g. 5).

positional arguments:
  image_folder  Folder with images to be analysed
  output        Path of CSV file with mean and max brightness for each image to be 
                written (default: image_brightness.csv)

optional arguments:
  -h, --help    show this help message and exit
  -f, --force   Overwrite existing output file (default: False)

Copyright (c) 2024-2026, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import argparse
from pathlib import Path

import imageio
import pandas as pd
import tqdm
import plotly.graph_objects as go
import plotly.offline as py
# import plotly.express as px

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=(
        "Compute mean and max brightness for each image in a folder. "
        "This can be used to find images with very bright spots, e.g. fish that were "
        "close to the strobes and cameras, or dark images, e.g. where the strobes did "
        "not trigger. "
        "These can then be excluded from the training of attenuation parameters, to "
        "avoid introducing artefacts. Use this in combination with "
        "find_images_with_artefacts.py. "
        "Note: To avoid getting many false positives for too bright images, apply this "
        "on images that have been corrected by correct_images with a low mean value "
        "(e.g. 5)."
    ),
)
parser.add_argument("image_folder", type=str, help="Folder with images to be analysed")
parser.add_argument(
    "output",
    type=str,
    nargs="?",
    default="image_brightness.csv",
    help="Path of CSV file with mean and max brightness for each image to be written",
)
parser.add_argument(
    "-f", "--force", action="store_true", help="Overwrite existing output file"
)
args = parser.parse_args()

output_path = Path(args.output)
if output_path.suffix.lower() != ".csv":
    print("Output file must have .csv extension")
    exit(1)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path_html = output_path.with_suffix(".html")
if output_path.exists() and not args.force:
    print(f"Output file {args.output} already exists. Use --force to overwrite.")
    exit(1)

image_files = [f for f in Path(args.image_folder).glob("*.jpg")]
image_files.extend([f for f in Path(args.image_folder).glob("*.png")])

mean_brightnesses = []
max_brightnesses = []
print(f"Computing max brightness for {len(image_files)} images")
for image_file in tqdm.tqdm(image_files, desc="Compute max brightness for each image"):
    img = imageio.v2.imread(image_file)
    # Convert to grayscale
    img = img.mean(axis=2)
    mean_brightnesses.append(img.mean())
    max_brightnesses.append(img.max())

df = pd.DataFrame({
    "path": [str(f) for f in image_files],
    "mean_brightness": mean_brightnesses,
    "max_brightness": max_brightnesses
})
df.to_csv(output_path, index=False, float_format="%.3f")

# Plot dataframe with plotly as time series, with image path as tooltips
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        y=df["mean_brightness"],
        name="Mean Brightness",
        hovertext=df["path"],
    )
)
fig.add_trace(
    go.Scatter(
        y=df["max_brightness"],
        name="Max Brightness",
        hovertext=df["path"],
    )
)
fig.update_layout(
    title="Image Brightness Analysis",
    xaxis_title="Image Index",
    yaxis_title="Brightness",
)
py.plot(
    fig,
    filename=str(output_path_html),
    auto_open=False,
)

# fig = px.line(
#     df,
#     y=["mean_brightness", "max_brightness"],
#     labels={"value": "Brightness", "index": "Image Index"},
#     title="Image Brightness Analysis",
#     hover_data={"path": True},
# )
# fig.write_html(str(output_path_html))

print(f"Results saved to {output_path} and {output_path_html}")
