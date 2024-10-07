#/bin/bash/

# This script generates symlinks to raw images from which their jpg or png images have been generated.
#
# The script looks up all the jpg or png images in a directory (usually the processed image folder of a dataset)
# and generates symlinks to files with the same stems but the file extension "tif" in the raw image folder.
# This is useful for generating the folder tree to be zipped and submitted to a data repository, such as BODC.
#
# Usage: generate_symlinks_to_raw_images.sh PROCESSED_IMAGES_FOLDER RAW_IMAGES_FOLDER DESTINATION_FOLDER
#
#
# Copyright (c) 2024, University of Southampton
# All rights reserved.
# Licensed under the BSD 3-Clause License.
# See LICENSE.md file in the project root for full license information.


processed_images="$1"  # Input directory containing jpg or png images
raw_images="$2"        # Input directory containing raw images
destination="$3"       # Destination directory for symlinks. This directory will be created if it does not exist.

usage="Usage: generate_symlinks_to_raw_images.sh PROCESSED_IMAGES_FOLDER RAW_IMAGES_FOLDER DESTINATION_FOLDER"

# Check if there is any input provided
if [[ -z "$destination" ]];
    then echo -e "ERROR: Missing arguments.\n$usage";
    exit 1;
fi

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]];
    then echo $usage;
    exit 0;
fi

if [[ ! -d "$processed_images" ]];
    then echo -e "ERROR: Cannot find image folder $processed_images";
    exit 1;
fi

raw_images=${raw_images%/}  # Remove trailing slash if present
if [[ ! -d "$raw_images" ]];
    then echo -e "ERROR: Cannot find raw image folder $raw_images";
    exit 1;
fi

tifs=$(find $raw_images -maxdepth 1 -iname '*.tif' )
if [[ -z "$tifs" ]];
    then echo "ERROR: No tif files found in raw image folder (${raw_images}). Note: this script assumes raw files are saved as tif files.";
    exit 1
fi

# Check if there are any png files in input folder. Abort if not.
pngs=$(find $processed_images -maxdepth 1 -iname '*.png' )
jpgs=$(find $processed_images -maxdepth 1 -iname '*.jpg' )
if [[ -z "$jpgs" ]] && [[ -z "$pngs" ]];  # `-z` returns true if argument is an empty string or an uninitialized variable
    then echo "ERROR: No png or jpg files found in input folder (${processed_images}).";
    exit 1
fi

if [[ ! -z "$jpgs" ]] && [[ ! -z "$pngs" ]];
    then echo "ERROR: There is a mix of png or jpg files in the input folder (${processed_images}).";
    exit 1
fi

if [[ -d "$destination" ]]
    then
    raws=$(find $destination -maxdepth 1 -iname '*.tif' )
    if [[ ! -z "$raws" ]];
        then echo "ERROR: There are already tif files (or symlinks to tif files) in the output folder (${destination}).";
        exit 1
    fi
fi

# Remember where we are, then `cd` into destination directory
original_dir="$(pwd)"
mkdir -p $destination
cd $destination || { echo "ERROR: Could not `cd` into destination directory."; exit 1; }

# Generate symlinks
echo "Generating symlinks..."
paths=$(find ${processed_images} -maxdepth 1 \( -iname '*.jpg' -o -iname '*.png' \))
stems=$(echo "$paths" | sed 's/.*\///' | sed 's/\.[^.]*$//')
echo "$stems" | xargs -I % ln -s ${raw_images}/%.tif %.tif

# `cd` back to original directory 
cd $original_dir

# Check symlinks
invalid_links=$(find "$destination" -type l ! -exec test -e {} \; -print)
if [[ ! -z "$invalid_links" ]];
    then echo -e "ERROR: There are invalid symlinks:\n$invalid_links";
    exit 1
fi
echo "... done generating symlinks."

echo "Done!"
