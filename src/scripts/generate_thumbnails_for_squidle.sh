#/bin/bash/

# This script generates thumbnails for all images (jpg or png) in a directory and writes 
# them to a folder next to the original folder, ending in _thumbnails.
# Requires ImageMagick to be installed.
#
# Usage: ./generate_thumbnails_for_squidle.sh PATH_TO_FOLDER_WITH_IMAGES
#
#
# Copyright (c) 2024, University of Southampton
# All rights reserved.
# Licensed under the BSD 3-Clause License.
# See LICENSE.md file in the project root for full license information.


d="$1"

# Check if there is any input provided
if [[ -z "$d" ]];
    then echo "ERROR: No input provided. Usage: ./generate_thumbnails_for_squidle.sh PATH_TO_FOLDER_WITH_IMAGES";
    exit 1;
fi

if [[ "$d" == "-h" ]] || [[ "$d" == "--help" ]];
    then echo "Usage: ./generate_thumbnails_for_squidle.sh PATH_TO_FOLDER_WITH_IMAGES";
    exit 0;
fi

# Check if there are any png files in input folder. Abort if not.
pngs=$(find $d -maxdepth 1 -iname '*.png' )
jpgs=$(find $d -maxdepth 1 -iname '*.jpg' )
if [[ -z "$jpgs" ]] && [[ -z "$pngs" ]];  # `-z` returns true if argument is an empty string or an uninitialized variable
    then echo "ERROR: No png or jpg files found in input folder (${d}).";
    exit 1
fi

if [[ ! -z "$jpgs" ]] && [[ ! -z "$pngs" ]];
    then echo "WARNING: There is a mix of png or jpg files in the input folder (${d}).";
fi

# Check if there are already jpg files in the _thumbnails folder. Abort if yes.
thumbnail_dir_relative="${d}_thumbnails"
if [[ -d "$thumbnail_dir_relative" ]]
    then
    jpg_thumbnails=$(find $thumbnail_dir_relative -maxdepth 1 -name "*.jpg")
    if [[ ! -z "$jpg_thumbnails" ]]
        then echo "ERROR: There are already jpg files in the output folder (${thumbnail_dir_relative}). Aborting to prevent overwriting.";
        exit 1
    fi
fi

# Remember where we are, then `cd` into target directory
original_dir="$(pwd)"
cd $d

# Generate thumbnails
echo "Converting images..."
files=$(find -maxdepth 1 \( -iname '*.jpg' -o -iname '*.png' \))
nb_original_files=$(echo "$files" | wc -l)
thumbnail_dir="$(pwd)_thumbnails"
mkdir -p "${thumbnail_dir}" &&\
echo "$files" | parallel --eta mogrify -resize 400X338 -format jpg -path "${thumbnail_dir}" {} ||\
    { echo "ERROR: Something went wrong while attempting to convert image(s)."; cd $original_dir; exit 1; }
echo "...done converting images."

echo "Renaming thumbnails..."
cd $thumbnail_dir
for filename in *.jpg; do mv $filename ${filename%.jpg}_THM.jpg; done;
echo "...done renaming thumbnails"

# `cd` back to original directory 
cd $original_dir

echo "Sanity checking: comparing number of files in original and in thumbnail folder:"
echo "Number of original image files: ${nb_original_files}"
nb_thumbnail_files=$(find $thumbnail_dir -maxdepth 1 -name "*.jpg" | wc -l)
echo "Number of thumbnail files: ${nb_thumbnail_files}"
if [ ${nb_original_files} -ne ${nb_thumbnail_files} ]; then
    echo "ERROR: Number of original files and number of thumbnail files do not match."
    exit 1
fi

echo "Done."
