# auv_cal

## Usage
```
auv_cal [-h] {mono,stereo,laser} path
```

`auv_cal laser` needs the stereo calibration. If it cannot find it, it will automatically generate it by calling `auv_cal stereo`.
`auv_cal stereo` needs the mono calibration for the cameras. If it cannot find it, it will autmatically generate it by calling `auv_cal mono`.

`auv_cal` uses a configuration file named "calibration.yaml" in the corresponding subfolder in the "configuration" branch of the file structure. If it cannot find it, it will genereate a default one there.

## Image conversion
The images used for computing the calibration files need to be stored as RGB images, e.g. as jpg or png. If they only exist as raw files, first convert them to an RGB format. This can be done with `correct_images` of the oplab_pipeline, where manual setting setting of parameters might be advisable, especially for converting laser line images. However, as `correct_images` does not currently support ACPI debayering, better results are usually achieved by using `image_conversion` (https://github.com/ocean-perception/image_conversion) with the  `-a` / `--acpi` flag enabled. ACPI debayering leads to better results with sharp black to white transitions, as they typically appear on calibration targets. Save the results where they would normally be expected, i.e. in the correct subfolder of the "processed" branch of the folder structure. Do this for all images you need, e.g. calibration pattern images and / or laser line projection images (if they are not already stored as png or jpg).

## Set up folder for results and configuration file
Create a folder "calibration" in the "processed" branch of the folder structure, at the same level as the dive folders. If there are any calibration files you want to reuse (e.g. a mono calibration file generated for a previous deployment, that is also valid for the current deployment) place them in there.
Check if a file calibration.yaml exists in a "calibration" folder in the "configuration" branch of the folder structure. If the file or the folder doesn't exist, run `auv_cal` in any of the modes (`mono`, `stereo` or `laser`) with the path pointing to the "calibration" folder in the processed branch (e.g. `auv_cal laser processed/2018/fk180731/ae2000f/calibration`). This will create the calibration.yaml in the "configuration" branch of the folder structure, in a folder "configuration".

Open the calibration.yaml (e.g. ./configuration/2018/fk180731/ae2000f/calibration/calibration.yaml) in a text editor. Set the camera_calibration and laser_calibration paths needed for the calibration routines you want to run the (processed) calibration image paths. Also set the camera calibration pattern or laser line extraction parameters, and save the file.

## Run `auv_cal`
Run `auv_cal` in the desired modes (`mono`, `stereo` or `laser`) with the path pointing to the "calibration" folder in the processed branch. This will compute the corresponding calibrations and save them as yaml files to the folder.
