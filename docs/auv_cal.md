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

Open the calibration.yaml (e.g. ./configuration/2018/fk180731/ae2000f/calibration/calibration.yaml) in a text editor. Set the camera_calibration and laser_calibration paths needed for the calibration routines you want to run the (processed) calibration image paths. Also set the camera calibration pattern or laser line extraction parameters as outlined below, and save the file.

### `camera_calibration`
Camera calibration is implemented using OpenCV. The algrithm supporst 3 types of calibration `pattern`s: "circles", "acircles" or "checkerboard". "circle" refers to the pattern where circles are aligned in rows and colums, whereas "acircles" is a pattern of circles where alternating columns are shifted down by half a circle-to-circle distance. See <https://docs.opencv.org/3.1.0/d4/d94/tutorial_camera_calibration.html> for more information on how the OpenCV camera calibration works.  

### `laser_calibration`
Use the laser_detection_tester of laser_bathyemtry (see <https://github.com/ocean-perception/laser_bathymetry?tab=readme-ov-file#determining-the-laser-extraction-parameters> for instructions) to identify the suitable `window_size`, `min_greenness_ratio`, `start_row` and`end_row`. Use  `start_row_b`, `end_row_b` and `two_lasers: true` if there are 2 laser lines. `num_columns` is for subsampling; e.g. `num_columns: 1024` will pick 1024 equally spaced out columns of pixels, rather than the entire image, to speed up processing. Set this to a value smaller or equal to the image width.  
`stratify` is for using the same number of points for each altitude band. Adapt the min and max values to the range within which there is data. Setting `min_z_m` too low or `max_z_m` too hight might cause it to pick up noisy detections, if there are any.


## Run `auv_cal`
Run `auv_cal` in the desired modes (`mono`, `stereo` or `laser`) with the path pointing to the "calibration" folder in the processed branch. This will compute the corresponding calibrations and save them as yaml files to the folder.

When you run the mono camera calibration, the reprojection error should be small (smaller than 1 pixel). A big reprojection error means the calibration is bad. A small reprojection error by iteself does not guarantee that the calibration is good; this is only the case if the photos of the pattern (those used in the final pass, after eliminating "bad" ones) cover a large area of the field of view.

The mono calibration code (potentially) runs the calibration multiple times. After the first run it identifies images of the calibration pattern that have a large reprojection error. This typically happens for blurred images, or images with lighting artefacts. If there are any such images, it then runs the calibration algorithm again, without the images of the calibration pattern  previsously flagged as bad ones. This is repeated multiple times, until the reprojection errors are sufficiently small.
This normally leads to a better calibration. However, in rare cases it can also lead to elminating too many or the wrong images of the calbiration pattern, and then overfitting to the few remaining images of the calibration pattern.
Therefore, when running the mono camera calibration, make sure the reprojection error is small and the final set of images of the calibration pattern the algorithm used covers the entire field of view.