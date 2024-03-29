# correct_images

`correct_images` has 4 commands:
- `parse`, which reads raw image files of filetypes `tif` and `raw` for Tuna Sand and Ae2000 dives respectively. Parse generates colour attenuation coefficients in the form of numpy arrays inside the parameters folder within the processed folder structure. the `parameters` folders are named after the corresponding camera systems to which the raw image files belong. Example `params_LC` for Tuna Sand dataset.
- `process`, which generates corrected images using the attenuation parameters from parse. corrected images are saved inside `develop` folders named after the corresponding camera systems to which the raw image files belong.
- `correct`, which runs parse followed by process in one go. This can be used for small datasets which are being processed for the first time.
- `rescale`, which generates rescaled image for a target image scale with or without maintaining total number of pixels in original image

### correct_images `parse` usage: ###
```sh
correct_images parse [-h] [-F] [--suffix SUFFIX] path [path ...]

positional arguments:
  path             Folderpath where the (raw) input data is. Needs to be a
                   subfolder of 'raw' and contain the mission.yaml
                   configuration file.

optional arguments:
  -h, --help       show this help message and exit
  -F, --Force      Force overwrite if correction parameters already exist.
  --suffix SUFFIX  Expected suffix for correct_images configuration and output
                   folders.
```
For parse the code looks for a configuration file `correct_images.yaml` inside the succesion of folders within configuration folder structure. If the code does not find a configuration file, it copies default `correct_images.yaml` from setup folder. The code reads camera systems, image format and image path from `mission.yaml` and automatically updates the default `correct_images.yaml` file copied into the configuration folder structure. At this point the code is ready to run the parse.

`parse` can be run in two modes of correction:
- `colour_correction` : correction parameters are computed for each colour channel and target image is automatically balanced for colour and image stats (brightness and contrast)
- `manual_balance` : target images are developed by applying user provided intenisty gains and subtractors for each channel

A. Example configuration for `colour_correction` : 

```yaml
# Yaml 1.0
version: 2

method: 'colour_correction'

colour_correction :
  distance_metric : 'altitude' # ['none', 'altitude', 'depth_map'] 
  metric_path : 'json_renav_*' # json nav path in case of altitude / depth map path in case of depth map metric
  altitude_filter :
    parse : # only use images in the altitude range below to calculate attenuation_modelling parameters. Look at the histogram of the bins when running `correct_images parse` and use range where bins contain at least ~20 samples.
      min_m : 4.2
      max_m : 6.3
    process: # only convert the images in the altitude range below
      min_m : 2
      max_m : 12
  smoothing : 'median' # ['mean' / 'median' / 'mean_trimmed']
  window_size : 3 # increase if the attenuation correction parameter looks noisy.
```

Configuration fields to generate `colour_correction` parameters :

- `distance_metric` : form of distance values to be used for colour correction. options are [* `none`, `altitude`, ** `depth_map`] assumes constant altitude, flat seafloor, 3d structure respectively.
                    * Note: select this distance metric for `grey_world` corrections
                    ** Note: Run `https://github.com/ocean-perception/laser_bathymetry` for generating `depth_map`
- `metric_path` : path to the file containing distance values corresponding to source images.
                For `altitude` based colour corrections, provide the name of the `json_renav_***` folder name containing the `\csv\ekf\auv_ekf_<camera_name>.csv` file.
                For `depth_map` based colour corrections, provide the path to folder containing depth map `*.npy` files.
                Note: for `depth_map` place the depth maps folder inside the processed folder chain relative to the dive.
                Note: Please ensure the metric_path inside correct_images.yaml actually points to the desired json_renav* folder within the processed folder chain for the dive. By default the code assumes the first json_renav* within the processed folder chaing for the dive.
- `altitude_filter` : `parse` : set `min_m` and `max_m` as minimum and maximum altitudes (in meters) to filter images within the range of altitudes that should be used for training the attenuation parameters. Initially set it to a wide range and run `correct_images parse`. Look at the histogram of altitude bins and determine range where bins contain at least ~20 samples. Then use this range in the setting here.
- `altitude_filter` : `process` : set `min_m` and `max_m` as minimum and maximum altitudes (in meters) to filter images within the range of altitudes that should be convert from raw to corrected colour.
- `smoothing` : sampling colour intensity values from window_size to develop attenuation model. options are ['mean' / 'median' / 'mean_trimmed']
- `window_size` : is applicable if `smoothing` is set to `median`

B. Example configuration for `manual_balance` :

```yaml
# Yaml 1.0
version: 2

method: 'colour_correction'
...

cameras : 
  ... 
    manual_balance :
      subtractors_rgb : [0, 0, 0] 
      colour_gain_matrix_rgb : [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
```
Configuration fields to perform `manual_balance` :

- `subtractors_rgb` : -c values (zero capped) 
- `colour_gain` : diagonal terms corresponding to gain values for red, green and blue channels

C. Example configuration for `parse` for setting up cameras :

```yaml
cameras : 
  - camera_name : 'LC'
    image_file_list : 
      parse : 'none'
```

Configuration fields :

- `camera_name` : name of the cameras in a particular imaging system (acfr, ae2000, biocam). adhere to the camera names as per `mission.yaml`.
- `image_file_list`: `parse` : provide path to a filelist.txt containing filenames for the images you want to use for `parse`.
                             Note : put the filelist.txt within `configuration` folder chain where `correct_images.yaml` is found.


### `correct_images process` usage: ###
```sh
correct_images process [-h] [-F] [--suffix SUFFIX] path

positional arguments:
  path             Path to raw directory till dive.

optional arguments:
  -h, --help       show this help message and exit
  -F, --Force      Force overwrite if correction parameters already exist.
  --suffix SUFFIX  Expected suffix for correct_images configuration and output
                   folders.
```

`process` function processes images based on user settings provided in the `correct_images.yaml` file. This function depends on the parameters generated by `parse` in order to apply the corrections to the images.
The `process` workflow is illustrated below.

![](https://github.com/ocean-perception/oplab_pipeline/blob/develop/docs/images/process_workflow.png "Process Workflow")

`process` supports the following user selectable operations for finally developing images :
- `colour_correction` : Adjusts brightness and contrast for output images. Uses `colour_correction` parameters from parse based on particular distance metric chosen during `parse`.
- `manual_balance` : Uses `manual_balance` parameters in `correct_images.yaml` for performing manual balancing of colours in output images.
- `distortion correction` : Performs distortion correction for images by using calibration parameters corresponding to each camera.

A. Example configuration for `colour_correction` based processing :

```yaml
cameras :
  - camera_name : 'LC' #in line with vehicle.yaml
    ...
    colour_correction :
      brightness : 30 # % target across dataset
      contrast : 3 # % target accross dataset
```

Configuration fields :

- `colour_correction` : `brightness` : mean value of pixel intensities targeted across the full dataset of images selected for `process`.
- `colour_correction` : `contrast` : std value of pixel intensities targeted across the full dataset of images selected for `process`.

B. Example configuration for `distortion correction` based processing:

```yaml
output_settings :
  undistort : False
```
Configuration fields :

- `output_settings` : `undistort` : Set this variable in order to perform distortion corrections

C. Example configuration for `process` for setting up cameras :

```yaml
cameras :
  - camera_name : 'LC'
    image_file_list :
      ...
      process : 'none'
```

Configuration fields :

- `camera_name` : name of the cameras in a particular imaging system (acfr, ae2000, biocam). adhere to the camera names as per `mission.yaml`.
- `image_file_list` : `process` : provide path to a filelist.txt containing filenames for the images you want to use for `process`.
                             Note : put the filelist.txt within `configuration` folder chain where `correct_images.yaml` is found.
                             Note : this list may be the same as that was used for `parse` or it can be different.

### `correct_images correct` usage: ###
```sh
correct_images correct [-h] path

positional arguments:
  path                  Path to bayer images.

optional arguments:
-F, --Force  Force overwrite if processed images already exist.
```

Note : `correct` function may be used in case a small data set needs to be processed for the first time. `correct` function runs `parse` and `process` stages in one go. Hence all default settings will be assumed.

### `correct_images rescale` usage : ###
```sh
correct_images rescale [-h] path

positional arguments:
  path        Path to raw folder

optional arguments:
  -h, --help  show this help message and exit
```

`rescale` is used to upscale or downscale images based on distances of images from object, without losing aspect ratio. Original size of image in pixels may or may not be maintained.

A. Example configuration for `rescale` :

```yaml
rescale :
  - camera_name : 'LC'
    path : # path to images relative to processed folder
    distance_path : #
    interpolate_method : 'bicubic' # bicubic, nearest_neighbour, bilinear, lanczos
    target_pixel_size : 1 # in centimeter
    maintain_pixels : 'Y'
    output_folder : # output path relative to processed images
```
Configuration fields :

- `path` : path to processed images - `processed/.../developed_<camera_name>/<parse_subdirectory>/<process_subdirectory>/`.
         Note: for images which are not processed and generated by `correct_images` use the same folder structure as above.
- `distance_path` : path to json_renav* folder for auv_ekf_cameraname.csv file.
                  Note: for distances not generated by `auv_nav` keep the distance file as auv_ekf_cameraname.csv and maintain the same folder struture as above.
- `interpolate_method` : method of interpolation to be used while rescaling the intensity values.
- `target_pixel_size` : target pixel size for output images. Bigger pixel size means downscaling and vice versa.
- `maintain_pixels` : flag to be set if the rescaled imaged are to bear the same number of pixels as the original image.
- `output_folder` : folder for saving the rescaled images. output folder will be created relative to the processed chain.

Example of rescaled images :

Downscaled image with target pixel size of 1 cm from original pixel size of 1 mm.
Original number of pixels are maintained. Hence, the image appears blurred.

![](https://github.com/ocean-perception/oplab_pipeline/blob/develop/docs/images/PR_20180811_163514_163_LC16_down_maintain_yes.png)

Upscaled image with target pixel size of 0.3 mm from original pixel size of 1 mm.
Original number of pixels are maintained. Hence, the image appears cropped from center.

![](https://github.com/ocean-perception/oplab_pipeline/blob/develop/docs/images/PR_20180811_163514_163_LC16_up_maintain_yes.png)

## Output folder structures ##

Folder structure for parse output :
`../processed/year/cruise/platform/dive/image/camera/attenuation_correction/params_$camera_name$/<parse_subfolder>` where `parse_subfolder` names can be : `altitude_corrected`, `depthmap_corrected`, `greyworld_corrected`

Folder structure for process output : 
`../processed/year/cruise/platform/dive/image/camera/attenuation_correction/developed_$camera_name$/<parse_subfolder>/<process_subfolder>` where `process_subfolder` names are : `m*_std*` corresponding to the `brightness` and `contrast` values provided in `correct_images.yaml`


## Example of required `correct_images.yaml` File ##

The correct_images.yaml file should be placed in the following path
`../configuration/year/cruise/platform/dive/correct_images.yaml`

Note regarding the `altitude_filter` parameter: Set `min_m` and `max_m` to the range of altitudes where there are plenty of images.  When running `coorect_images` it prints the number of images per bin (sections of 0.1m). Set `min_m` and `max_m` to the range where each bin has a sufficient number of images (typically at least 10).

```yaml
# Yaml 1.0
version: 2

method: 'colour_correction'

colour_correction :
  # Distance metric can be
  #  uniform: assumes constant altitude
  #  altitude: assumes a flat seafloor
  #  depth_map: assumes 3d structure respectively
  # If using depth_maps, they need to be stored at a subdirectory called "depth_map" in
  # the dive processed folder with the same filename as the images
  distance_metric : 'altitude'
  metric_path : 'json_renav_*' # json nav path in case of altitude / depth map
  altitude_filter :
    parse : # only use images in the altitude range below to calculate attenuation_modelling parameters. Look at the histogram of the bins when running `correct_images parse` and use range where bins contain at least ~20 samples.
      min_m : 2
      max_m : 12
    process : # only convert the images in the altitude range below
      min_m : 2
      max_m : 12
  smoothing : 'median' # 'mean' / 'median' / 'mean_trimmed' sampling colour intensity values from window_size to develop model
  window_size : 3 # increase if the attenuation correction parameter looks noisy.

cameras : 
  - camera_name : 'LC' #in line with vehicle.yaml
    image_file_list : 
      parse : 'none' # Path of file in configuration folder containing list of images to be used. Only indicate this if you want to limit the images used, 'none' otherwise.
      process : 'none'  
    colour_correction :
      brightness : 30 # % target across dataset
      contrast : 3 # % target accross dataset  
    manual_balance :
      subtractors_rgb : [0, 0, 0] # -c values (zero capped)
      colour_gain_matrix_rgb : [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # diagonal terms corresponding to r gain, g gain, b gain
  
  - camera_name : 'RC' #in line with vehicle.yaml
    image_file_list : 
      parse : 'none' # Path of file in configuration folder containing list of images to be used. Only indicate this if you want to limit the images used, 'none' otherwise.
      process : 'none'
    colour_correction :
      brightness : 30 # % target across dataset
      contrast : 3 # % target accross dataset  
    manual_balance:
      subtractors_rgb: [0, 0, 0] # -c values (zero capped)
      colour_gain_matrix_rgb: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # diagonal terms corresponding to r gain, g gain, b gain
    
output_settings :
  undistort : False
  compression_parameter : 'png' # 'tiff'

rescale :
  - camera_name : 'LC'
    path : # path to images relative to processed folder
    distance_path : # path to json_renav* folder for auv_ekf_<cameraname>.csv file
    interpolate_method : 'bicubic' # bicubic, nearest_neighbour, bilinear, lanczos
    target_pixel_size : 1 # in centimeter
    maintain_pixels : 'Y'
    output_folder : # output path relative to processed images
```