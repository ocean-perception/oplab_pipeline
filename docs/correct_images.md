# correct_images

correct_images has 4 commands:
- `parse`, which reads raw image files of filetypes `tif` and `raw` for Tuna Sand and Ae2000 dives respectively. Parse generates colour attenuation coefficients in the form of numpy arrays inside the parameters folder within the processed folder structure. the `parameters` folders are named after the corresponding camera systems to which the raw image files belong. Example `params_LC` for Tuna Sand dataset.
- `process`, which generates corrected images using the attenuation parameters from parse. corrected images are saved inside `develop` folders named after the corresponding camera systems to which the raw image files belong.
- `debayer`, which converts `raw` and `tif` bayer images to debayered `png` files using a particular bayer filter pattern.
- `correct`, which runs parse followed by process in one go. This can be used for small datasets which are being processed for the first time.
- `rescale`, which generates rescaled image for a target image scale with or without maintaining total number of pixels in original image

correct_images `parse` usage:
```
correct_images parse [-h] [-F] path

positional arguments:
  path         Path to raw directory till dive.

optional arguments:
  -h, --help   show this help message and exit
  -F, --Force  Force overwrite if correction parameters already exist.
```
For parse the code looks for a configuration file `correct_images.yaml` inside the succesion of folders within configuration folder structure. If the code does not find a configuration file, it copies default `correct_images.yaml` from setup folder. The code reads camera systems, image format and image path from `mission.yaml` and automatically updates the default `correct_images.yaml` file copied into the configuration folder structure. At this point the code is ready to run the parse.

`parse` can be run in two modes of correction:

(1) `colour_correction` : correction parameters are computed for each colour channel and target image is automatically balanced for colour and image stats (brightness and contrast)

(2) `manual_balance` : target images are developed by applying user provided intenisty gains and subtractors for each channel

A. Example configuration for `colour_correction` : 

```
# Yaml 1.0
version: 1

method: 'colour_correction'

colour_correction :
  distance_metric : 'altitude' # ['none', 'altitude', 'depth_map'] 
  metric_path : 'json_renav_*' # json nav path in case of altitude / depth map path in case of depth map metric
  altitude_filter : # only use images in the altitude range below to calculate attenuation_modelling parameters
    max_m : 12
    min_m : 4
  smoothing : 'median' # ['mean' / 'median' / 'mean_trimmed']
  window_size : 3 # increase if the attenuation correction parameter looks noisy.
  curve_fitting_outlier_rejection : False
```

Configuration fields to generate `colour_correction` parameters :

`distance_metric` : form of distance values to be used for colour correction. options are [* `none`, `altitude`, ** `depth_map`] assumes constant altitude, flat seafloor, 3d structure respectively.
                    * Note: select this distance metric for `grey_world` corrections
                    ** Note: Run `https://github.com/ocean-perception/laser_bathymetry` for generating `depth_map`

`metric_path` : path to the file containing distance values corresponding to source images.
                For `altitude` based colour corrections, provide the name of the `json_renav_***` folder name containing the `\csv\ekf\auv_ekf_<camera_name>.csv` file.
                For `depth_map` based colour corrections, provide the path to folder containing depth map `*.npy` files.
                Note: for `depth_map` place the depth maps folder inside the processed folder chain relative to the dive.
                Note: Please ensure the metric_path inside correct_images.yaml actually points to the desired json_renav* folder within the processed folder chain for the dive. By default the code assumes the first json_renav* within the processed folder chaing for the dive.

`altitude_filter` : set `max_m` and `min_m` as maximum and minimum altitude (in meters) to filter images which are too far from or too close to the seafloor.
                    Note: the altitude range is used only to generate correction parameters and are not used in `process` step for `correct_images`.

`smoothing` : sampling colour intensity values from window_size to develop attenuation model. options are ['mean' / 'median' / 'mean_trimmed']

`window_size` : is applicable if `smoothing` is set to `median`

`curve_fitting_outlier_rejection` : set this variable if filtering the attenuation parameters is needed.
                                    Note: code ignores `smoothing` and `window_size` if this variable is `False`.

B. Example configuration for `manual_balance` :

```
# Yaml 1.0
version: 1

method: 'colour_correction'
...

cameras : 
  ... 
    manual_balance :
      subtractors_rgb : [0, 0, 0] 
      colour_gain_matrix_rgb : [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
```
Configuration fields to perform `manual_balance` :

`subtractors_rgb` : -c values (zero capped) 
`colour_gain` : diagonal terms corresponding to gain values for red, green and blue channels

C. Example configuration for `parse` for setting up cameras :

```
cameras : 
  - camera_name : 'LC'
    image_file_list : 
      parse : 'none'
```

Configuration fields :

`camera_name` : name of the cameras in a particular imaging system (acfr, ae2000, biocam). adhere to the camera names as per `mission.yaml`.
`image_file_list`: `parse` : provide path to a filelist.txt containing filenames for the images you want to use for `parse`.
                             Note : put the filelist.txt within `configuration` folder chain where `correct_images.yaml` is found.



`correct_images process` usage:
```
correct_images process [-h] [-F] path

positional arguments:
  path         Path to processed directory till dive.

optional arguments:
  -h, --help   show this help message and exit
  -F, --Force  Force overwrite if processed images already exist.
```

`process` function processes images based on user settings provided in the `correct_images.yaml` file. this function depends on the parameters generated by `parse` in order to apply the corrections to the images.
!(../correct_images/docs/process_workflow.png?raw=true "Process workflow")

Else if the value is set to `True` then the code looks for path to camera calibration parameters in the processed folder structure.

`correct_images debayer` usage:
```
correct_images debayer [-h] [-p PATTERN] [-i IMAGE] [-o OUTPUT] [-o_format OUTPUT_FORMAT] path filetype

positional arguments:
  path                  Path to bayer images.
  filetype              type of image: raw / tif / tiff

optional arguments:
  -h, --help            show this help message and exit
  -p PATTERN, --pattern PATTERN
                        Bayer pattern (GRBG for Unagi, BGGR for BioCam)
  -i IMAGE, --image IMAGE
                        Single raw image to test.
  -o OUTPUT, --output OUTPUT
                        Output folder.
  -o_format OUTPUT_FORMAT, --output_format OUTPUT_FORMAT
                        Output image format. 
```

debayer function is to debayer the raw input image files even without going through the correction pipeline.

debayer function operates either in single file or bat-F, --Force  Force overwrite if processed images already exist.ch mode. in single file mode path should include filename for the input image. batch mode works on all the images inside the path provided.

default bayer filter pattern used by code is `GRBG`.

default value for output image format is `png`.

`correct_images correct` usage:
```
correct_images correct [-h] path

positional arguments:
  path                  Path to bayer images.

optional arguments:
-F, --Force  Force overwrite if processed images already exist.
```
```
Note:
      (1) correct function may be used in case a small data set needs to be processed for the first time. correct function runs parse and process stages in one go.
```

## Output folder structures ##
```
`Folder structure for parse output`
../processed/year/cruise/platform/dive/image/camera/attenuation_correction/params_$camera_name$/

`Folder structure for process output`
../processed/year/cruise/platform/dive/image/camera/attenuation_correction/developed_$camera_name$/
```

## Example of required `correct_images.yaml` File ##

the correct_images.yaml file should be placed in the following path
`../configuration/year/cruise/platform/dive/correct_images.yaml`

```
#YAML 1.0

version: 1

method: 'colour_correction'

colour_correction :
  distance_metric : 'altitude' # ['none', 'altitude', 'depth_map'] assumes constant altitude, flat seafloor, 3d structure respectively. depth_map requires **** to have been run
  metric_path : 'json_renav_*' # json nav path in case of altitude / depth map path in case of depth map metric
  altitude_filter : # only use images in the altitude range below to calculate attenuation_modelling parameters
    max_m : 12
    min_m : 4
  smoothing : 'median' # 'mean' / 'median' / 'mean_trimmed' sampling colour intensity values from window_size to develop model
  window_size : 3 # increase if the attenuation correction parameter looks noisy.
  curve_fitting_outlier_rejection : False

cameras : 
  - camera_name : 'LC' #in line with vehicle.yaml
    image_file_list : 'none' # Only inidicate this if you want to limit the images used. folder is in the image path  
    colour_correction :
      brightness : 30 # % target across dataset
      contrast : 3 # % target accross dataset  
    manual_balance :
      subtractors_rgb : [0, 0, 0] # -c values (zero capped)
      colour_correction_matrix_rgb : [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # diagonal terms corresponding to r gain, g gain, b gain
  
  - camera_name : 'RC' #in line with vehicle.yaml
    image_file_list : 'none' # Only inidicate this if you want to limit the images used
    colour_correction :
      brightness : 30 # % target across dataset
      contrast : 3 # % target accross dataset  
    manual_balance:
      subtractors_rgb: [0, 0, 0] # -c values (zero capped)
      colour_correction_matrix_rgb: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # diagonal terms corresponding to r gain, g gain, b gain
    
output_settings :
  undistort : False
  compression_parameter : 'png' # 'tiff'
```
Note: image_file_list option is for user to provide a partial list of images for each camera to be processed. The code supports csv files of the same format as that of auv_ekf_cameraname.csv in the json_renav* folders.
