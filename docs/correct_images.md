# correct_images

correct_images has 3 commands:
- `parse`, which reads raw image files of filetypes `tif` and `raw` for Tuna Sand and Ae2000 dives respectively. Parse generates colour attenuation coefficients in the form of numpy arrays inside the parameters folder within the processed folder structure. the `parameters` folders are named after the corresponding camera systems to which the raw image files belong. Example `params_LC` for Tuna Sand dataset.
- `process`, which generates corrected images using the attenuation parameters from parse. corrected images are saved inside `develop` folders named after the corresponding camera systems to which the raw image files belong.
- `debayer`, which converts `raw` and `tif` bayer images to debayered `png` files using a particular bayer filter pattern.

`correct_images parse` usage:
```
correct_images parse [-h] [-F] path

positional arguments:
  path         Path to raw directory till dive.

optional arguments:
  -h, --help   show this help message and exit
  -F, --Force  Force overwrite if correction parameters already exist.
```
For parse the code looks for a configuration file `correct_images.yaml` inside the succesion of folders within configuration folder structure. If the code does not find a configuration file, it copies default `correct_images.yaml` from setup folder. The code reads camera systems, image format and image path from `mission.yaml` and automatically updates the default `correct_images.yaml` file copied into the configuration folder structure. At this point the code is ready to run the parse.

Currently, parse function iterates over all the camera systems mentioned in the `mission.yaml` file except for `LM165 (ae2000 dives)`

parse function automatically checks for altitude values from the auv_nav CSV file. If none of the altitudes are found between the range of altitude (`maximum and minimum altitude`) provided in `correct_image.yaml` file parse exits with an information message.


`correct_images process` usage:
```
correct_images process [-h] [-F] path

positional arguments:
  path         Path to processed directory till dive.

optional arguments:
  -h, --help   show this help message and exit
  -F, --Force  Force overwrite if processed images already exist.
```

process function checks value of `distortion_correction` flag in `correct_images.yaml` file. If the value is set to `False` code ignores path to camera parameters mentioned in `correct_images.yaml` file and directly continues with developing corrected images. It assumes a distortion correction.

Else if the value is set to `True` then the code looks for path to camera calibration parameters. If path is mentioned `None` then the code exits with an information message. path to camera parameters in the `correct_images.yaml` file should be provided as in the example below:

for the absolute path to camera calibration file `path/to/camera/parameters/mono_$camera_name$.yaml`
the path to be provided is `path/to/camera/parameters`. the process funstion appends `mono_$camera_name$.yaml` automatically.

`correct_images debayer` usage:
```
correct_images debayer [-h] [-p PATTERN] [-i IMAGE] [-o OUTPUT] path filetype

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
```

debayer function is to debayer the raw input image files even without going through the correction pipeline.

debayer function operates either in single file or batch mode. in single file mode path should include filename for the input image. batch mode works on all the images inside the path provided.

default bayer filter pattern used by code is `GRBG`. 

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

config:
  # path of the CSV file. mention only the 'dive_start_end_time' name
  auv_nav_path: 'json_renav_20180809_193000_20180809_235527'
  format: 'seaxerocks_3'
  # Camera system to which the target data set belongs---------------- 
  # Cam51707923 / Cam51707925 / LM165 for format: seaxerocks_3
  # LC / RC for format: acfr_standard
  camera1: 'RC'
  camera2: 'LC'

  # the index of processed images. If -1 is given, all images in filelist.csv are processed
  src_img_index: -1  # example: src_img_index: [100,101,102,103] or src_img_index: [99]

attenuation_correction:
  # the images captured at the altitude only in the range from altitude_min to altitude_max are used for calculating parameters.
  altitude:
    - max: 10
    - min: 4

  sampling_method: 'median' # sampling method from raw data. for local computers, 'median' is preferable because the calculation cost is low.
  median_filter_kernel_size: 3 # If the attenuation correction parameter looks noisy.
  
normalization:
  target_mean: 30 # target mean and standard deviation for pixel stat
  target_std: 3
  debayer_option: 'vng' # debayer option. 'linear', 'ea' or 'vng'. default = 'linear'

output:
  dst_file_format: 'png' # The format of developed images. Currently, only png can be given.
  
flags:
  apply_attenuation_correction: True # If false, the process will develop only pixel stat applied image
  apply_gamma_correction: True        
  apply_distortion_correction: False
  camera_parameter_file_path: None # needed if apply_distortion_correction is True
```
