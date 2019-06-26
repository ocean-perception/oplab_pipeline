# README - correct_images

correct_images is a pipeline for developing visual images from raw data.

Execution with the 'parse' option estimates the parameters for developing, for example, light attenuation curve parameters, mean and standard deviation values of each pixel in dataset.

Execution with the 'process' option develops png format images form rawdata, corrected with the parameters estimated by 'parse' option.

correct_images assumes
- Grey-world assumption
- While gathering the images in the dataset, the posture of camera on AUV is always nearly horizontal
- The seafloor is nearly flat

The correction result would be skewed if the dataset does not match these assumptions.

## Downloading and Updating the code ##

To download the code, go to directory you want it to be in, open a terminal/command prompt there and type 
```
git clone https://github.com/ocean-perception/correct_images.git
```

To push updates, stage changes, commit and push to a branch, usually master
```
git add -u
git commit -m "Some message about the change"
git push origin master
```

## Dependencies

Dependencies are:
- matplotlib==3.0.3
- joblib==0.13.2
- numpy==1.16.2
- opencv_python==4.0.0.21
- colour_demosaicing==0.1.4
- imageio==2.5.0
- scipy==1.2.1
- pandas==0.24.2
- tqdm==4.31.1
- Pillow==6.0.0
- PyYAML==5.1



## Usage

For new dataset, you have to execute the code with 'parse option'.

```python correct_images parse yaml_file_path```

In the yaml file in the argument, you have to designate the path of the raw image files (.raw format for AE2000 or .tif format for tunasand) and the path of the navigation file that includes altitude data (the output csv file of auv_nav).

Once you have finished 'parse', the code save the parameters for attenuation correction and pixel stat data in the subdirectory of '/processed/'. You can develop png images with arbitrary parameters (mean and std) with 'process' option.

```python correct_images process yaml_file_path```

See the statement below and the samples of yaml files in 'missions/' directory.

## yaml file format (for 'parse')

**auv_nav_filepath: string.**

CSV file path which contains image raw file name with altitude information.
The output csv file of auv_nav is suitable.

**src_file_dirpath: string.**

Directory path which contains raw image data (.raw format for AE2000 and .tif format for tunasand)

**src_file_format: string.**

Set '.raw' format for AE2000 and '.tif' format for tunasand.

**camera_lr: 'LC' or 'RC'. only for tunasand camera. default -'LC'**

Designate 'LC' for processing Left Camera image of 'RC' for Right Camera image, when you will process tunasand image.


**If the estimated parameters look bad or the code would not work on your dataset format, there are more parameters which can be tuned. Please check sample file in */configuration_samples***

## yaml file format (for 'process')

**params_dir_path: string.**

Directory path which includes the result of 'parse'.
The path would end with */attenuation_correction/params*

**target_mean: float. default: 30**
The target mean value of developed png images, scaled into 0 to 100.
If the target_mean is 30, means of all pixel values in developed image would be 255*30/100=76.5.

**target_std: float. default: 5**
The target std value of developed png images, scaled into 0 to 100.
If the target_std is 30, stds of all pixel values in developed image would be 255*5/100=12.75.

**If the developed images look bad or the code would not work on your dataset format, there are more parameters which can be tuned. Please check sample file in */configuration_samples***
