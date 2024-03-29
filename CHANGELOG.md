# Changelog

## Since 0.1.11
 - [auv_nav] Fix error with loading YAML matrices as strings
 - [auv_nav] Read datetime from data entries on AE2000f parser

## 0.1.11 (2021-11-30)
 - [auv_nav] Add support for ALR data
 - [auv_nav] Increase number of parameters that can be set: different uncertainties (stdev) for x, y and z velocities, Mahalanobis distance threshold and (de)activating of Kalman smoother (via auv_nav.yaml), as well as verbosity (via CLI)
 - [auv_nav] Generate plots of EKF / EKS processed positions, orientatiions and velocities with unvertainties (stdev) against time. Plot rejected sensor measurements.
 - [auv_nav] Change EKF to use sensor measurements at timestamps when they were recorder instead of using interpolated values
 - [auv_nav] Compute relative postion / angular uncertainty between 2 points
 - First dockerfile and singularity definition files released.

## 0.1.10 (2021-08-23)
 - Improve support for NTNU data
 - [correct_images] Add support for BioCam4000_15C

## 0.1.9 (2021-06-22)
 - Add NTNU ROV parser (incl. EIVA Navipac)

## 0.1.8 (2021-01-28)
 - Add GitHub Actions CI
 - correct_images now saves gains and attenuation parameter plots

## 0.1.7 (2021-01-28)
 - auv_nav convert has been renamed to auv_nav export and import
 - hybis format is not supported as import
 - Improved correct_images processing pipeline to use less hard drive and RAM.
 
## 0.1.6 (2020-12-08)
 - Sensors are initialised to None by default
 - Removed unused covariance outputs for DR
 - Standard deviation outputs in CSV
 - Covariance CSVs for EKF
 - Added tool to scale images
 - Corrected bayer pattern bugs

## 0.1.0 (2020-06-01)
 - First public release
 - Added changelog
 - Changed CSV headers to lowercase
 - Camera CSV files contain the relative path to the images from the dive folder
 - Store memmaps from correct_images in the processed folder
 - Added parser template at [auv_nav/parsers/parser_template.py](auv_nav/parsers/parser_template.py)
 - Added documentation at [read the docs](oplab-pipeline.readthedocs.io).

## 1.1.0 (2023-03-27)
- [mission.yaml]  Update mission.yaml definition to support records_laser field in camera nodes, but no longer have type and bit_depth fields, as this information is already contained in correct_images.yaml
- [auv_nav]        Add support for payload node to generate poses for sensors mounted to AUV
- [auv_nav]        Improve output to terminal
- [correct_images] Update correct_images.yaml definition to allow for different altitude filters in parse and processed
- [correct_images] Fix issue where memmaps were not deleted after use
- [correct_images] Fix issue where correct_images process could not be run without force (-F) flag
- [correct_images] Fix issue where processed folders were generated in parse mode
- [Dockerfile]     Fix issue that caused it to be unable to determine the version of oplab_pipeline
