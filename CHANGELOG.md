# Changelog

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
