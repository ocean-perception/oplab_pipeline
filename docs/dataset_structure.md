# Dataset structure
All datasets share a common folder structure from a root folder. In this example, the root folder will be `/data`, but can be any folder or subfolder structure. From this point, the dataset must comply with the following convention:
`/data/raw` : Raw data from the sensors is stored in its subfolders
`/data/configuration` : Software configuration to process the data is stored here. If none is provided, a default configuration is written.
`/data/processed` : Processed navigation and imaging data is written here.
Then these three folders follow the same convention as well, with the following nested directories:

    year/cruise/platform/YYYYMMDD_hhmmss_platform_sensor_data

For example, if we go on a cruise named YK209, with autosub long range (ALR) with Biocam as payload, and we recorded a dataset on the 21st of September of 2019 we would write:

    /data/raw/2019/YK209/20190921_213000_alr_biocam

Once our software pipeline processes the dataset, it will automatically create the homologous configuration and processed folder as:

    /data/processed/2019/YK209/20190921_213000_alr_biocam
    /data/configuration/2019/YK209/20190921_213000_alr_biocam

Where, as explained, processed data and the configuration used to obtain it are stored.
At every dive folder, two files are essential for the software: vehicle.yaml and mission.yaml:
Mission.yaml: This file describes the mission's details and parameters of each sensor (e.g. where is the file path of the data, its timezone format, etc).
vehicle.yaml This file describes the location of the sensors relative to the defined position (origin) of the vehicle.
These two files are normally provided by the sensor or the platform. At the worst-case scenario, they can be tailored to fit any particular setup.

## Testing dataset

How to access the google bucket:
Install gsutil
Run the command
    gsutil -m cp -r gs://university-southampton-squidle /destination/folder


