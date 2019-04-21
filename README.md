# auv_nav

## Downloading and updating the code ##
To download the code, go to directory you want download the code to, open a terminal/command prompt there and type
```
git clone https://github.com/ocean-perception/auv_nav.git
```

To push updates you made to the repository on github (assuming you are using the master branch, which is the default), type
```
git add -u
git commit -m "Some message about the change"
git push origin master
```


## Running the code ##
Requires [python3.6.2](https://www.python.org/downloads/release/python-362/) or later ([latest](https://www.python.org/downloads/release/python3)).  
You can also use [Anaconda](https://www.anaconda.com/download/), which comes with the Spyder IDE, that in turn uses the IPython shell.

### When using Python from standard terminal ###
The auv_nav project is compiled into a binary. Navigate to the auv_nav folder, where setup.py is saved and execute
```
python3 setup.py install
python3 setup.py test
```
`auv_nav` can now be used as a command from any folder on your computer. You can test that by trying to display the help: 
```
auv_nav -h
```

### When using Anaconda (with the Spyder IDE) ###
Start the Anaconda prompt and run the following lines: 
```
pip install prettytable
conda install plotly
```
Open Spyder and use its shell (it is the IPython shell) to navigate to the auv_nav folder (the one which contains README.md, etc.) and execute
```
import auv_nav
run auv_nav/auv_nav.py -h
```
This should display the help. To use auv_nav, replace `-h` with the command you want to run.  
This way you can run auv_nav from within Spyder, but you always need to navigate to this directory to execute the code

Alternatively, you can us the Anaconda prompt to make binaries by navigating to the auv_nav folder and executing the following lines:
```
python setup.py install
python setup.py test
```
After doing so auv_nav can be called from any location (no need to navigate to the auv_nav directory) using the Anaconda prompt. However, this will only work from the Anaconda prompt and not from IPython, the reason being that IPython not supporting execution of binaries.


### Required libraries ###
Below is a list of the required libraries. You should not need to install them manually, as the instructions above should have already installed these. The list and the commands are just listed here for completeness.
1. [matplotlib](https://matplotlib.org/2.0.2/users/installing.html) (version used: 2.2.2) `pip3 install matplotlib`
2. [numpy](https://pypi.python.org/pypi/numpy) (version used: 1.14.5) `pip3 install numpy`
3. [PyYAML](http://pyyaml.org/download/pyyaml/PyYAML-3.12.tar.gz) (version used: 3.12) `pip3 install pyyaml`
4. [plotly](https://pypi.python.org/pypi/plotly) (version used: 2.5.1) `pip3 install plotly`
5. [pandas](https://pypi.org/project/pandas/0.22.0/) (version used: 0.22.0) `pip3 install pandas`
6. [xlrd](https://pypi.org/project/xlrd/) (version used: 1.1.0) `pip3 install xlrd`
7. [prettytable](https://pypi.python.org/pypi/PrettyTable) (version used: 0.7.2) `pip3 install prettytable`
(The commands for installing the libaries are for standard python 3. If you are using Anaconda, `pip3` is replaced either by `conda` or by `pip`.)

If for some reason you want to install the above libraries manually, the command is (on Linux)
```
pip3 install matplotlib numpy pyyaml plotly pandas xlrd prettytable --user
```
If you are using a different distribution, you might have to to install all packages manually using one of the following commands (depending on your distribution it might be  `pip` instead of `pip3` and `python` instead of `python3`)


## Usage ##

auv_nav has 3commands:
- `parse`, which reads raw data from multiple sensors and writes it in chornological order to a text file in an intermediate oplab data format.
- `process`, which outputs previously parsed data in a format that the 3D mapping pipeline can read
- `convert`, which converts previously parsed data to ACFR format at this moment.

`auv_nav parse` usage:
```
auv_nav parse [-h] [-F] path

positional arguments:
  path                  Folderpath where the (raw) input data is. Needs to be
                        a subfolder of 'raw' and contain the mission.yaml
                        configuration file.

optional arguments:
  -h, --help            show this help message and exit
  -F, --Force           Force file overwrite
```
The algorithm replicated the same folder structure as the input data, but instead of using a subfolder of 'raw', a folder 'processed' is created (if doesn't already exist), with the same succession of subfolders as there are in 'raw'. The mission.yaml and the vehicle.yaml files are copied to this folder. The interlaced data is written in oplab format to a file called 'nav_standard.json' in a subfolder called 'nav'.


`auv_nav process` usage:
```
auv_nav process [-h] [-F] [-s START_DATETIME] [-e END_DATETIME] path

positional arguments:
  path                  Path to folder where the data to process is. The
                        folder has to be generated using auv_nav parse.

optional arguments:
  -h, --help            show this help message and exit
  -F, --Force           Force file overwrite
  -s START_DATETIME, --start START_DATETIME
                        Start date & time in YYYYMMDDhhmmss from which data
                        will be processed. If not set, start at beginning of
                        dataset.
  -e END_DATETIME, --end END_DATETIME
                        End date & time in YYYYMMDDhhmmss up to which data
                        will be processed. If not set process to end of
                        dataset.
```

`auv_nav convert` usage:
```
auv_nav convert [-h] [-f FORMAT] [-s START_DATETIME] [-e END_DATETIME] path

positional arguments:
  path                  Path to folder where the data to process is. The
                        folder has to be generated using auv_nav parse.

optional arguments:
  -h, --help            show this help message and exit
  -f FORMAT, --format FORMAT
                        Format in which the data is output. Default: 'acfr'.
  -s START_DATETIME, --start START_DATETIME
                        Start date & time in YYYYMMDDhhmmss from which data
                        will be processed. If not set, start at beginning of
                        dataset.
  -e END_DATETIME, --end END_DATETIME
                        End date & time in YYYYMMDDhhmmss up to which data
                        will be processed. If not set process to end of
                        dataset.
```
The algorithm will read in the nav_standard.json file obtained after the parsing and will write the required formats and outputs. At v0.0.1.6 the following output formats are available:
* acfr: The AFCR format uses a 'dRAWLOGS_cv' folder name and outputs its navigation solution to a file called 'combined.RAW.auv' as well as to a 'mission.cfg' to the processing folder root.

## Examples ##
An example data set can be downloaded from here:

    `https://console.cloud.google.com/storage/browser/university-southampton-   squidle/raw/year/cruise/platform/YYYYMMDD_hhmmss_platform_sensor/`

Make sure you copy the folder format from 'raw/year/cruise/platform/YYYYMMDD_hhmmss_platform_sensor/'

For OPLAB output format:
1. Parse raw data into json file format 'nav_standard.json' and visualise data output by parsing the target directory where mission.yaml is stored.

    `auv_nav parse -F '<container directory>/raw/year/cruise/platform/YYYYMMDD_hhmmss_platform_sensor/'`

    Example of output:
    ```
    'oplab' - nav_standard.json
    [{"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "velocity", "data": [{"x_velocity": -0.075, "x_velocity_std": 0.200075}, {"y_velocity": 0.024, "y_velocity_std": 0.200024}, {"z_velocity": -0.316, "z_velocity_std": 0.20031600000000002}]},
    {"epoch_timestamp": 1501974002.1, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "orientation", "data": [{"heading": 243.777, "heading_std": 2.0}, {"roll": 4.595, "roll_std": 0.1}, {"pitch": 0.165, "pitch_std": 0.1}]},
    {"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.875, "class": "measurement", "sensor": "phins", "frame": "body", "category": "altitude", "data": [{"altitude": 31.53, "altitude_std": 0.3153}, {"sound_velocity": 1546.0, "sound_velocity_correction": 0.0}]},
    {"epoch_timestamp": 1501974002.7, "epoch_timestamp_depth": 1501974002.674, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "depth", "data": [{"depth": -0.958, "depth_std": -9.58e-05}]},
    {"epoch_timestamp": 1502840568.204, "class": "measurement", "sensor": "gaps", "frame": "inertial", "category": "usbl", "data_ship": [{"latitude": 26.66935735000014, "longitude": 127.86623359499968}, {"northings": -526.0556603025898, "eastings": -181.08730736724087}, {"heading": 174.0588800058365}], "data_target": [{"latitude": 26.669344833333334, "latitude_std": -1.7801748803947248e-06}, {"longitude": 127.86607166666667, "longitude_std": -1.992112444781924e-06}, {"northings": -527.4487693247576, "northings_std": 0.19816816183128352}, {"eastings": -197.19537408743128, "eastings_std": 0.19816816183128352}, {"depth": 28.8}]},{"epoch_timestamp": 1501983409.56, "class": "measurement", "sensor": "unagi", "frame": "body", "category": "image", "camera1": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_LC16.tif"}], "camera2": [{"epoch_timestamp": 1501983409.56, "filename": "PR_20170816_023649_560_RC16.tif"}]}
    ]
    ```

2. Extract information from nav_standard.json output (start and finish time can be selected based on output in step 2)

    `auv_nav process -f oplab -s 20170817032000 -e 20170817071000 '\\oplab-surf\data\reconstruction\processed\2017\SSK17-01\ts_un_006'`

For ACFR output format:
1. Parse raw data into combined.RAW.auv and mission.cfg

    `auv_nav parse -f acfr '\\oplab-surf\data\reconstruction\raw\2017\SSK17-01\ts_un_006'`

    Example of output:
    ```
    'acfr' - combined.RAW.auv
        PHINS_COMPASS: 1444452882.644 r: -2.29 p: 17.21 h: 1.75 std_r: 0 std_p: 0 std_h: 0
        RDI: 1444452882.644 alt:200 r1:0 r2:0 r3:0 r4:0 h:1.75 p:17.21 r:-2.29 vx:0.403 vy:0 vz:0 nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:32768 h_true:0 p_gimbal:0 sv: 1500
        PAROSCI: 1444452882.644 298.289
        VIS: 1444452882.655 [1444452882.655] sx_073311_image0003805_AC.tif exp: 0
        VIS: 1444452882.655 [1444452882.655] sx_073311_image0003805_FC.tif exp: 0
        SSBL_FIX: 1444452883 ship_x: 402.988947 ship_y: 140.275056 target_x: 275.337171 target_y: 304.388346 target_z: 299.2 target_hr: 0 target_sr: 364.347071 target_bearing: 127.876747
    ```

## Folder Structure ##

The output files are stored in a mirrored file location where the input raw data is stored as follows with the paths to raw data as defined in mission.yaml
```
e.g.
    raw /<YEAR> /<CRUISE> /<DIVE>   /mission.yaml
                                    /vehicle.yaml
                                    /nav/gaps/
                                    /nav/phins/
                                    /image/r20170816_023028_UG069_sesoko/i20170816_023028/
```

For this example, the outputs would be stored in the follow location, where folders will be automatically generated
```
e.g.
    processed /<YEAR> /<CRUISE> /<DIVE> /nav/nav_standard.json
                                        /dRAWLOGS_cv/combined.RAW.auv
                                        /mission.cfg
```

## Example of Required YAML Configuration Files ##

These files need to be in the root raw folder. Further examples can be found in [default_yaml](default_yaml)

1. mission.yaml:
This file describes the mission's details and parameters of each sensor (e.g. where is the filepath of the data, its timezone format, etc).
```
#YAML 1.0
version: 1  # New vehicle.yaml format

origin:
  latitude: 22.745000
  longitude: 153.266667
  coordinate_reference_system: wgs84

velocity:
  format: ae2000  # phins or ae2000
  filepath: nav/ae_log/  # where is the filepath located
  filename: pos171123064542.csv  # what is the filename
  timezone: jst  # what time zone is the timestamps in
  timeoffset: 0.0  # what is the time offset
  std_factor: 0.01 # what is the std factor
  std_offset: 0    # what is the std offset

orientation:
  format: ae2000  # phins or ae2000
  filepath: nav/ae_log/
  filename: pos171123064542.csv
  timezone: jst
  timeoffset: 0.0
  std_factor: 0.01
  std_offset: 0

depth:
  format: ae2000  # phins or ae2000
  filepath: nav/ae_log/
  filename: pos171123064542.csv
  timezone: jst
  timeoffset: 0.0
  std_factor: 0.01
  std_offset: 0

altitude:
  format: ae2000  # phins or ae2000
  filepath: nav/ae_log/
  filename: pos171123064542.csv
  timezone: jst
  timeoffset: 0.0
  std_factor: 0.01
  std_offset: 0

usbl:
  format: usbl_dump  # gaps or usbl_dump
  filepath: nav/ssbl/
  filename: 20171123_AE2000f_LOG.csv  # filename of data, only for usbl_dump format
  timezone: 9
  timeoffset: 0.0
  label: T1  # selected usbl label
  std_factor: 0.01
  std_offset: 0

image:
  format: seaxerocks_3  # acfr_standard or seaxerocks_3
  cameras:
    - name: fore
      type: bayer_rggb
      path: image/SeaXeroxData20171123_095119/Xviii/Cam51707923
    - name: aft
      type: bayer_rggb
      path: image/SeaXeroxData20171123_095119/Xviii/Cam51707925
    - name: laser
      type: grayscale
      path: image/SeaXeroxData20171123_095119/LM165
  timezone: jst
  timeoffset: 0.0
```

2. vehicle.yaml
This file describes the location of the sensors relative to the defined position (origin) of the vehicle.
```
#YAML 1.0
origin: #centre of robot
  surge_m: 0
  sway_m: 0
  heave_m: 0
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

# distance with reference to origin/centre of robot
usbl:
  surge_m: 0 
  sway_m: 0 
  heave_m: -0.289 
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

ins:
  surge_m: -0.09 
  sway_m: 0 
  heave_m: 0 
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

dvl:
  surge_m: -0.780625
  sway_m: 0
  heave_m: 0.204
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

depth:
  surge_m: 0 
  sway_m: 0 
  heave_m: 0 
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

camera1: # Front camera
  surge_m: 0.262875
  sway_m: 0
  heave_m: 0.5 
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

camera2: #Back Camera
  surge_m: 0.012875
  sway_m: 0
  heave_m: 0.5 
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

camera3: # Laser
  surge_m: 0.147875
  sway_m: 0
  heave_m: 0.5 
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0
```

## Example Dataset for testing ##
An example dataset can be downloaded from the following link with the expected folder structure: https://drive.google.com/drive/folders/0BzYMMCBxpT8BUF9feFpEclBzV0k?usp=sharing

Download, extract and specify the folder location and run as
```
OPLAB format:
auv_nav parse -f oplab ~/raw/2017/cruise/dive
auv_nav process -f oplab -s 20170816032345 -e 20170816034030  ~/processed/2017/cruise/dive
Convert to ACFR
auv_nav convert -f acfr ~/raw/2017/cruise/dive
```

The coordinate frames used are those defined in Thor Fossen Guidance, Navigation and Control of Ocean Vehicles

i.e. Body frame:
        x-direction: +ve aft to fore
        y-direction: +ve port to starboard
        z-direction: +ve top to bottom
i.e. Intertial frame:
        north-direction: +ve north
        east-direction: +ve east
        down-direction: +ve depth downwards

Parameter naming conventions
    long and descriptive names should be used with all lower case letters.
