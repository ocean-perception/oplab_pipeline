# auv_nav
## Downloading and Updating the code ##

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

## Libraries Required ##

*Requires [python3.6.2](https://www.python.org/downloads/release/python-362/) or later*

1. [matplotlib](https://matplotlib.org/2.0.2/users/installing.html) (version used: 2.2.2) `pip3 install matplotlib`
2. [numpy](https://pypi.python.org/pypi/numpy) (version used: 1.14.5) `pip3 install numpy`
3. [PyYAML](http://pyyaml.org/download/pyyaml/PyYAML-3.12.tar.gz) (version used: 3.12) `pip3 install pyyaml`
4. [plotly](https://pypi.python.org/pypi/plotly) (version used: 2.5.1) `pip3 install plotly`
5. [pandas](https://pypi.org/project/pandas/0.22.0/) (version used: 0.22.0) `pip3 install pandas`
6. [xlrd](https://pypi.org/project/xlrd/) (version used: 1.1.0) `pip3 install xlrd`
7. [prettytable](https://pypi.python.org/pypi/PrettyTable) (version used: 0.7.2) `pip3 install prettytable`

Depending on the Python distribution you are using, you might already have these packages, or you might have to install them manually. If you are using Anaconda (https://www.anaconda.com/download/) will proably only have to install prettytable and plotly, which you can install from the Conda prompt: 
```
pip install prettytable
conda install plotly
```
(In general, when using Conda, it is preferable to use `conda` for installing packages rather than `pip`, but if `conda` doesn't work, `pip` can be used.)  
If you are using a different distribution, you might have to to install all packages manually using one of the following commands (depending on your distribution it might be  `pip` instead of `pip3` and `python` instead of `python3`):
```
pip3 install matplotlib numpy pyyaml plotly pandas xlrd prettytable --user
```
Some of the packages above are in [third_party](third_party) which can be installed by:
* executing the following terminal commands within the folder...
```
python3 setup.py install
python3 setup.py test
```
* ... or executing this command for whl files (if you are using Windows)
```
pip3 install pandas-0.22.0-cp36-cp36m-win_amd64.whl
```

## Usage ##

auv_nav has 2 commands:  
- `parse`, which reads raw data from multiple sensors and writes it in chornological order to a text file in an intermediate data format. There are 2 different intermediate formats to choose: "oplab" or "acfr"
- `process`, which outputs previously parsed data in a format that the 3D mapping pipeline can read

`auv_nave parse' usage:
```
auv_nav.py parse [-h] [-f FORMAT] path

positional arguments:
  path                  Folderpath where the (raw) input data is. Needs to be
                        a subfolder of 'raw' and contain the mission.yaml
                        configuration file.

optional arguments:
  -h, --help            show this help message and exit
  -f FORMAT, --format FORMAT
                        Format in which the data is output. 'oplab' or 'acfr'.
                        Default: 'oplab'.
```
The algorithm replicated the same folder structure as the input data, but instead of using a subfolder of 'raw', a folder 'processed' is created (if doesn't already exist), with the same succession of subfolders as there are in 'raw'. The mission.yaml and the vehicle.yaml files are copied to this folder. When using the 'oplab' format, the interlaced data is written in oplab format to a file called 'nav_standard.json' in a subfolder called 'nav'. When using the 'acfr' format, the interlaced data is written in acfr format to a file called combined.RAW.auv in a folder called dRAWLOGS_cv.


`auv_nav.py process` usage:
```
auv_nav.py process [-h] [-f FORMAT] [-s START_DATETIME] [-e END_DATETIME] path

positional arguments:
  path                  Path to folder where the data to process is. The
                        folder has to be generated using auv_nav parse.

optional arguments:
  -h, --help            show this help message and exit
  -f FORMAT, --format FORMAT
                        Format in which the data to be processed is stored.
                        'oplab' or 'acfr'. Default: 'oplab'.
  -s START_DATETIME, --start START_DATETIME
                        Start date & time in YYYYMMDDhhmmss from which data
                        will be processed. If not set, start at beginning of
                        dataset.
  -e END_DATETIME, --end END_DATETIME
                        End date & time in YYYYMMDDhhmmss up to which data
                        will be processed. If not set process to end of
                        dataset.
```


## Examples ##

For OPLAB output format:
1. Parse raw data into json file format 'nav_standard.json' and visualise data output

    `python3 auv_nav.py parse -f oplab '\\oplab-surf\data\reconstruction\raw\2017\SSK17-01\ts_un_006'`

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

    `python3 auv_nav.py process -f oplab -s 20170817032000 -e 20170817071000 '\\oplab-surf\data\reconstruction\processed\2017\SSK17-01\ts_un_006'`

For ACFR output format:
1. Parse raw data into combined.RAW.auv and mission.cfg

    `python3 auv_nav.py parse -f acfr '\\oplab-surf\data\reconstruction\raw\2017\SSK17-01\ts_un_006'`

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

These files need to be in the root raw folder. Further examples can be found in [sample_yaml](sample_yaml)

mission.yaml:
```
#YAML 1.0
origin:
    latitude: 26.674083
    longitude: 127.868054               
    coordinate_reference_system: wgs84  
    date: 2017/08/17              

velocity:
    format: phins
    thread: dvl
    filepath: nav/phins/
    filename: 20170817_phins.txt
    timezone: utc
    timeoffset: 0.0
    headingoffset: -45.0

orientation:
    format: phins
    filepath: nav/phins/
    filename: 20170817_phins.txt
    timezone: utc
    timeoffset: 0.0
    headingoffset: -45.0

depth:
    format: phins
    filepath: nav/phins/
    filename: 20170817_phins.txt
    timezone: utc
    timeoffset: 0.0

altitude:
    format: phins
    filepath: nav/phins/
    filename: 20170817_phins.txt
    timezone: utc
    timeoffset: 0.0

usbl:
    format: gaps
    filepath: nav/gaps/
    timezone: utc
    timeoffset: 0.0
    id: 1

image:
    format: acfr_standard
    filepath: image/r20170817_041459_UG117_sesoko/i20170817_041459/
    camera1: LC
    camera2: RC
    timezone: utc
    timeoffset: 0.0
```

vehicle.yaml
```
#YAML 1.0
origin: #centre of robot
  x_offset: 0
  y_offset: 0
  z_offset: 0

# distance with reference to origin/centre of robot
usbl:
  x_offset: 0.1
  y_offset: 0
  z_offset: -0.5

ins:
  x_offset: 0.1
  y_offset: 0
  z_offset: 0

dvl:
  x_offset: -0.45
  y_offset: 0
  z_offset: 0.45

depth:
  x_offset: 0.16
  y_offset: 0
  z_offset: 0

camera1:
  x_offset: -0.05
  y_offset: -0.3
  z_offset: 0.18

camera2:
  x_offset: -0.05
  y_offset: -0.1
  z_offset: 0.18
```

## Example Dataset for testing ##
An example dataset can be downloaded from the following link with the expected folder structure: https://drive.google.com/drive/folders/0BzYMMCBxpT8BUF9feFpEclBzV0k?usp=sharing

Download, extract and specify the folder location and run as
```
ACFR format:
python3 auv_nav.py parse ~/raw/2017/cruise/dive -f acfr

OPLAB format:
python3 auv_nav.py parse ~/raw/2017/cruise/dive -f oplab
python3 auv_nav.py process ~/processed/2017/cruise/dive -f oplab -s 20170816032345 -e 20170816034030
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
