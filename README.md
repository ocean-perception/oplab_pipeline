# auv_nav
# Downloading and Updating the code #

To download the code, go to directory you want it to be in, open a terminal/command prompt there and type 
```
git clone --recursive https://github.com/ocean-perception/auv_nav.git
```

To push updates, stage changes, commit and push to a branch, usually master
```
git add -A
git commit -m "Some message about the change"
git push origin master
```

# Libraries Required #

*Requires [python3.6.2](https://www.python.org/downloads/release/python-362/) or later*

1. [matplotlib](https://matplotlib.org/2.0.2/users/installing.html) (version used: 2.2.2) `pip3 install matplotlib`
2. [numpy](https://pypi.python.org/pypi/numpy) (version used: 1.14.5) `pip3 install numpy`
3. [PyYAML](http://pyyaml.org/download/pyyaml/PyYAML-3.12.tar.gz) (version used: 3.12) `pip3 install pyyaml`
4. [plotly](https://pypi.python.org/pypi/plotly) (version used: 2.5.1) `pip3 install plotly`
5. [pandas](https://pypi.org/project/pandas/0.22.0/) (version used: 0.22.0) `pip3 install pandas`
6. [xlrd](https://pypi.org/project/xlrd/) (version used: 1.1.0) `pip3 install xlrd`
7. [prettytable](https://pypi.python.org/pypi/PrettyTable) (version used: 0.7.2) `pip3 install prettytable`

Some of the packages above are in [third_party](third_party) which can be installed by:
* executing the following terminal commands within the folder...
```
python3 setup.py install
python3 setup.py test
```
* ... or executing this command for whl files 
```
pip3 install pandas-0.22.0-cp36-cp36m-win_amd64.whl
```

# Functionality #

Parses and interleave navigation data for oplab standard and acfr standard formats. 

**Arguments**

Required arguments (only run one of them at a time):

| Parameter                 | Value           | Description   |
| :------------------------ |:--------------- | :-------------|
| -i --rawinputpath         | Path to root raw folder  | Parses the raw data
| -v --visualizeoutput      | Path to root processed folder where parsed data exist | Generate brief summaries of the output data
| -e --extractdata          | Path to root processed folder where parsed data exist | Extract useful information from output files

Optional arguments:

| Parameter                 | Value           |Default | Description   |
| :------------------------ |:--------------- |:------ | :-------------|
| -o --outputformat         | oplab/acfr | oplab | to select oplab or acfr format
| -start --startdatetime    | YYYYMMDDhhmmss | 20170817000000 (YYYYMMDD is automatically selected) | to select start date time of data to be processed
| -finish --finishdatetime  | YYYYMMDDhhmmss | 20170817235959 (YYYYMMDD is automatically selected) | to select finish date time of data to be processed
| -plot --plotpdfoption     | none | False | to select whether to output pdf plots
| -plotly --plothtmloption  | none | True | to select whether to output html interactive plots using plotly library
| -csv --csvoption          | none | False | to select whether to output csv files which contain navigation information for multiple sensors (can be further configured in [localisaion.yaml](localisation.yaml))
| -DR --deadreckoning       | none | False | to select whether to output dead reckoning csv outputs
| -PF --particlefilter      | none | False | to select whether to perform particle filter data fusion (can be further configured in [localisaion.yaml](localisation.yaml))

**Example commands to run**

For OPLAB output format:
1. Parse raw data into json file format 'nav_standard.json'
`python3 auv_nav.py -i '\\oplab-surf\reconstruction\raw\2017\SSK17-01\ts_un_006' -o oplab`

2. Visualise information in nav_standard.json output
`python3 auv_nav.py -v '\\oplab-surf\reconstruction\processed\2017\SSK17-01\ts_un_006' -o oplab`

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

3. Extract information from nav_standard.json output (start and finish time can be selected based on output in step 2)
`python3 auv_nav.py -e '\\oplab-surf\reconstruction\processed\2017\SSK17-01\ts_un_006' -o oplab -start 20170817032000 -finish 20170817071000 -plotly -csv -PF -DR`

For ACFR output format:
1. Parse raw data into combined.RAW.auv and mission.cfg
`python3 auv_nav.py -i '\\oplab-surf\reconstruction\raw\2017\SSK17-01\ts_un_006' -o acfr`

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

# Folder Structure #

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

# Example of Required YAML Configuration Files #

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

# Example Dataset for testing #
An example dataset can be downloaded from the following link with the expected folder structure: https://drive.google.com/drive/folders/0BzYMMCBxpT8BUF9feFpEclBzV0k?usp=sharing

Download, extract and specify the folder location and run as
```
python3 auv_nav.py -i ~/raw/2017/cruise/dive/ -o acfr
python3 auv_nav.py -i ~/raw/2017/cruise/dive/ -o oplab
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