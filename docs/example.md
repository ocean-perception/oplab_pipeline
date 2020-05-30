# Examples
## YAML Configuration Files ##

These files need to be in the root raw folder. Further examples can be found in default_yaml folder

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

## Dataset for testing ##
An example dataset can be downloaded from the following link with the expected folder structure: https://drive.google.com/drive/folders/0BzYMMCBxpT8BUF9feFpEclBzV0k?usp=sharing

Download, extract and specify the folder location and run as
```
OPLAB format:
auv_nav parse ~/raw/2017/cruise/dive
auv_nav process -s 20170816032345 -e 20170816034030  ~/processed/2017/cruise/dive
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
