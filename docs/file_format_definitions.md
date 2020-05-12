# File format definitions #

## mission.yaml file format definition ##

### Version 0 ###
- One node each for the "standard nodes" `velocity`, `orientation,` `depth`, `altitude` and `usbl`, as well as one node `origin` and one node `image`
- The standard nodes have the fields `format`, `filepath`, `filename`, `timezone` (optional, default: utc), `timeoffset` (optional, default: 0) and `headingoffset` (optional, default: 0).
- The node `origin` has the fields `latitude`, `longitude`, `coordinate_reference_system` and `date`. The only supported coordinate reference system is wgs84. The date has to be indicated in the format YYYY/MM/DD.
- The node `image` contains the fields `format`, `filepath`, `camera1`, `camera2`, `camera3` (optional), `timezone` and `timeoffset`. 
- No field indicating the file format version.

Partial example:
```yaml
#YAML 1.0
origin:
  latitude: 26.674083
  longitude: 127.868054
  coordinate_reference_system: wgs84
  date: 2017/08/17

velocity:
  format: phins
  filepath: nav/phins/
  filename: 20170817_phins.txt
  timezone: utc
  timeoffset: 0.0
  headingoffset: -45.0

usbl:
  format: gaps
  filepath: nav/gaps/
  timezone: utc
  timeoffset: 0.0

image:
  format: acfr_standard
  filepath: image/r20170817_022018_UG115_sesoko/i20170817_022018/
  camera1: LC
  camera2: RC
  timezone: utc
  timeoffset: 0.0
```

### Version 1 ###
Changes since verison 0:
- Node `version` indicating the file format version (1).
- `headingoffset` in the standard nodes is no longer supported. This is now indicated in vehicle.yaml
- The standard nodes have 3 additional fields: `origin` (optional, default determined depending on type of node), `std_factor` (optional, default: 0) and `std_offset` (optional, default: 0). `origin` has to be the name of a node in the vehicle.yaml file. The coordinates in that node in the vehicle indicate where the device is located on the vehicle
- Additional standard node `tide`
- The ndoe `usbl` has an optional field `id` (or `label`)
- The node `image` now has the fields `format`, `timezone`, `timeoffset` as well as a node `cameras` containing the list of cameras (maximum 3 cameras currently supported by auv_nav). Each item in the list of `cameras` has the fields `name`, `origin` (optional, default: same as the `name`), `type`, `bit_depth` (currently unused) and `path`. `origin` (resp. if `origin` is not indicated then `name`) indicates where the camera is located on the vehicle, and the name stated must exist as a node in the vehicle.yaml file.

Partial example:
```yaml
version: 1 # Node version has been added

origin:
  latitude: 22.8778145
  longitude: 153.38397866666668
  coordinate_reference_system: wgs84
  date: 2018/11/19

velocity:
  format: autosub
  filepath: nav/
  filename: M155.mat
  timezone: utc
  timeoffset: 0.0
  std_factor: 0.001 # Field std_factor has been added
  std_offset: 0.2   # Field std_offset has been added
  # Field headingoffset has been removed

usbl:
  format: gaps
  filepath: nav/gaps/
  timezone: 0
  timeoffset: .0
  id: 4             # Field id has been added
  std_factor: 0.01
  std_offset: 2.0
  

image:
  format: seaxerocks_3
  cameras: # Cameras are defined now as a list
    - name: fore
      type: bayer_rggb
      bit_depth: 12
      path: image/SeaXerocksData20181119_073812_laserCal/Xviii/Cam51707923
    - name: aft
      type: bayer_rggb
      bit_depth: 12
      path: image/SeaXerocksData20181119_073812_laserCal/Xviii/Cam51707925
    - name: laser
      type: grayscale
      bit_depth: 12
      path: image/SeaXerocksData20181119_073812_laserCal/LM165
  timezone: 10
  timeoffset: 0.0
```

## vehicle.yaml file format definition ##
Currently only file format version 2 is supported. For examples check the subfolders of oplab/default_yaml/.

### Version 0 ###
- One node each for the "standard nodes" `origin`, `usbl`, `ins`, `dvl`, `depth`, `chemical` (optional).
- One node for each of the cameras, named `camera1`, `camera2`, `camera3` (optional).
- Each of the standard and camera nodes has a fields `x_offset`, `y_offset` and `z_offset`, which references each item described by a node with respect to the origin of the vehicle. The x-axis is aligned with the forward direction of the vehicle, the y-axis points to the right-hand side and the z-axis points downwards with respect to the vehicle.
- Nodes do not contain information about orientation.
- No field indicating the file format version.

Partial example:
```yaml
origin:
  x_offset: 0
  y_offset: 0
  z_offset: 0

camera1: # Cameras named camera1, camera2, camera3
  x_offset: -0.05
  y_offset: -0.3
  z_offset: 0.18
```

### Version 1 ###
Changes since version 0:  
- x, y and z-offsets have been renamed `surge_m`, `sway_m` and `heave_m`.
- Nodes contain Euler angles `roll_deg`, `pitch_deg` and `yaw_deg` for determining the orienation of each item. The rotations are carried out in the following order: Roll about x-axis, pitch about y-axis, yaw abour z-axis.

Partial example:
```yaml
origin:
  surge_m: 0 # fomer x_offset
  sway_m:  0 # fomer y_offset
  heave_m: 0 # fomer z_offset
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

camera1: # Cameras named camera1, camera2, camera3
  surge_m: 0.262875
  sway_m: 0.
  heave_m: 0.5
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0
```

### Version 2 ###
Changes since version 1:  
- Camera nodes are named after the `name` given in mission.yaml instead of camera1, camera2, camera3. The maximum number of cameras currently supported is 3. 

Partial example:
```yaml
origin:
  surge_m: 0
  sway_m: 0
  heave_m: 0
  roll_deg: 0
  pitch_deg: 0
  yaw_deg: 0

cam61003146: # Cameras name corresponding to camera label in mission.yaml
  surge_m: 1.484
  sway_m: 0
  heave_m: 0.327
  roll_deg: 0
  pitch_deg: -90
  yaw_deg: 0
```

## Compatibility with auv_nav ##
Currently both file format versions of mission.yaml and all 3 versions of vehicle.yaml are supported. However, when using vehicle.yaml file formats 0 or 1, the cameras in mission.yaml have to have their `name`s fields set to "camera1", "camera2" and (if present) "camera3".
Also, in file format verison 0 of both files the heading offset of sensors (e.g. DVL) is set in the mission.yaml file (`headingoffset`), whereas from format version 1 of both files it is set in vehicle.yaml (`yaw_deg`). It is therefore important not to mix these file format versions.