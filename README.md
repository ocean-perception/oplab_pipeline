# auv_nav
parsers for navigation data for oplab standard and acfr standard formats

inputs are 

nav_parser.py <options>
  -i <path to mission.cfg>
  -o <output type> 'acfr' or 'oplab'

mission.cfg points to  input data structure with relative paths and defines origins, timeoffsets
eg. mission.cfg

origin {
	latitude = "26.674083";
	longitude = "127.868054";
	coordinate_reference_system = "wgs84";
}
velocity {
	format = "phins";
	filepath = "nav/phins/";
	filename = "20170816_phins.txt";
	timezone = "utc";
	timeoffset = "0.0";
}
orientation {
	format = "phins";
	filepath = "nav/phins/";
	filename = "20170816_phins.txt";
	timezone = "utc";
	timeoffset = "0.0";
}
depth {
	format = "phins";
	filepath = "nav/phins/";
	filename = "20170816_phins.txt";
	timezone = "utc";
	timeoffset = "0.0";
}
altitude {
	format = "phins";
	filepath = "nav/phins/";
	filename = "20170816_phins.txt";
	timezone = "utc";
	timeoffset = "0.0";
}
usbl {
	format = "gaps";
	filepath = "nav/gaps/";
	filename = "20170816091630-001.dat";
	timezone = "utc";
	timeoffset = "0.0";
}
images{
	format = "acfr_standard";
	filepath = "image/r20170816_023028_UG069_sesoko/i20170816_023028/";
	timezone = "utc";
	timeoffset = "0.0";
}

oplab output is in json formats
e.g. nav_standard.json
[{"epoch_timestamp": 1501974125.926, "epoch_timestamp_dvl": 1501974125.926, "class": "measurement", "sensor": "phins", "frame": "body", "category": "velocity", "data": [{"xx_velocity": -0.075, "xx_velocity_std": 0.00075}, {"yy_velocity": 0.024, "yy_velocity_std": 0.00024}, {"zz_velocity": -0.316, "zz_velocity_std": 0.00316}]},{"epoch_timestamp": 1501974003.738, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "orientation", "data": [{"heading": 82.706, "heading_std": 2.0}, {"roll": 4.849, "roll_std": 0.1}, {"pitch": -0.429, "pitch_std": 0.1}]},{"epoch_timestamp": 1501974002.7, "epoch_timestamp_depth": 1501974002.674, "class": "measurement", "sensor": "phins", "frame": "inertial", "category": "depth", "data": [{"depth": -0.958, "depth_std": -9.58e-05}]},{"epoch_timestamp": 1501974241.549, "epoch_timestamp_dvl": 1501974000.0, "class": "measurement", "sensor": "phins", "frame": "body", "category": "altitude", "data": [{"altitude": 0.0, "altitude_std": 0.0}, {"sound_velocity": 0.0, "sound_velocity_correction": 0.0}]}]
