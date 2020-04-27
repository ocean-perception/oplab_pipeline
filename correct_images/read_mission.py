import yaml


class Image:
    def __init__(self, cdict):
        self.format = cdict['format']
        self.timezone = cdict['timezone']
        self.timeoffset = cdict['timeoffset']
        self.cameras = cdict['cameras']
        self.cameras_number = self.cameras

        if self.format == 'seaxerocks_3':
            self.cameras_0 = cdict['cameras'][0]
            self.cameras_1 = cdict['cameras'][1]
            self.cameras_2 = cdict['cameras'][2]
        elif self.format == 'biocam':
            self.cameras_0 = cdict['cameras'][0]
            self.cameras_1 = cdict['cameras'][1]

        else:
            self.cameras_0 = cdict['cameras'][0]
            self.cameras_1 = cdict['cameras'][1]


class config:
    def __init__(self, cdict):
        self.auv_nav_path = cdict['auv_nav_path']
        self.src_img_index = cdict['src_img_index']
        self.format = cdict['format']
        if self.format == 'seaxerocks_3':
            self.camera_1 = cdict['camera1']
            self.camera_2 = cdict['camera2']
            #self.camera_3 = cdict['camera3']
        else:
            self.camera_1 = cdict['camera1']
            self.camera_2 = cdict['camera2']
        #self.camera = cdict['camera1']


class attenuation_correction:
    def __init__(self, cdict):
        self.altitude_max = cdict['altitude']['max']
        self.altitude_min = cdict['altitude']['min']
        self.sampling_method = cdict['sampling_method']
        self.median_filter_kernel_size = cdict['median_filter_kernel_size']


class normalization:
    def __init__(self, cdict):
        self.target_mean = cdict['target_mean']
        self.target_std = cdict['target_std']
            


class output:
    def __init__(self, cdict):
        self.debayer_option = cdict['debayer_option']
        self.bayer_pattern = cdict['bayer_pattern']
        self.dst_file_format = cdict['dst_file_format']


class flags:
    def __init__(self, cdict):
        self.apply_attenuation_correction = cdict['apply_attenuation_correction']
        self.apply_gamma_correction = cdict['apply_gamma_correction']
        self.apply_distortion_correction = cdict['apply_distortion_correction']
        self.camera_parameter_file_path = cdict['camera_parameter_file_path']


class Parameters:
    def __init__(self, file, type):
        with file.open('r') as f:
            params = yaml.safe_load(f)

        print('filepath:',file)

        if type is 'mission':
            # image parameters
            cdict = params['image']
            self.image = Image(cdict)

        if type is 'correct':
            # image parameters
            cdict = params['config']
            self.config = config(cdict)
            cdict = params['attenuation_correction']
            self.attenuation_correction = attenuation_correction(cdict)
            cdict = params['normalization']
            self.normalization = normalization(cdict)
            cdict = params['output']
            self.output = output(cdict)
            cdict = params['flags']
            self.flags = flags(cdict)
            

def read_params(path, type):
	pm = Parameters(path, type)
	return pm