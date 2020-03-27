import yaml

class Parameters:
    def __init__(self, file, type):
        with file.open('r') as f:
            params = yaml.safe_load(f)

        

        if type is 'mission':
            # camera parameters from mission.yaml
            self.format = params['image']['format']
            self.cameras = params['image']['cameras']

        if type is 'correct_config':
            # correction parameters from correct_config.yaml
            self.auv_nav_path = params['config']['auv_nav_path']
            self.image_type = params['config']['image_type']
            self.src_img_index = params['config']['src_img_index']
            self.altitude_max = params['attenuation_correction']['altitude']['max']
            self.altitude_min = params['attenuation_correction']['altitude']['min']
            self.sampling_method = params['attenuation_correction']['sampling_method']
            self.median_filter_kernel_size = params['attenuation_correction']['median_filter_kernel_size']
            self.target_mean = params['normalization']['target_mean']
            self.target_std = params['normalization']['target_std']
            self.debayer_option = params['output']['debayer_option']
            self.bayer_pattern = params['output']['bayer_pattern']
            self.dst_file_format = params['output']['dst_file_format']
            self.apply_attenuation_correction = params['flags']['apply_attenuation_correction']
            self.apply_gamma_correction = params['flags']['apply_gamma_correction']
            self.apply_distortion_correction = params['flags']['apply_distortion_correction']
            self.camera_parameter_file_path = params['flags']['camera_parameter_file_path']
            
