import yaml

class Parameters:
    def __init__(self, file):
        with file.open('r') as f:
            params = yaml.safe_load(f)

        # correction parameters from correct_config.yaml
        self.camera_system = params['camera_system']
        self.json_path = params['json_path']
        self.camera_path_list = params['image_path']
        self.altitude_max = params['image_filter']['altitude_range']['max']
        self.altitude_min = params['image_filter']['altitude_range']['min']
        self.sampling_method = params['attenuation_model']['sampling_method']
        self.window_edge_length = params['attenuation_model']['window_edge_length']
        self.distance_metric = params['attenuation_model']['distance_metric']
        self.target_mean = params['output_settings']['brightness']
        self.target_std = params['output_settings']['contrast']
        self.apply_distortion_correction = params['output_settings']['apply_distortion_correction']

