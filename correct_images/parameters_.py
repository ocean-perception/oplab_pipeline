import yaml

class Parameters:
    def __init__(self, file, type):
        with file.open('r') as f:
            params = yaml.safe_load(f)

        # parameters from mission.yaml
        if type == 'mission':
            self.image_system = params['image']['image_system']
            self.cameras_mission = params['image']['cameras']

        # correction parameters from correct_config.yaml
        if type == 'config':
            self.cameras_config = params['cameras']
            self.undistort = params['output_settings']['undistort']
            self.output_image_format = params['output_settings']['compression_parameter']
        