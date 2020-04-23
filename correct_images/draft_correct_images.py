# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, University of Southampton
All rights reserved.
"""

from pathlib import Path

from auv_nav.mission import Mission
from auv_nav.console import Console
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder

from correct_images.camera_system import CameraSystem
from correct_images.corrections import DebayerCorrection
from correct_images.corrections import PixelStatsCorrection
from correct_images.corrections import AttenuationCorrection

class CorrectImages:
    def __init__(self, path, force):
        path = get_raw_folder(Path(path).resolve())

        # Read mission.yaml
        path_mission = path / "mission.yaml"
        path_correct = get_config_folder(path) / "correct_images.yaml"
        Console.info('Loading mission.yaml at: ', path_mission)
        self.mission = Mission(path_mission)

        # Read correct_images.yaml. If not there, copy default one
        Console.info('Loading correct_images.yaml at: ', path_correct)
        if not path_correct.exists():
            root = Path(__file__).parents[1]
            default_file = root / 'correct_images/default_yaml' / 'correct_images.yaml'
            Console.warn("Cannot find {}, generating default from {}".format(path_correct, default_file))
            default_file.copy(path_correct)
            Console.warn('File was not found. Copying default file instead.')
            
            # TODO Update default correct_images.yaml parameters with values specific to camera systems in mission.yaml 
        
        # TODO implement Configuration class to read in correct_images configuration (a.k.a. parser)
        self.config = Configuration(path_correct)

        # Instantiate camera system
        # TODO implement inner functions
        self.camera_system = CameraSystem(path, self.mission)

        # TODO Possible routes, we can use argparse for some
        # 1) Just debayer
        # 2) Pixelstats (do we want / need to keep it?)
        # 3) Attenuation correction
        # 3.1) With DVL altitude
        # 3.2) With laser bathymetry depth

        # TODO fix the output folder structure
        # self.output_dirs = None ...

    def distortion_correct(self):
        pass
    
    def apply_corrector(self):
        pass
    
    def debayer(self):
        d = DebayerCorrection(self.camera_system, self.config)
        return d
        
    def write_output(self):
        # self.debayer_image
        # self.output_dirs
        pass

    def parse(self):
        if self.config.pixel_stats:
            self.corrector = PixelStatsCorrection(
                self.camera_system, self.config)
        elif self.config.attenuation_correction:
            self.corrector = AttenuationCorrection(
                self.camera_system, self.config)
    
    def process(self):
        # TODO add code to check if calibration paremeters are available     
        if (calibration != None):
            self.distortion_correct()
        self.apply_corrector()
        self.debayer_image = self.debayer()
        self.write_output(self.debayer_image, self.output_dirs)
