import yaml
from auv_nav.tools.folder_structure import get_config_folder


class SensorOffset:
    def __init__(self):
        self.surge = 0.0
        self.sway = 0.0
        self.heave = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._empty = True

    def empty(self):
        return self._emtpy

    def load(self, node):
        self._empty = False
        self.surge = node['surge_m']
        self.sway = node['sway_m']
        self.heave = node['heave_m']
        self.roll = node['roll_deg']
        self.pitch = node['pitch_deg']
        self.yaw = node['yaw_deg']


class Vehicle:
    def __init__(self, filename):
        self.origin = SensorOffset()
        self.ins = SensorOffset()
        self.dvl = SensorOffset()
        self.depth = SensorOffset()
        self.usbl = SensorOffset()
        self.camera1 = SensorOffset()
        self.camera2 = SensorOffset()
        self.camera3 = SensorOffset()
        self.camera3 = SensorOffset()
        self.chemical = SensorOffset()

        vehicle_file = get_config_folder(filename)
        with vehicle_file.open('r') as stream:
            data = yaml.load(stream)
            if 'origin' in data:
                self.origin.load(data['origin'])
            if 'ins' in data:
                self.ins.load(data['ins'])
            if 'dvl' in data:
                self.dvl.load(data['dvl'])
            if 'depth' in data:
                self.depth.load(data['depth'])
            if 'usbl' in data:
                self.usbl.load(data['usbl'])
            if 'camera1' in data:
                self.camera1.load(data['camera1'])
            if 'camera2' in data:
                self.camera2.load(data['camera2'])
            if 'camera3' in data:
                self.camera3.load(data['camera3'])
            if 'chemical' in data:
                self.camera3.load(data['chemical'])