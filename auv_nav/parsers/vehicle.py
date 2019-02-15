import yaml
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.tools.folder_structure import get_raw_folder


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

    def load(self, node, mission_node = []):
        self._empty = False
        if 'surge_m' in node:
            self.surge = node['surge_m']
            self.sway = node['sway_m']
            self.heave = node['heave_m']
            self.roll = node['roll_deg']
            self.pitch = node['pitch_deg']
            self.yaw = node['yaw_deg']
        elif 'x_offset' in node:
            self.surge = node['x_offset']
            self.sway = node['y_offset']
            self.heave = node['z_offset']
            if mission_node:
                self.yaw = mission_node['headingoffset']

    def print(self, name):
        print('{}: XYZ ({:.2f}, {:.2f}, {:.2f}) RPY ({:.2f}, {:.2f}, {:.2f})'.format(name, self.surge, self.sway, self.heave, self.roll, self.pitch, self.yaw))

class Vehicle:
    def __init__(self, filename=None):
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

        if filename is None:
            return

        vehicle_file = get_raw_folder(filename)
        mission_file = vehicle_file.parents[0] / 'mission.yaml'
        old_format = False
        mission_data = []

        with vehicle_file.open('r') as stream:
            data = yaml.load(stream)
            if 'origin' in data:
                self.origin.load(data['origin'])
                if 'x_offset' in data['origin']:
                    mission_stream = mission_file.open('r')
                    mission_data = yaml.load(mission_stream)
                    old_format = True
            if 'ins' in data:
                if old_format:
                    self.ins.load(data['ins'], mission_data['orientation'])
                else:
                    self.ins.load(data['ins'])
            if 'dvl' in data:
                if old_format:
                    self.dvl.load(data['dvl'], mission_data['velocity'])
                else:
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
                self.chemical.load(data['chemical'])
