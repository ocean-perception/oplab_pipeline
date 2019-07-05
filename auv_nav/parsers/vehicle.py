import yaml
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.console import Console
# Workaround to dump OrderedDict into YAML files
from collections import OrderedDict


def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


yaml.add_representer(OrderedDict, represent_ordereddict)


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
        return self._empty

    def load(self, node, mission_node=[]):
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

    def write(self, node):
        node['surge_m'] = self.surge
        node['sway_m'] = self.sway
        node['heave_m'] = self.heave
        node['roll_deg'] = self.roll
        node['pitch_deg'] = self.pitch
        node['yaw_deg'] = self.yaw

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

        mission_file = filename.parent / 'mission.yaml'
        old_format = False
        mission_data = []

        try:
            with filename.open('r') as stream:
                data = yaml.safe_load(stream)
                self.data = data
                if 'origin' in data:
                    self.origin.load(data['origin'])
                    if 'x_offset' in data['origin']:
                        mission_stream = mission_file.open('r')
                        mission_data = yaml.safe_load(mission_stream)
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
        except FileNotFoundError:
            Console.error('The file vehicle.yaml could not be found at the location:')
            Console.error(filename)
            Console.quit('vehicle.yaml not provided')
        except PermissionError:
            Console.error('The file mission.yaml could not be opened at the location:')
            Console.error(filename)
            Console.error('Please make sure you have the correct access rights.')
            Console.quit('vehicle.yaml not provided')

    def write_metadata(self, node):
        node['username'] = Console.get_username()
        node['date'] = Console.get_date()
        node['hostname'] = Console.get_hostname()
        node['version'] = Console.get_version()

    def write(self, filename):
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        with filename.open('w') as f:
            vehicle_dict = OrderedDict()
            vehicle_dict['version'] = 1
            vehicle_dict['metadata'] = OrderedDict()
            self.write_metadata(vehicle_dict['metadata'])
            vehicle_dict['origin'] = OrderedDict()
            self.origin.write(vehicle_dict['origin'])
            if not self.ins.empty():
                vehicle_dict['ins'] = OrderedDict()
                self.ins.write(vehicle_dict['ins'])
            if not self.dvl.empty():
                vehicle_dict['dvl'] = OrderedDict()
                self.dvl.write(vehicle_dict['dvl'])
            if not self.depth.empty():
                vehicle_dict['depth'] = OrderedDict()
                self.depth.write(vehicle_dict['depth'])
            if not self.usbl.empty():
                vehicle_dict['usbl'] = OrderedDict()
                self.usbl.write(vehicle_dict['usbl'])
            if not self.camera1.empty():
                vehicle_dict['camera1'] = OrderedDict()
                self.camera1.write(vehicle_dict['camera1'])
            if not self.camera2.empty():
                vehicle_dict['camera2'] = OrderedDict()
                self.camera2.write(vehicle_dict['camera2'])
            if not self.camera3.empty():
                vehicle_dict['camera3'] = OrderedDict()
                self.camera3.write(vehicle_dict['camera3'])
            if not self.chemical.empty():
                vehicle_dict['chemical'] = OrderedDict()
                self.chemical.write(vehicle_dict['chemical'])
            yaml.dump(vehicle_dict, f, allow_unicode=True, default_flow_style=False)
