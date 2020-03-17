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
            if mission_node and 'headingoffset' in mission_node:
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
        self.chemical = SensorOffset()

        if filename is None:
            return

        self.filename = filename

        mission_file = filename.parent / 'mission.yaml'
        old_format = False

        from auv_nav.parsers.mission import Mission
        mission = Mission(mission_file)
        mission_stream = mission_file.open('r')
        mission_data = yaml.safe_load(mission_stream)

        try:
            with filename.open('r') as stream:
                self.data = OrderedDict()
                self.data = yaml.safe_load(stream)
                if 'origin' in self.data:
                    self.origin.load(self.data['origin'])
                    if 'x_offset' in self.data['origin']:
                        old_format = True
                if 'ins' in self.data:
                    if old_format:
                        self.ins.load(self.data['ins'], mission_data['orientation'])
                    else:
                        self.ins.load(self.data['ins'])
                if 'dvl' in self.data:
                    if old_format:
                        self.dvl.load(self.data['dvl'], mission_data['velocity'])
                    else:
                        self.dvl.load(self.data['dvl'])
                if 'depth' in self.data:
                    self.depth.load(self.data['depth'])
                if 'usbl' in self.data:
                    self.usbl.load(self.data['usbl'])
                if len(mission.image.cameras) > 0:
                    camera_name = mission.image.cameras[0].name
                    if camera_name not in self.data:
                        Console.error(
                            'Could not find the position of the camera with name: ', camera_name)
                        Console.error('Please make sure that the name used in mission.yaml and \
                            the frame name used in vehicle.yaml are the same.')
                        Console.quit(
                            'Your vehicle.yaml or your mission.yaml are malformed.')
                    self.camera1.load(self.data[camera_name])
                if len(mission.image.cameras) > 1:
                    camera_name = mission.image.cameras[1].name
                    if camera_name not in self.data:
                        Console.error(
                            'Could not find the position of the camera with name: ', camera_name)
                        Console.error('Please make sure that the name used in mission.yaml and \
                            the frame name used in vehicle.yaml are the same.')
                        Console.quit(
                            'Your vehicle.yaml or your mission.yaml are malformed.')
                    self.camera2.load(self.data[camera_name])
                if len(mission.image.cameras) > 2:
                    camera_name = mission.image.cameras[2].name
                    if camera_name not in self.data:
                        Console.error(
                            'Could not find the position of the camera with name: ', camera_name)
                        Console.error('Please make sure that the name used in mission.yaml and \
                            the frame name used in vehicle.yaml are the same.')
                        Console.quit(
                            'Your vehicle.yaml or your mission.yaml are malformed.')
                    self.camera3.load(self.data[camera_name])
                if 'chemical' in self.data:
                    self.chemical.load(self.data['chemical'])
        except FileNotFoundError:
            Console.error('The file vehicle.yaml could not be found at the location:')
            Console.error(filename)
            Console.quit('vehicle.yaml not provided')
        except PermissionError:
            Console.error(
                'The file vehicle.yaml could not be opened at the location:')
            Console.error(filename)
            Console.error('Please make sure you have the correct access rights.')
            Console.quit('vehicle.yaml not provided')

    def write_metadata(self, node):
        node['username'] = Console.get_username()
        node['date'] = Console.get_date()
        node['hostname'] = Console.get_hostname()
        node['firmware'] = Console.get_version()

    def write(self, filename):
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        self.filename.copy(filename)
        with filename.open('a') as f:
            f.write('\n\nmetadata:\n')
            f.write('    username: ' + str(Console.get_username()) + '\n')
            f.write('    date: ' + str(Console.get_date()) + '\n')
            f.write('    hostname: ' + str(Console.get_hostname()) + '\n')
            f.write('    firmware: ' + str(Console.get_version()) + '\n')
