import yaml
import sys
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


def warn_and_exit():
    Console.warn('If you specified a origin frame, check that they match. Otherwise, stick to default frame names:')
    Console.warn('dvl ins depth usbl camera1 camera2 camera3')
    Console.quit('Inconsistency between mission.yaml and vehicle.yaml')


class OriginEntry:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.crs = ''
        self.date = ''
        self._empty = True

    def empty(self):
        return self._empty

    def load(self, node):
        self._empty = False
        self.latitude = node['latitude']
        self.longitude = node['longitude']
        self.crs = node['coordinate_reference_system']
        self.date = node['date']

    def write(self, node):
        node['latitude'] = self.latitude
        node['longitude'] = self.longitude
        node['coordinate_reference_system'] = self.crs
        node['date'] = self.date


class CameraEntry:
    def __init__(self, node=None):
        if node is not None:
            self.name = node['name']
            self.type = node['type']
            self.path = node['path']
            if 'origin' in node:
                self.origin = node['origin']
                Console.info('Using camera ' + self.name + ' mounted at ' + self.origin)

    def write(self, node):
        node['name'] = self.name
        node['origin'] = self.origin
        node['type'] = self.type
        node['path'] = self.path


class ImageEntry:
    def __init__(self):
        self.format = ''
        self.timezone = 0
        self.timeoffset = 0
        self.cameras = []
        self.calibration = None
        self._empty = True

    def empty(self):
        return self._empty

    def load(self, node, version=1):
        self._empty = False
        self.format = node['format']
        self.timezone = node['timezone']
        # read in timezone
        if isinstance(self.timezone, str):
            if self.timezone == 'utc' or self.timezone == 'UTC':
                self.timezone = 0
            elif self.timezone == 'jst' or self.timezone == 'JST':
                self.timezone = 9
            else:
                try:
                    self.timezone = float(self.timezone)
                except ValueError:
                    print('Error: timezone', self.timezone,
                          'in mission.yaml not recognised, please enter value from UTC in hours')

        self.timeoffset = node['timeoffset']
        if version == 1:
            for camera in node['cameras']:
                self.cameras.append(CameraEntry(camera))
            if 'origin' not in node['cameras'][0]:
                Console.warn('Assuming that the camera order in mission.yaml corresponds to the order in vehicle.yaml')
                for i, camera in enumerate(self.cameras):
                    camera.origin = 'camera' + str(i + 1)
        else:
            self.cameras.append(CameraEntry())
            self.cameras.append(CameraEntry())
            if self.format == 'seaxerocks_3':
                self.cameras.append(CameraEntry())
                self.cameras[0].name = 'fore'
                self.cameras[0].origin = 'camera1'
                self.cameras[0].type = 'bayer_rggb'
                self.cameras[0].path = node['filepath'] + node['camera1']
                self.cameras[1].name = 'aft'
                self.cameras[1].origin = 'camera2'
                self.cameras[1].type = 'bayer_rggb'
                self.cameras[1].path = node['filepath'] + node['camera2']
                self.cameras[2].name = 'laser'
                self.cameras[2].origin = 'camera3'
                self.cameras[2].type = 'grayscale'
                self.cameras[2].path = node['filepath'] + node['camera3']
            elif self.format == 'acfr_standard':
                self.cameras[0].name = node['camera1']
                self.cameras[0].origin = 'camera1'
                self.cameras[0].type = 'bayer_rggb'
                self.cameras[0].path = node['filepath']
                self.cameras[1].name = node['camera2']
                self.cameras[1].origin = 'camera2'
                self.cameras[1].type = 'bayer_rggb'
                self.cameras[1].path = node['filepath']

    def write(self, node):
        node['format'] = self.format
        node['timezone'] = self.timezone
        node['timeoffset'] = self.timeoffset
        node['cameras'] = []
        for c in self.cameras:
            cam_dict = OrderedDict()
            c.write(cam_dict)
            node['cameras'].append(cam_dict)
        if 'calibration' in node:
            calibration_dict = OrderedDict()
            self.calibration.write(calibration_dict)
            node['calibration'].append(calibration_dict)


class DefaultEntry:
    def __init__(self):
        self.format = ''
        self.filepath = ''
        self.filename = ''
        self.timezone = 0
        self.timeoffset = 0
        self.label = 0
        self.std_factor = 0.0
        self.std_offset = 0.0
        self._empty = True

    def empty(self):
        return self._empty

    def load(self, node):
        self._empty = False
        self.format = node['format']
        self.timezone = node['timezone']
        self.timeoffset = node['timeoffset']
        if 'filepath' in node:
            self.filepath = node['filepath']
        if 'filename' in node:
            self.filename = node['filename']
        if 'label' in node:
            self.label = node['label']
        elif 'id' in node:
            self.label = node['id']
        if 'std_factor' in node:
            self.std_factor = node['std_factor']
        if 'std_offset' in node:
            self.std_offset = node['std_offset']
        if 'origin' in node:
            self.origin = node['origin']

    def write(self, node):
        node['format'] = self.format
        node['origin'] = self.origin
        node['timezone'] = self.timezone
        node['timeoffset'] = self.timeoffset
        node['filepath'] = self.filepath
        node['filename'] = self.filename
        node['label'] = self.label
        node['id'] = self.label
        node['std_factor'] = self.std_factor
        node['std_offset'] = self.std_offset


class Mission:
    def __init__(self, filename=None):
        self.version = 0
        self.origin = OriginEntry()
        self.velocity = DefaultEntry()
        self.orientation = DefaultEntry()
        self.depth = DefaultEntry()
        self.altitude = DefaultEntry()
        self.usbl = DefaultEntry()
        self.image = ImageEntry()
        self.tide = DefaultEntry()

        if filename is None:
            return
        try:
            # Check that mission.yaml and vehicle.yaml are consistent
            vehicle_file = filename.parent / 'vehicle.yaml'
            vehicle_stream = vehicle_file.open('r')
            vehicle_data = yaml.safe_load(vehicle_stream)

            with filename.open('r') as stream:
                data = yaml.safe_load(stream)
                if 'version' in data:
                    self.version = data['version']
                self.origin.load(data['origin'])
                if 'velocity' in data:
                    self.velocity.load(data['velocity'])
                    if 'origin' not in data['velocity']:
                        self.velocity.origin = 'dvl'
                    if self.velocity.origin in vehicle_data:
                        Console.info('Using velocity sensor mounted at ' + self.velocity.origin)
                    else:
                        Console.warn('The velocity sensor mounted at ' + self.velocity.origin + ' does not correspond to any frame in vehicle.yaml.')
                        warn_and_exit()
                if 'orientation' in data:
                    self.orientation.load(data['orientation'])
                    if 'origin' not in data['orientation']:
                        self.orientation.origin = 'ins'
                    if self.orientation.origin in vehicle_data:
                        Console.info('Using orientation sensor mounted at ' + self.orientation.origin)
                    else:
                        Console.warn('The orientation sensor mounted at ' + self.orientation.origin + ' does not correspond to any frame in vehicle.yaml.')
                        warn_and_exit()
                if 'depth' in data:
                    self.depth.load(data['depth'])
                    if 'origin' not in data['depth']:
                        self.depth.origin = 'depth'
                    if self.depth.origin in vehicle_data:
                        Console.info('Using depth sensor mounted at ' + self.depth.origin)
                    else:
                        Console.warn('The depth sensor mounted at ' + self.depth.origin + ' does not correspond to any frame in vehicle.yaml.')
                        warn_and_exit()
                if 'altitude' in data:
                    self.altitude.load(data['altitude'])
                    if 'origin' not in data['altitude']:
                        self.altitude.origin = 'dvl'
                    if self.altitude.origin in vehicle_data:
                        Console.info('Using altitude sensor mounted at ' + self.altitude.origin)
                    else:
                        Console.warn('The altitude sensor mounted at ' + self.altitude.origin + ' does not correspond to any frame in vehicle.yaml.')
                        warn_and_exit()
                if 'usbl' in data:
                    self.usbl.load(data['usbl'])
                    if 'origin' not in data['usbl']:
                        self.usbl.origin = 'usbl'
                    if self.usbl.origin in vehicle_data:
                        Console.info('Using usbl sensor mounted at ' + self.usbl.origin)
                    else:
                        Console.warn('The usbl sensor mounted at ' + self.usbl.origin + ' does not correspond to any frame in vehicle.yaml.')
                        warn_and_exit()

                if 'tide' in data:
                    self.tide.load(data['tide'])

                if 'image' in data:
                    self.image.load(data['image'], self.version)
        except FileNotFoundError:
            Console.error('The file mission.yaml could not be found at the location:')
            Console.error(filename)
            Console.quit('mission.yaml not provided')
        except PermissionError:
            Console.error('The file mission.yaml could not be opened at the location:')
            Console.error(filename)
            Console.error('Please make sure you have the correct access rights.')
            Console.quit('mission.yaml not provided')

    def write_metadata(self, node):
        node['username'] = Console.get_username()
        node['date'] = Console.get_date()
        node['hostname'] = Console.get_hostname()
        node['version'] = Console.get_version()

    def write(self, filename):
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        with filename.open('w') as f:
            mission_dict = OrderedDict()
            mission_dict['version'] = 1
            mission_dict['metadata'] = OrderedDict()
            mission_dict['origin'] = OrderedDict()
            self.write_metadata(mission_dict['metadata'])
            self.origin.write(mission_dict['origin'])
            if not self.velocity.empty():
                mission_dict['velocity'] = OrderedDict()
                self.velocity.write(mission_dict['velocity'])
            if not self.orientation.empty():
                mission_dict['orientation'] = OrderedDict()
                self.orientation.write(mission_dict['orientation'])
            if not self.depth.empty():
                mission_dict['depth'] = OrderedDict()
                self.depth.write(mission_dict['depth'])
            if not self.altitude.empty():
                mission_dict['altitude'] = OrderedDict()
                self.altitude.write(mission_dict['altitude'])
            if not self.usbl.empty():
                mission_dict['usbl'] = OrderedDict()
                self.usbl.write(mission_dict['usbl'])
            if not self.image.empty():
                mission_dict['image'] = OrderedDict()
                self.image.write(mission_dict['image'])
            yaml.dump(mission_dict, f, allow_unicode=True, default_flow_style=False)
