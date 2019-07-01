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
            if 'camera_calibration' in node:
                self.camera_calibration = node['camera_calibration']
            if 'laser_calibration' in node:
                self.laser_calibration = node['laser_calibration']

    def write(self, node):
        node['name'] = self.name
        node['type'] = self.type
        node['path'] = self.path
        if 'camera_calibration' in node:
            node['camera_calibration'] = self.camera_calibration
        if 'laser_calibration' in node:
            node['laser_calibration'] = self.laser_calibration


class CalibrationEntry:
    def __init__(self, node=None):
        if node is not None:
            self.pattern = node['pattern']
            self.cols = node['cols']
            self.rows = node['rows']
            self.size = node['size']

    def write(self, node):
        node['pattern'] = self.pattern
        node['cols'] = self.cols
        node['rows'] = self.rows
        node['size'] = self.size


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
        self.timeoffset = node['timeoffset']
        if version == 1:
            for camera in node['cameras']:
                self.cameras.append(CameraEntry(camera))
        else:
            self.cameras.append(CameraEntry())
            self.cameras.append(CameraEntry())
            if self.format == 'seaxerocks_3':
                self.cameras.append(CameraEntry())
                self.cameras[0].name = 'fore'
                self.cameras[0].type = 'bayer_rggb'
                self.cameras[0].path = node['filepath'] + node['camera1']
                self.cameras[1].name = 'aft'
                self.cameras[1].type = 'bayer_rggb'
                self.cameras[1].path = node['filepath'] + node['camera2']
                self.cameras[2].name = 'laser'
                self.cameras[2].type = 'grayscale'
                self.cameras[2].path = node['filepath'] + node['camera3']
            elif self.format == 'acfr_standard':
                self.cameras[0].name = node['camera1']
                self.cameras[0].type = 'bayer_rggb'
                self.cameras[0].path = node['filepath']
                self.cameras[1].name = node['camera2']
                self.cameras[1].type = 'bayer_rggb'
                self.cameras[1].path = node['filepath']
        if 'calibration' in node:
            self.calibration = CalibrationEntry(node['calibration'])

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

    def write(self, node):
        node['format'] = self.format
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

        if filename is None:
            return
        try:
            mission_file = get_raw_folder(filename)
            with mission_file.open('r') as stream:
                data = yaml.safe_load(stream)
                if 'version' in data:
                    self.version = data['version']
                self.origin.load(data['origin'])
                if 'velocity' in data:
                    self.velocity.load(data['velocity'])
                if 'orientation' in data:
                    self.orientation.load(data['orientation'])
                if 'depth' in data:
                    self.depth.load(data['depth'])
                if 'altitude' in data:
                    self.altitude.load(data['altitude'])
                if 'usbl' in data:
                    self.usbl.load(data['usbl'])
                if 'image' in data:
                    self.image.load(data['image'], self.version)
        except FileNotFoundError:
            Console.error('The file mission.yaml could not be found at the location:')
            Console.error(mission_file)
            Console.quit('mission.yaml not provided')
        except PermissionError:
            Console.error('The file mission.yaml could not be opened at the location:')
            Console.error(mission_file)
            Console.error('Please make sure you have the correct access rights.')
            Console.quit('mission.yaml not provided')

    def write_metadata(self, node):
        node['username'] = Console.get_username()
        node['date'] = Console.get_date()
        node['hostname'] = Console.get_hostname()
        node['version'] = Console.get_version()

    def write(self, filename):
        with filename.open('w') as f:
            mission_dict = OrderedDict()
            mission_dict['version'] = 1
            mission_dict['metadata'] = OrderedDict()
            mission_dict['origin'] = OrderedDict()
            mission_dict['velocity'] = OrderedDict()
            mission_dict['orientation'] = OrderedDict()
            mission_dict['depth'] = OrderedDict()
            mission_dict['altitude'] = OrderedDict()
            mission_dict['usbl'] = OrderedDict()
            mission_dict['image'] = OrderedDict()
            self.write_metadata(mission_dict['metadata'])
            self.origin.write(mission_dict['origin'])
            self.velocity.write(mission_dict['velocity'])
            self.orientation.write(mission_dict['orientation'])
            self.depth.write(mission_dict['depth'])
            self.altitude.write(mission_dict['altitude'])
            self.usbl.write(mission_dict['usbl'])
            self.image.write(mission_dict['image'])
            yaml.dump(mission_dict, f, allow_unicode=True, default_flow_style=False)
