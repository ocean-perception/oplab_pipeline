import yaml
import sys
from auv_nav.tools.folder_structure import get_config_folder


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


class CameraEntry:
    def __init__(self, node):
        self.name = node['name']
        self.type = node['type']
        self.path = node['path']


class ImageEntry:
    def __init__(self):
        self.format = ''
        self.timezone = 0
        self.timeoffset = 0
        self.cameras = []
        self._empty = True

    def empty(self):
        return self._empty

    def load(self, node):
        self._empty = False
        self.format = node['format']
        self.timezone = node['timezone']
        self.timeoffset = node['timeoffset']
        for camera in node['cameras']:
            self.cameras.append(CameraEntry(camera))


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
        if 'id' in node:
            self.id = node['id']
        if 'std_factor' in node:
            self.std_factor = node['std_factor']
        if 'std_offset' in node:
            self.std_offset = node['std_offset']


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

        mission_file = get_config_folder(filename)
        with mission_file.open('r') as stream:
            data = yaml.load(stream)
            self.version = data['version']

            if self.version != 1:
                print('--------------------------------------------')
                print('| ERROR: Mission version not supported.    |')
                print('| Please convert your mission to version 1 |')
                print('--------------------------------------------')
                print('You are using and old mission.yaml format that is no \
                       longer compatible. Please refer to the example \
                       mission.yaml file and modify yours to fit.')
                print('auv_nav will now exit')
                sys.exit(1)

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
                self.image.load(data['image'])
