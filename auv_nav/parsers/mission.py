import yaml
import sys
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.tools.folder_structure import get_raw_folder


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
    def __init__(self, node = None):
        if node is not None:
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

    def load(self, node, version = 1):
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

        mission_file = get_raw_folder(filename)
        with mission_file.open('r') as stream:
            data = yaml.load(stream)
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
