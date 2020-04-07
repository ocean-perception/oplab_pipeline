import yaml
import sys
from auv_nav.tools.folder_structure import get_config_folder
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.console import Console
# Workaround to dump OrderedDict into YAML files
from collections import OrderedDict
from pathlib import Path


class CameraEntry:
    def __init__(self, node=None):
        if node is not None:
            self.name = node['name']
            self.type = node['type']
            self.bit_depth = node['bit_depth']
            self.path = node['path']
            self.extension = node['extension']
            self.timestamp_file = node.get('timestamp_file', None)
            self.columns = node.get('columns', None)
            self.filename_to_date = node.get('filename_to_date', None)
            if self.timestamp_file is None and self.filename_to_date is None:
                Console.error('The camera ', self.name, ' is missing its \
                    timestamp format')
                Console.error('You can provide it by means of filename:')
                Console.error('e.g. ')
                Console.error('or using a separate timestamp file:')
                Console.error('e.g. ')
                Console.quit('Missing timestamp format for a camera.')

    def write(self, node):
        node['name'] = self.name
        node['origin'] = self.origin
        node['type'] = self.type
        node['path'] = self.path

    def get_image_list(self):
        curr_dir = Path.cwd()
        raw_dir = get_raw_folder(curr_dir)
        img_dir = raw_dir.glob(self.path)
        img_list = []
        for i in img_dir:
            [img_list.append(str(_)) for _ in i.rglob('*.' + self.extension)]
        img_list.sort()
        return img_list


class CameraSystem:
    def __init__(self, filename=None):
        self.cameras = []
        self.camera_system = None

        if filename is None:
            return

        if isinstance(filename, str):
            filename = Path(filename)
        
        try:
            with filename.open('r') as stream:
                data = yaml.safe_load(stream)
                self._parse(data)
        except FileNotFoundError:
            Console.error('The file camera.yaml could not be found \
                at ', filename)
            Console.quit('camera.yaml not provided')
        except PermissionError:
            Console.error('The file camera.yaml could not be opened \
                at ', filename)
            Console.error(filename)
            Console.error('Please make sure you have the correct \
                access rights.')
            Console.quit('camera.yaml not provided')

    def __str__(self):
        msg = ''
        if self.camera_system is not None:
            msg += 'CameraSystem: ' + str(self.camera_system)
            if len(self.cameras) > 0:
                msg += ' with cameras ['
                for c in self.cameras:
                    msg += str(c.name) + ' '
                msg += ']'
            else:
                msg += ' is empty'
        else: 
            msg += 'Empty CameraSystem'
        return msg

    def _parse(self, node):
        if 'camera_system' not in node:
            Console.error('The camera.yaml file is missing the \
                camera_system entry.')
            Console.quit('Wrong camera.yaml format or content.')
        self.camera_system = node['camera_system']

        if 'cameras' not in node:
            Console.error('The camera.yaml file is missing the \
                cameras entry.')
            Console.quit('Wrong camera.yaml format or content.')
        for camera in node['cameras']:
            self.cameras.append(CameraEntry(camera))