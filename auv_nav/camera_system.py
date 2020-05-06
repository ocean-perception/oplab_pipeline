import yaml
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.console import Console
from auv_nav.tools.filename_to_date import FilenameToDate
# Workaround to dump OrderedDict into YAML files
from pathlib import Path
import imageio


class CameraEntry:
    def __init__(self, node=None, path_to_mission=None):
        self._image_list = []
        self._stamp_list = []
        self.path_to_mission = path_to_mission
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
                Console.error('The camera ', self.name, ' is missing its timestamp format')
                Console.error('You can provide it by means of filename:')
                Console.error('e.g. PR_20180811_153729_762_RC16.tif -> xxxYYYYMMDDxhhmmssxfffxxxxx.xxx')
                Console.error('or using a separate timestamp file:')
                Console.error('e.g. FileTime.csv, where separate columns z define the date.')
                Console.error('Find examples in default_yaml folder.')
                Console.quit('Missing timestamp format for a camera.')
            self.convert_filename = FilenameToDate(
                self.filename_to_date,
                self.timestamp_file,
                self.columns,
                path_to_mission)

    def write(self, node):
        pass

    @property
    def image_list(self):
        if len(self._image_list) > 0:
            return self._image_list
        if self.path_to_mission is None:
            curr_dir = Path.cwd()
        else:
            curr_dir = self.path_to_mission
        raw_dir = get_raw_folder(curr_dir)
        img_dir = raw_dir.glob(self.path)
        self._image_list = []
        for i in img_dir:
            if self.name == 'LC':
                [self._image_list.append(str(_)) for _ in i.rglob('*LC*.' + self.extension)]
            elif self.name == 'RC':
                [self._image_list.append(str(_)) for _ in i.rglob('*RC*.' + self.extension)]
            else:
                [self._image_list.append(str(_)) for _ in i.rglob('*.' + self.extension)]
     
        self._image_list.sort()
        return self._image_list

    @property
    def image_properties(self):
        imagelist = self.image_list
        image_path = imagelist[0]
        self._image_properties = []
        
        # read tiff
        if self.extension == 'tif':
            image_matrix = imageio.imread(image_path)
            image_shape = image_matrix.shape
            self._image_properties.append(image_shape[0])
            self._image_properties.append(image_shape[1])
            if len(image_shape) == 3:
                self._image_properties.append(image_shape[2])
            else:
                self._image_properties.append(1)

        # read raw
        if self.extension == 'raw':
            self._image_properties = [1024, 1280, 1]
        return self._image_properties
    

    @property
    def stamp_list(self):
        if len(self._stamp_list) > 0:
            return self._stamp_list
        self._stamp_list = []
        for p in self.image_list:
            n = Path(p).name
            self._stamp_list.append(self.convert_filename(n))
        return self._stamp_list


class CameraSystem:
    def __init__(self, filename=None, path_to_mission=None):
        self.cameras = []
        self.camera_system = None
        if filename is None:
            return
        if isinstance(filename, str):
            filename = Path(filename)
        data = ''
        try:
            with filename.open('r') as stream:
                data = yaml.safe_load(stream)
        except FileNotFoundError:
            Console.error('The file camera.yaml could not be found at ', filename)
            Console.quit('camera.yaml not provided')
        except PermissionError:
            Console.error('The file camera.yaml could not be opened at ', filename)
            Console.error(filename)
            Console.error('Please make sure you have the correct access rights.')
            Console.quit('camera.yaml not provided')
        self._parse(data, path_to_mission)

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

    def _parse(self, node, path_to_mission):
        if 'camera_system' not in node:
            Console.error('The camera.yaml file is missing the camera_system entry.')
            Console.quit('Wrong camera.yaml format or content.')
        self.camera_system = node['camera_system']

        if 'cameras' not in node:
            Console.error('The camera.yaml file is missing the cameras entry.')
            Console.quit('Wrong camera.yaml format or content.')
        for camera in node['cameras']:
            self.cameras.append(CameraEntry(camera, path_to_mission))