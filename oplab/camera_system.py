import yaml
from pathlib import Path
import imageio
from oplab import get_raw_folder
from oplab import Console
from oplab import FilenameToDate


class CameraEntry:
    """
    Camera class to read filenames and timestamps.

    Parameters
    ---------
    node
        A YAML dictionary that contains the camera information.
    """
    def __init__(self, node=None):
        self._image_list = []
        self._stamp_list = []
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
                self.columns)

    def write(self, node):
        pass

    @property
    def image_list(self):
        """
        Method to retrieve the list of images taken with the camera.

        Returns:
            list: a list of image paths
        """
        if len(self._image_list) > 0:
            return self._image_list
        curr_dir = Path.cwd()
        raw_dir = get_raw_folder(curr_dir)

        split_glob = str(self.path).split('*')
        img_dir = ''
        print(split_glob)
        if len(split_glob) == 2:
            pre_glob = split_glob[0] + '*'
            glob_vec = raw_dir.glob(pre_glob)    
            img_dirs = [k for k in glob_vec]
            print(img_dirs)
            img_dir = Path(str(img_dirs[0]) + '/' + str(split_glob[1]))
        elif len(split_glob) == 1:
            img_dir = self.path
        else:
            Console.error('Multiple globbing is not supported.')
        print(img_dir)
        [self._image_list.append(str(_)) for _ in img_dir.rglob('*.' + self.extension)]
        self._image_list.sort()
        return self._image_list

    @property
    def stamp_list(self):
        """
        Method to retrieve the list of images taken with the camera.

        Returns:
            list: a list of timestamps
        """
        if len(self._stamp_list) > 0:
            return self._stamp_list
        self._stamp_list = []
        for p in self.image_list:
            n = Path(p).name
            self._stamp_list.append(self.convert_filename(n))
        return self._stamp_list

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
            #TODO: provide a raw reader and get these properties from the file.
            self._image_properties = [1024, 1280, 1]
        return self._image_properties


class CameraSystem:
    """Class to describe a camera system, for example, SeaXerocks3 or BioCam.
    Parses camera.yaml files that define camera systems mounted on ROVs or AUVs
    """
    def __init__(self, filename=None):
        """Constructor of camera system. If a filename is provided, the file will be parsed and its contents loaded into the class.

        Keyword Arguments:
            filename {string} -- Path to the camera.yaml file describing the camera system (default: {None})
        """        
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
        self._parse(data)

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
            Console.error('The camera.yaml file is missing the camera_system entry.')
            Console.quit('Wrong camera.yaml format or content.')
        self.camera_system = node['camera_system']

        if 'cameras' not in node:
            Console.error('The camera.yaml file is missing the cameras entry.')
            Console.quit('Wrong camera.yaml format or content.')
        for camera in node['cameras']:
            self.cameras.append(CameraEntry(camera))