import os
import imageio
import re
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from pathlib import Path
import yaml

def generate_filelist(directory):
    files = []
    if not os.path.isdir(directory):
        raise ValueError(str(directory)+ " is not a directory!")

    for file in os.listdir(directory):
        if file[-3:] in ["jpg", "png","tif"]:
            files.append(file)

    files_str = '\n'.join(files)
    filelist = open(os.path.join(directory, "flist.txt"), 'w')
    filelist.write(files_str)
    filelist.close()
    return "flist.txt"


def get_filenames(directory, filelist):
    file = open(os.path.join(directory, filelist), 'r')
    filenames = file.read().rstrip().split("\n")
    return filenames


def validate_filenames(directory, filenames):
    for filename in filenames:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        if filename == "":
            raise ValueError("filename can't be empty, check for empty entries in the file list text file")
        if not os.path.isfile(os.path.join(directory, filename)):
            raise ValueError(os.path.join(directory, filename) + " isn't a file, check entries in the file list text file")
    return True


def get_outpath(path, method, target_mean, target_std):
    outpath = raw_path_to_processed_path(path)
    outpath = os.path.join(outpath, method, "mean_" + str(target_mean) + "_" + "std_" + str(target_std))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    return outpath


def raw_path_to_processed_path(path):
    if not isinstance(path, str):
        raise TypeError("The path argument you passed isn't a string value")

    index = re.search(r'[^\\/]', path).start()
    if index > 0:
        outpath = path[0:index]
    else:
        outpath = ""

    path = path.replace('\\', '/')
    sub_path = path.split('/')
    is_subfolder_of_processed = False
    for i in range(len(sub_path)-1, -1, -1):
        if sub_path[i] == "raw":
            sub_path[i] = "processed"
            is_subfolder_of_processed = True
            break
    if not is_subfolder_of_processed:
        raise ValueError("The path you provided is not a subfolder of a folder called 'raw'")

    for current_folder in sub_path:
        if len(current_folder)>0 and current_folder[-1] == ':':
            # Special case for directories on root directories (e.g. C:/) on Windows
            current_folder = current_folder + '/'
        outpath = os.path.join(outpath, current_folder)

    return outpath


def getBitDepth(image):
    if image.dtype == "uint8":
        return 8
    elif image.dtype == "uint16":
        return 16
    else:
        raise Exception("ERROR, not uint8 or uint16")


def adjust_to_bitdepth(f, bitdepth):
    return (float(f)/100)*((2**int(bitdepth))-1)


def read_image(filepath, debayer, filter_pattern):
    I = imageio.imread(filepath)
    if debayer:
        I = np.array(I)
        # I = np.array(demosaicing_CFA_Bayer_Menon2007(I, filter_pattern))
    else:
        I = np.array(I)
    if I.ndim == 3:
        return I
    elif I.ndim == 2:
        return np.reshape(I, (I.shape[0], I.shape[1], 1))
    else:
        raise Exception("Invalid image file, number of dimensions should be 2 for greyscale or 3 for rgb, bgr, etc")

def cv2np(node):
    rows = node['rows']
    cols = node['cols']
    if rows == 1:
        a = np.array(node['data']).reshape((int(cols), 1))
    elif cols == 1:
        a = np.array(node['data']).reshape((int(rows), 1))
    else:
        a = np.asarray(node['data']).reshape((int(rows), int(cols)))
    return a

class MonoCamera:
    def __init__(self, filename=None):
        self.K = np.zeros((3, 3))
        self.d = np.zeros((5, 1))
        self.R = np.eye(3)
        self.P = np.zeros((3, 4))
        self.image_width = 0
        self.image_height = 0
        self.name = ''

        if filename is not None:
            filename = Path(filename)
            stream = filename.open('r')
            d = yaml.safe_load(stream)
            self.from_node(d)

    def from_node(self, node):
        self.name = node['camera_name']
        self.image_width = node['image_width']
        self.image_height = node['image_height']
        self.K = cv2np(node['camera_matrix'])
        self.d = cv2np(node['distortion_coefficients'])
        self.R = cv2np(node['rectification_matrix'])
        self.P = cv2np(node['projection_matrix'])

    def to_str(self):
        msg = (""
               + "image_width: " + str(self.image_width) + "\n"
               + "image_height: " + str(self.image_height) + "\n"
               + "camera_name: " + self.name + "\n"
               + "camera_matrix:\n"
               + "  rows: 3\n"
               + "  cols: 3\n"
               + "  data: [" + ", ".join(["%8f" % i for i in self.K.reshape(1,9)[0]]) + "]\n"
               + "distortion_model: " + ("rational_polynomial" if self.d.size > 5 else "plumb_bob") + "\n"
               + "distortion_coefficients:\n"
               + "  rows: 1\n"
               + "  cols: 5\n"
               + "  data: [" + ", ".join(["%8f" % self.d[i,0] for i in range(self.d.shape[0])]) + "]\n"
               + "rectification_matrix:\n"
               + "  rows: 3\n"
               + "  cols: 3\n"
               + "  data: [" + ", ".join(["%8f" % i for i in self.R.reshape(1,9)[0]]) + "]\n"
               + "projection_matrix:\n"
               + "  rows: 3\n"
               + "  cols: 4\n"
               + "  data: [" + ", ".join(["%8f" % i for i in self.P.reshape(1,12)[0]]) + "]\n"
               + "")
        return msg
