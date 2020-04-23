import numpy as np
import yaml
from pathlib import Path
import cv2


class MonoCamera():
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
            fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
            self.from_node(fs)

    def from_node(self, fs):
        self.name = fs.getNode('camera_name').string()
        self.image_width = int(fs.getNode('image_width').real())
        self.image_height = int(fs.getNode('image_height').real())
        self.K = fs.getNode('camera_matrix').mat()
        self.d = fs.getNode('distortion_coefficients').mat()
        self.R = fs.getNode('rectification_matrix').mat()
        self.P = fs.getNode('projection_matrix').mat()

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


class StereoCamera():
    def __init__(self, filename=None, left=None, right=None):
        self.left = MonoCamera(left)
        self.right = MonoCamera(right)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        if filename is not None:
            print(filename)
            filename = Path(filename)
            fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
            self.left.from_node(fs.getNode('left'))
            self.right.from_node(fs.getNode('right'))
            self.from_node(fs.getNode('extrinsics'))

    def from_node(self, fs):
        self.R = fs.getNode('rotation_matrix').mat()
        self.t = fs.getNode('translation_vector').mat()

    def to_str(self):
        msg = (""
               + "rotation_matrix:\n"
               + "  rows: 3\n"
               + "  cols: 3\n"
               + "  data: [" + ", ".join(["%8f" % i for i in self.R.reshape(1,9)[0]]) + "]\n"
               + "translation_vector:\n"
               + "  rows: 1\n"
               + "  cols: 3\n"
               + "  data: [" + ", ".join(["%8f" % self.t[i, 0] for i in range(self.t.shape[0])]) + "]\n")
        d = {}
        d['left'] = self.left.to_str()
        d['right'] = self.right.to_str()
        d['extrinsics'] = msg
        return yaml.dump(d)
