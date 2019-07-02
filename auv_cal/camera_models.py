import numpy as np
import yaml
from pathlib import Path


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


class StereoCamera():
    def __init__(self, filename=None, left=None, right=None):
        self.left = MonoCamera(left)
        self.right = MonoCamera(right)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        if filename is not None:
            filename = Path(filename)
            stream = filename.open('r')
            d = yaml.safe_load(stream)
            self.left.from_node(d['left'])
            self.right.from_node(d['right'])
            self.from_node(d['extrinsics'])

    def from_node(self, node):
        self.R = cv2np(node['rotation_matrix'])
        self.t = cv2np(node['translation_vector'])

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
