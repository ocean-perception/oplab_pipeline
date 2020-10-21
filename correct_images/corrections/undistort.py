from oplab import MonoCamera
import numpy as np
import cv2


# correct image for distortions using camera calibration parameters
def distortion_correct(camera_params_file_path, image, dst_bit=8):
    """Perform distortion correction for images

    Parameters
    -----------
    camera_params_file_path: str
        Path to the camera parameters file
    image : numpy.ndarray
        image data to be corrected for distortion
    dst_bit : int
        target bitdepth for output image

    Returns
    -------
    numpy.ndarray
        Image
    """

    monocam = MonoCamera(camera_params_file_path)
    map_x, map_y = monocam.rectification_maps
    image = np.clip(image, 0, 2 ** dst_bit - 1)
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return image
