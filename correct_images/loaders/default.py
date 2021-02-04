import cv2
import numpy as np

def loader(image_filepath, image_width=None, image_height=None, src_bit=8, dst_bit=8):
    """Default image loader using ImageIO

    Parameters
    ----------
    image_filepath : Path
        Image file path
    image_width : int
        Image width
    image_height : int
        Image height

    Returns
    -------
    np.ndarray
        Loaded image in matrix form (numpy)
    """
    image =  cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)
    image = image.astype(np.float32) * (2 **(dst_bit - src_bit))
    return image
