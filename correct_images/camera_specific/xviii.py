from numba import njit
import numpy as np
from pathlib import Path


# read binary raw image files for xviii camera
@njit
def load_xviii_bayer_from_binary(binary_data, image_height, image_width):
    """Read XVIII binary images into bayer array

    Parameters
    -----------
    binary_data : numpy.ndarray
        binary image data from XVIII
    image_height : int
        image height
    image_width : int
        image width

    Returns
    --------
    numpy.ndarray
        Bayer image
    """

    img_h = image_height
    img_w = image_width
    bayer_img = np.zeros((img_h, img_w), dtype=np.uint32)

    # read raw data and put them into bayer pattern.
    count = 0
    for i in range(0, img_h, 1):
        for j in range(0, img_w, 4):
            chunk = binary_data[count : count + 12]
            bayer_img[i, j] = (
                ((chunk[3] & 0xFF) << 16) | ((chunk[2] & 0xFF) << 8) | (chunk[1] & 0xFF)
            )
            bayer_img[i, j + 1] = (
                ((chunk[0] & 0xFF) << 16) | ((chunk[7] & 0xFF) << 8) | (chunk[6] & 0xFF)
            )
            bayer_img[i, j + 2] = (
                ((chunk[5] & 0xFF) << 16)
                | ((chunk[4] & 0xFF) << 8)
                | (chunk[11] & 0xFF)
            )
            bayer_img[i, j + 3] = (
                ((chunk[10] & 0xFF) << 16)
                | ((chunk[9] & 0xFF) << 8)
                | (chunk[8] & 0xFF)
            )
            count += 12

    bayer_img = bayer_img / 1024
    return bayer_img


def xviii_to_np_file(np_filename, raw_filename, dtype, image_height, image_width):
    np_fn_path = Path(np_filename)
    if not np_fn_path.exists():
        binary_data = np.fromfile(raw_filename, dtype)
        image_raw = load_xviii_bayer_from_binary(
                binary_data[:], image_height, image_width
            )
        np.save(np_filename, image_raw)