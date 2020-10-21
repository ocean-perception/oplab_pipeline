import numpy as np


def gamma_correct(image, bitdepth=8):
    """ performs gamma correction for images
    Parameters
    -----------
    image : numpy.ndarray
        image data to be corrected for gamma
    bitdepth : int
        target bitdepth for output image


    Returns
    -------
    numpy.ndarray
        Image
    """
    image = np.divide(image, (2 ** bitdepth - 1))
    if all(i < 0.0031308 for i in image.flatten()):
        image = 12.92 * image
    else:
        image = 1.055 * np.power(image, (1 / 1.5)) - 0.055
    image = np.multiply(np.array(image), np.array(2 ** bitdepth - 1))
    image = np.clip(image, 0, 2 ** bitdepth - 1)
    return image
