import numpy as np

def pixel_stat(img, img_mean, img_std, target_mean, target_std, dst_bit=8):
    """Generate target stats for images

    Parameters
    -----------
    img : numpy.ndarray
        image data to be corrected for target stats
    img_mean : int
        current mean
    img_std : int
        current std
    target_mean : int
        desired mean
    target_std : int
        desired std
    dst_bit : int
        destination bit depth

    Returns
    -------
    numpy.ndarray
        Corrected image
    """
    if len(img.shape) == 2:
        # Monochrome
        return pixel_stat_impl(img, img_mean, img_std, target_mean, target_std, dst_bit)
    elif len(img.shape) == 3:
        [a, b, c] = img.shape
        corrected_img = np.zeros((a, b, c), dtype=np.float32)
        for c in range(img.shape[3]):
            corrected_img[:, :, c] = pixel_stat_impl(img[:, :, c], img_mean[:, :, c], img_std[:, :, c], target_mean, target_std, dst_bit)
        return corrected_img
        # Colour

def pixel_stat_impl(img, img_mean, img_std, target_mean, target_std, dst_bit=8):
    """Generate target stats for single channel images

    Parameters
    -----------
    img : numpy.ndarray
        image data to be corrected for target stats
    img_mean : int
        current mean
    img_std : int
        current std
    target_mean : int
        desired mean
    target_std : int
        desired std
    dst_bit : int
        destination bit depth

    Returns
    -------
    numpy.ndarray
        Corrected image
    """
    image = (((img - img_mean) / img_std) * target_std) + target_mean
    image = np.clip(image, 0, 2 ** dst_bit - 1)
    return image
