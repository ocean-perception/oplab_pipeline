import imageio

def loader(image_filepath, image_width=None, image_height=None):
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
    return imageio.imread(image_filepath)