import numpy as np
import joblib
from correct_images.tools.curve_fitting import curve_fitting
from correct_images.tools.joblib_tqdm import tqdm_joblib
from tqdm import tqdm, trange


def attenuation_correct(img: np.ndarray,
                        altitude: np.ndarray,
                        atn_crr_params: np.ndarray,
                        gain: np.ndarray) -> np.ndarray:
    """ apply attenuation coefficients to an input image

    Parameters
    -----------
    img : numpy.ndarray
        input image
    altitude :
        distance matrix corresponding to the image
    atn_crr_params : numpy.ndarray
        attenuation coefficients
    gain : numpy.ndarray
        gain value for the image

    Returns
    -------
    numpy.ndarray
        Corrected image
    """
    atn_crr_params = atn_crr_params.squeeze()
    den = atn_crr_params[:, :, 0] * np.exp(atn_crr_params[:, :, 1] * altitude)
    den += atn_crr_params[:, :, 2]
    img = (gain / den * img)
    return img.astype(np.float32)


def attenuation_correct_memmap(image_memmap: np.ndarray,
                               distance_memmap: np.ndarray,
                               attenuation_parameters: np.ndarray,
                               gains: np.ndarray) -> np.ndarray:
        """Apply attenuation corrections to an image memmap

        Parameters
        -----------
        image_memmap : numpy.ndarray
            input image memmap
        distance_memmap : numpy.ndarray
            input distance memmap
        attenuation_parameters : numpy.ndarray
            attenuation coefficients
        gains : numpy.ndarray
            gain values for the image


        Returns
        -------
        numpy.ndarray
            Resulting images after applying attenuation correction
        """
        print('Applying attenuation corrections to images')

        for i_img in trange(image_memmap.shape[0]):
            atn_crr_params = attenuation_parameters.squeeze()
            den = atn_crr_params[:, :, 0] * np.exp(atn_crr_params[:, :, 1] * distance_memmap[i_img, ...])
            den += atn_crr_params[:, :, 2]
            img = (gains / den * image_memmap[i_img, ...])
            image_memmap[i_img, ...] = img.astype(np.float32)
        return image_memmap


# compute gain values for each pixel for a targeted altitude using the attenuation parameters
def calculate_correction_gains(target_altitude : np.ndarray,
                               attenuation_parameters : np.ndarray
                               ) -> np.ndarray:
    """Compute correction gains for an image

    Parameters
    -----------
    target_altitude : numpy.ndarray
        target distance for which the images will be corrected
    attenuation_parameters : numpy.ndarray
        attenuation coefficients

    Returns
    -------
    numpy.ndarray
        The correction gains
    """

    attenuation_parameters = attenuation_parameters.squeeze()
    return (
        attenuation_parameters[:, :, 0]
        * np.exp(attenuation_parameters[:, :, 1] * target_altitude)
        + attenuation_parameters[:, :, 2]
    )


# calculate image attenuation parameters
def calculate_attenuation_parameters(
        images, distances, image_height, image_width
):
    """Compute attenuation parameters for all images

    Parameters
    -----------
    images : numpy.ndarray
        image memmap reshaped as a vector
    distances : numpy.ndarray
        distance memmap reshaped as a vector
    image_height : int
        height of an image
    image_width : int
        width of an image

    Returns
    -------
    numpy.ndarray
        attenuation_parameters
    """

    print("Start curve fitting...")

    with tqdm_joblib(tqdm(desc="Curve fitting", total=image_height * image_width)) as progress_bar:
        results = joblib.Parallel(n_jobs=-2, verbose=0)(
            [
                joblib.delayed(curve_fitting)(np.array(distances[:, i_pixel]),
                                              np.array(images[:, i_pixel]))
                for i_pixel in range(image_height * image_width)
            ]
        )

        attenuation_parameters = np.array(results)
        attenuation_parameters = attenuation_parameters.reshape(
            [image_height, image_width, 3]
        )
        return attenuation_parameters


