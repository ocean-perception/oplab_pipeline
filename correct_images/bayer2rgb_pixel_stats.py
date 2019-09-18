import os
import cv2
from correct_images.gamma_correct import gamma_correct
from correct_images.utilities import getBitDepth, read_image, adjust_to_bitdepth
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import numpy as np


def bayer2rgb_pixel_stats(filepath,filename,filter_pattern,current_mean,current_std,target_mean,target_std,outpath,show,debayer):

# brightness
###########################################################################
# Author: jw22g14@soton.ac.uk Jennifer Walker , based on original Matlab code by B.Thornton@soton.ac.uk Blair Thornton
#
# assumption of grey_world for each pixel over a series of images. Corrects for vignetting.
# equalises r,g,b, channels over image set and applies target brightness
# and contrast and apply gamma correction
###########################################################################

    #read in image, shape, and bitdepth
    I = read_image(filepath + filename)
    datatype=I.dtype
    [m, n, channels] = I.shape
    bitdepth = getBitDepth(I)

    # set brightness and contrast targets in term of bit.
    target_mean = adjust_to_bitdepth(target_mean, bitdepth)
    target_std = adjust_to_bitdepth(target_std, bitdepth)

    if debayer:
        I, current_mean, current_std = run_debayer(I, current_mean, current_std, filter_pattern)

    normalised_img = normalise_colour_bands(I, current_mean, current_std, target_mean, target_std)



    #apply gamma correction
    corrected_img = batch_gamma_correct(normalised_img, bitdepth)

    # Recombine separate color channels into a single, true color RGB image.

    Icor = np.array(corrected_img, dtype=datatype)
    # if saving an rgb image, rearrange to bgr for saving
    if Icor.shape[2] == 3:
        Icor = cv2.cvtColor(Icor.astype(datatype), cv2.COLOR_RGB2BGR)

#     if show==True:
#         figure(1)  # To use this add import matplotlib.pyplot as plt and use replace figure(1) by plt.figure(1). Same for imshow()
#         imshow(Irgb)
#         figure(2)
#         imshow(Istats_mean_rgb)
#         figure(3)
#         imshow(Istats_std_rgb)
#         figure(4)
#         imshow(Icor)

    # write image in outpath
    output_file_path = os.path.join(outpath, filename[0:len(filename)-4]+'_colour_corrected.tif')
    print("Writing to " + output_file_path)
    cv2.imwrite(output_file_path, Icor)


def normalise_colour_bands(rgb_image, current_mean, current_std, target_mean, target_std):
    normalised_img = np.zeros(rgb_image.shape)
    for i in range(rgb_image.shape[-1]):
        normalised_img[...,i] = normalise_colour_band(rgb_image[...,i], current_mean[...,i], target_mean, current_std[...,i], target_std)
    return normalised_img


def normalise_colour_band(band, band_mean, target_mean, band_std, target_std):
    gain = np.divide(target_mean,band_mean)
    stretch = np.divide(target_std,band_std)
    band = np.multiply((band - band_mean), stretch) + np.multiply(band, gain)
    return band


def batch_gamma_correct(image, bitdepth):
    corrected_image = np.zeros(image.shape)
    for i in range(image.shape[-1]):
        corrected_image[:,:,i] = gamma_correct(image[:,:,i],bitdepth)

    return corrected_image


def run_debayer(image, current_mean, current_std, filter_pattern):
    if image.shape[2] == 1:
        image = image[:,:,0]
        current_mean = current_mean[:,:,0]
        current_std = current_std[:,:,0]
    else:
        raise ValueError("trying to run debayer on a non-greyscale image")
    rgb_image = np.array(demosaicing_CFA_Bayer_bilinear(image, pattern=filter_pattern))
    current_mean = np.array(demosaicing_CFA_Bayer_bilinear(current_mean, pattern=filter_pattern))
    current_std = np.array(demosaicing_CFA_Bayer_bilinear(current_std, pattern=filter_pattern))
    return(rgb_image, current_mean, current_std)
