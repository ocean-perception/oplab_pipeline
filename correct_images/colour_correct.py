import os
import math
import multiprocessing
from multiprocessing import Pool
import yaml
import numpy as np
from correct_images.bayer2rgb_pixel_stats import bayer2rgb_pixel_stats
from correct_images.utilities import get_filenames, generate_filelist, validate_filenames, get_outpath, read_image

#
# Author: jw22g14@soton.ac.uk Jennifer Walker, based on original Matlab code by B.Thornton@soton.ac.uk Blair Thornton
#
# script to colour correct bayer pattern or colour images (8bit or 16bit) into
# greyworld images based on a world assumption set by <method>.
# corrected images are output in a mirrored path automatically generated by
# the input <filepath>
#
# <list>            ascii list of images to process
# <filepath>        path to the images and <list> must contain a subfolder raw.
# <filter_pattern>  bayer pattern 'grgb', 'gbrg', 'grbg', 'bggr', 'rggb'.
#                   Ignored if debayer set to false
# <method>          'grey_world': assumption of grey_world within each image. Use only when you have a small number of images
#                   'pixel_stat': assumption of grey_world for each pixel over a series of images. Corrects for vignetting.
# <target mean>     targeted mean image brightness as a percent of full range
# <target std>      targeted contrast of image as a percent of full range
# <show>            true to show images and pause after each, false to run
#                   continuously without showing the images in a popup
#                   figure
# <debayer>         true for bayer pattern images, false for 3 colour
#                   images
#



#TODO: write tests for functions, and split functions down further


def mean_per_pixel(directory, filenames, debayer, filter_pattern):
    I = read_image(os.path.join(directory, filenames[0]), debayer, filter_pattern)
    # print("Type in which images are saved: " + str(I.dtype))
    [a, b, channels] = I.shape
    Istats_mean = np.zeros([a, b, channels])
    num_of_files = len(filenames)
    # compute average intensity of each pixel in each image specified in <list>
    k = 0
    for file in filenames:
        if file != '':
            I = read_image(os.path.join(directory, file), debayer, filter_pattern)
            Istats_mean = update_mean_per_pixel(I, Istats_mean, num_of_files)
            progress = str(k+1) + ' of '+ str(num_of_files) + ' images pass 1 of 3'
            print(progress)
            k += 1
        else:
            print('EMPTY FILE')
    return Istats_mean


def update_mean_per_pixel(image, Istats_mean, num_of_files):
    [a, b, channels] = image.shape
    for i in range(a):
        for j in range(b):
            for channel in range(channels):
                Istats_mean[i][j][channel] = Istats_mean[i][j][channel]+(image[i][j][channel]/num_of_files)
    return Istats_mean


def sum_delta_mean_squared(directory, filenames, Istats_mean, debayer, filter_pattern):
    #compute standard deviation of intensity of each pixel in each image specified in <list>
    I = read_image(os.path.join(directory, filenames[0]), debayer, filter_pattern)
    [a, b, channels] = I.shape
    current_sum_delta_mean_squared = np.zeros([a, b, channels])
    num_files = len(filenames)
    k = 0
    for file in filenames:
        if file != '':
            I = read_image(os.path.join(directory, file), debayer, filter_pattern)
            current_sum_delta_mean_squared = update_delta_mean_squared(I, current_sum_delta_mean_squared, Istats_mean)
        progress = str(k+1) + ' of ' + str(num_files)+ ' images pass 2 of 3'
        print(progress)
        k += 1
    return current_sum_delta_mean_squared


def update_delta_mean_squared(image, current_sum_delta_mean_squared, Istats_mean):
    [a, b, channels] = image.shape
    for i in range(a):
        for j in range(b):
            for channel in range(channels):
                current_sum_delta_mean_squared[i,j,channel] += (image[i,j,channel]-Istats_mean[i,j,channel])**2
    return current_sum_delta_mean_squared


def pixelstat_all_images(directory, filenames, filter_pattern, Istats_mean, Istats_std, target_mean, target_std, outpath, show, debayer):
    print("Apply pixelstats on all images...")
    print("Outputting images to " + outpath)
    num_files = len(filenames)
    n_thread = multiprocessing.cpu_count()-1
    print('Number of cores being used: ', n_thread)
    arguments = []
    for n in range(num_files):
        arguments.append([directory, filenames[n], filter_pattern, Istats_mean, Istats_std, target_mean, target_std, outpath, show, debayer])
        n = n+1
    Irgb_pxstats = pixelstat_images_parallel(arguments, n_thread)
    #brightness
    print("...done applying pixelstats on all images.")


def pixelstat_images_parallel(arguments, threads):
    pool = Pool(threads)
    results = pool.starmap(bayer2rgb_pixel_stats, arguments)
    pool.close()
    pool.join()
    return results


def mean_per_pixel_parallel(arguments, threads):
    pool = Pool(threads)
    results = pool.starmap(mean_per_pixel, arguments)
    pool.close()
    pool.join()
    return results


def std_per_pixel_parallel(arguments, threads, file_count):
    try:
        pool = Pool(threads)
    except AttributeError as e:
        print("Error: ",e, "\n===============\nThis error is known to happen when running the code more than once from the same console in IPython (among others used in Spyder). Please run the code from a new console to prevent this error from happening. You may close the current console.\n==============")
        raise AttributeError(e)
    results = pool.starmap(sum_delta_mean_squared, arguments)
    pool.close()
    pool.join()

    results = np.sum(results, axis=0)
    results = np.sqrt(np.divide(results, file_count))
    return results


def save_images_to_memorymap(directory, filenames, debayer, filter_pattern):
    I = read_image(os.path.join(directory, filenames[0]), debayer, filter_pattern)
    [a, b, channels] = I.shape
    filename_images_map = os.path.realpath(os.path.join(directory, "../images.map"))
    memmap = np.memmap(filename=filename_images_map, mode='w+', shape=(len(filenames), a, b, channels), dtype=I.dtype)
    i = 0
    for imagename in filenames:
        I = read_image(os.path.join(directory,imagename), debayer, filter_pattern)
        I = np.array(I)
        memmap[i] = I
        i += 1
        if i%1000 == 0:
            print("loaded " + str(i) + " images out of " + str(len(filenames))+ " into memory map")
    print("loaded images into memory map")


def mean_per_pixel_memorymap(directory, filenames, debayer, filter_pattern):
    I = read_image(os.path.join(directory, filenames[0]), debayer, filter_pattern)
    [a, b, channels] = I.shape
    filename_images_map = os.path.realpath(os.path.join(directory, "../images.map"))
    memmap = np.memmap(filename=filename_images_map, mode='r+', shape=(len(filenames), a, b, channels), dtype=I.dtype)
    Istats_mean = np.mean(memmap, axis=0)
    return Istats_mean


def colour_correct_pixel_stat(path, filelist, target_mean, target_std, show_figures):
    [directory, do_debayer, filter_pattern] = load_mission_yaml(path)
    if filelist is None:
        filelist = generate_filelist(directory)
    filenames = get_filenames(directory, filelist)
    validate_filenames(directory, filenames)
    outpath = get_outpath(directory, 'pixel_stat', target_mean, target_std)
    save_images_to_memorymap(directory, filenames, do_debayer, filter_pattern)
    Image_mean_per_pixel = mean_per_pixel_memorymap(directory, filenames, do_debayer, filter_pattern)

    print("Compute standard deviation for all imges...")
    n_thread = multiprocessing.cpu_count()-1
    print('Number of cores being used: ', n_thread)
    arguments = []
    if len(filenames) < n_thread:
        n_thread = len(filenames)
    for n in range(n_thread):
        start = math.floor(n*len(filenames)/n_thread)
        finish = math.floor(((n+1)*len(filenames)/n_thread))
        arguments.append([directory, filenames[int(start):int(finish)], Image_mean_per_pixel, do_debayer, filter_pattern])
        n = n+1
    Image_std_per_pixel = std_per_pixel_parallel(arguments, n_thread, len(filenames))
    print("...done computing standard deviation for all imges.")
    # print(Image_std_per_pixel)
    # print("compared to original std_per_pixel")
    # Image_std_per_pixel = std_per_pixel(directory, filenames, Image_mean_per_pixel)
    # Image_std_per_pixel = np.sqrt(np.divide(Image_std_per_pixel,len(filenames)))
    # print(Image_std_per_pixel)
    # exit()
    pixelstat_all_images(directory, filenames, filter_pattern, Image_mean_per_pixel, Image_std_per_pixel, target_mean, target_std, outpath, show_figures, do_debayer)


def load_mission_yaml(path):
    print('Loading mission.yaml...')

    with open(path, 'r') as stream:
        load_data = yaml.load(stream)

    image_type = load_data['image']['type']
    if image_type == 'bayer':
        do_debayer = True
        filter_pattern = load_data['image']['filter_pattern']
    else:
        do_debayer = False
        filter_pattern = ''

    yaml_filepath = ('/').join(path.split('/')[:-1])
    directory = yaml_filepath + '/' + load_data['image']['filepath']

    print("...done loading mission.yaml")

    return directory, do_debayer, filter_pattern
