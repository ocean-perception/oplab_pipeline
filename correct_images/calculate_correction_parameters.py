import datetime
import os
import sys
import random
import socket
import uuid

import cv2
import imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import yaml
from scipy import optimize
from tqdm import trange

from correct_images.read_mission import read_params

def calculate_correction_parameters(path_raw,path_processed,path_mission,path_correct,force):
    '''

    :param path_mission: Path to 'mission.yaml'.
    :param path_correct: Path to 'correct_images.yaml'
    :return: None. Result image files and configurations are saved as files.
    '''
    # calculate parameters before demosaicing to reduce calculation
    # load configuration from yaml file
    path_r = os.path.expanduser(path_raw)
    path_p = os.path.expanduser(path_processed)
    path_m = os.path.expanduser(path_mission)
    path_c = os.path.expanduser(path_correct)
    print('loading', path_m, datetime.datetime.now())
    print('loading', path_c, datetime.datetime.now())
    
    mission = read_params(path_m,'mission') # read_params(path to file, type of file: mission/correct_config)
    config_ = read_params(path_c,'correct')
    
    #load parameters from correct_images.yaml   
    label_raw_file = 'raw file'         
    label_altitude = ' Altitude [m]'    
    altitude_max = config_.attenuation_correction.altitude_max.get('max') 
    altitude_min = config_.attenuation_correction.altitude_min.get('min') 
    calculated_atn_crr_params_path = None 
    sampling_method = config_.attenuation_correction.sampling_method 
    camera_ = mission.image.format
    print(camera_)
    
    if camera_ == 'seaxerocks_3':
        src_file_format = 'raw'
    if camera_ == 'unagi':
        src_file_format = 'tif'
        camera_lr = 'LC'
   
    '''
    else:
        src_file_format = 'tif'
    '''
    #camera_lr = 'LC' 
    dst_file_format = config_.output.dst_file_format 
    joblib_verbose = 3 
    init_atn_crr_params_path = None 
    trim_ratio = 0.2 

    # load src file or target files to data frame
    src_filelist_path = None 
    if src_filelist_path is not None:
        df_all = pd.read_csv(os.path.expanduser(src_filelist_path))
        if src_file_format == 'tif' or src_file_format == 'tif':
            # for tunasand camera, left camera (LC) or right camera (RC) should be selected
            if camera_lr == 'LC':
                df_all = df_all.query('Imagenumber.str.contains("LC")', engine='python')
            elif camera_lr == 'RC':
                df_all = df_all.query('Imagenumber.str.contains("RC")', engine='python')

    else:
        img_path = mission.image.cameras_1.get('path')
        anf = config_.config.auv_nav_path
        src_file_dirpath = path_r + '/' + img_path
        auv_nav_filepath = path_p + '/' + anf + '/csv/dead_reckoning/auv_dr_aft.csv'
        
       
        
        df_all = pd.read_csv(auv_nav_filepath, dtype={'Imagenumber': object})
        raw_file_list = [None] * len(df_all)
        for i_file in range(len(raw_file_list)):
            if src_file_format == 'raw':
                raw_file_list[i_file] = os.path.expanduser(src_file_dirpath) + '/' + df_all['Imagenumber'][i_file].zfill(7) + '.raw'
            elif src_file_format == 'tif':
                raw_file_list[i_file] = os.path.expanduser(src_file_dirpath) + '/' + df_all['Imagenumber'][i_file]
            else:
                print('src_file_format:', src_file_format, 'is incorrect.')
                return
        df_all = pd.concat([df_all, pd.DataFrame(raw_file_list, columns=[label_raw_file])], axis=1)
    src_filelist = df_all[label_raw_file]

    # for attenuation correction
    target_altitude = None # load_data.get('target_altitude', None)
    curve_fit_trial_num = 1 # load_data.get('curve_fit_trial_num', 1)
    # attenuation_correction_parameter_file_path = load_data.get('attenuation_correction_parameter_file_path', None)
    bin_band = 0.1 # load_data.get('bin_band', 0.1)  # 0.1m for AE2000
    min_sample_per_bin = 5 # load_data.get('min_sample_per_bin', 5)
    max_sample_per_bin = 100 # load_data.get('max_sample_per_bin', 100)
    median_filter_kernel_size = 1 # load_data.get('median_filter_kernel_size', 1)

    # remove too low or too high altitude file and too small file size file
    altitudes_all = df_all[label_altitude].values
    idx_effective_data = np.where((altitudes_all >= altitude_min) & (altitudes_all <= altitude_max))

    # configure output file path
    dirpath = os.path.dirname(src_filelist[0])
    #dirpath = os.path.dirname(os.path.expanduser(src_filelist[0]))
    print(dirpath)
    if 'raw' in dirpath:
        dirpath = dirpath.replace('raw', 'processed', 1)
    print(dirpath)
    
           
    if os.path.exists(dirpath) is True:
        if force is True:
            print('Attenuation correction parameters already exist.')
            print('Code will overwrite existing parameters.')
            
        else:
            print('Code will quit - correction parameters already exist.')
            print('Run correct_images with [parse] [-f] option for overwriting existing correction parameters.')
            sys.exit()
    else:
        print('code will compute correction parameters for first time.')


    dirpath = dirpath + '/attenuation_correction'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath_atn_crr = dirpath + '/tmp_atn_crr'
    # if not os.path.exists(dirpath_atn_crr):
    #     os.makedirs(dirpath_atn_crr)
    dirpath_bayer = dirpath + '/bayer'
    if not os.path.exists(dirpath_bayer):
        os.makedirs(dirpath_bayer)

    # file path of output image data
    dst_filelist = [None] * len(df_all)
    bayer_filelist = [None] * len(df_all)
    atn_crr_filelist = [None] * len(df_all)
    for i_dst_file in range(len(dst_filelist)):
        tmp_filepath = src_filelist[i_dst_file]
        if '/raw/' in tmp_filepath:
            tmp_filepath = tmp_filepath.replace('/raw/', '/processed/', 1)
        dst_file_name = os.path.basename(tmp_filepath)
        title, ext = os.path.splitext(dst_file_name)

        dst_filelist[i_dst_file] = dirpath + '/' + title + '.' + dst_file_format
        bayer_filelist[i_dst_file] = dirpath_bayer + '/' + title + '.npy'
        atn_crr_filelist[i_dst_file] = dirpath_atn_crr + '/' + title + '.npy'

    # file path of metadata
    if src_file_format == 'raw':
        dir_path_image_crr_params = os.path.expanduser(os.path.dirname(dst_filelist[0])) + '/params'
    elif src_file_format == 'tiff' or src_file_format == 'tif':
        if camera_lr == 'LC':
            dir_path_image_crr_params = os.path.expanduser(os.path.dirname(dst_filelist[0])) + '/params_LC'
        elif camera_lr == 'RC':
            dir_path_image_crr_params = os.path.expanduser(os.path.dirname(dst_filelist[0])) + '/params_RC'

    if os.path.exists(dir_path_image_crr_params):
        dir_path_image_crr_params_org = dir_path_image_crr_params
        count = 1
        while os.path.exists(dir_path_image_crr_params):
            dir_path_image_crr_params = dir_path_image_crr_params_org + '_' + str(count).zfill(3)
            count = count + 1
    if not os.path.exists(dir_path_image_crr_params):
        os.makedirs(dir_path_image_crr_params)

    if calculated_atn_crr_params_path is None:
        calculated_atn_crr_params_path = dir_path_image_crr_params + '/atn_crr_params.npy'

    file_list_raw = df_all[label_raw_file].values.tolist()

    if src_file_format == 'raw':
        # xviii camera
        a, b = 1024, 1280
        # developing .raw data to bayer data of uint32 numpy array.
        print('start loading bayer images', len(file_list_raw), 'files to', dirpath_bayer, datetime.datetime.now())
        src_file_list_not_exist = []
        bayer_file_list_not_exsit = []
        for idx_raw in range(len(file_list_raw)):
            if not os.path.exists(bayer_filelist[idx_raw]):
                src_file_list_not_exist.append(file_list_raw[idx_raw])
                bayer_file_list_not_exsit.append(bayer_filelist[idx_raw])

        print(len(file_list_raw) - len(bayer_file_list_not_exsit), 'files have already existed.')

        task_num = 100
        num_loop = int(len(bayer_file_list_not_exsit) / task_num) + 1
        start_idx = 0
        idx_total = start_idx
        end_idx = 0
        while start_idx < len(bayer_file_list_not_exsit):
            # for debug
            #     break

            print('processing load_xviii_bayer_from_binary', int(start_idx / task_num) + 1, '/', num_loop, 'of total',
                  len(bayer_file_list_not_exsit), 'files',
                  datetime.datetime.now(), flush=True)

            end_idx = start_idx + task_num
            if end_idx > len(bayer_file_list_not_exsit):
                end_idx = len(bayer_file_list_not_exsit)

            raw_img_for_size = np.fromfile(os.path.expanduser(src_file_list_not_exist[start_idx]), dtype=np.uint8)
            arg_bayer_img = np.zeros((end_idx - start_idx, raw_img_for_size.shape[0]), dtype=raw_img_for_size.dtype)
            for idx_raw in range(start_idx, end_idx):
                arg_bayer_img[idx_raw - start_idx, :] = np.fromfile(
                    os.path.expanduser(src_file_list_not_exist[idx_raw]),
                    dtype=raw_img_for_size.dtype)

            results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)([joblib.delayed(load_xviii_bayer_from_binary)(
                arg_bayer_img[idx_arg, :]) for idx_arg in
                range(end_idx - start_idx)])

            for idx_raw in range(start_idx, end_idx):
                np.save(bayer_file_list_not_exsit[idx_raw], results[idx_raw - start_idx])

            start_idx = end_idx

    elif src_file_format == 'tif' or src_file_format == 'tiff':
        # unaggi camera
        print('start loading tif images', len(file_list_raw), 'files to', dirpath_bayer, datetime.datetime.now())
        src_file_list_not_exist = []
        bayer_file_list_not_exsit = []
        for idx_raw in range(len(file_list_raw)):
            if not os.path.exists(bayer_filelist[idx_raw]):
                src_file_list_not_exist.append(file_list_raw[idx_raw])
                bayer_file_list_not_exsit.append(bayer_filelist[idx_raw])

        print(len(file_list_raw) - len(bayer_file_list_not_exsit), 'files have already existed.')

        tmp_tif_for_size = imageio.imread(file_list_raw[0])
        a = tmp_tif_for_size.shape[0]
        b = tmp_tif_for_size.shape[1]

        for i_file_not_exist in range(len(src_file_list_not_exist)):
            tmp_tif = imageio.imread(src_file_list_not_exist[i_file_not_exist])
            tmp_npy = np.zeros([a, b], np.uint16)
            tmp_npy[:, :] = np.array(tmp_tif, np.uint16)
            np.save(bayer_file_list_not_exsit[i_file_not_exist], tmp_npy)

    # caluculate attenuation correction parameter
    if target_altitude is None:
        target_altitude = float(np.mean(altitudes_all[idx_effective_data]))

    # memmap is created at local directory
    file_name_memmap_raw, memmap_raw = load_memmap_from_npy_filelist(bayer_filelist)
    # TODO for debug. read existing file.
    # file_name_memmap_raw = '/home/ty1u18/PycharmProjects/correct_images/memmap_raw_img_d953693d-47aa-403d-9ece-f8f3b19d8b98.map'
    # memmap_raw = np.memmap(file_name_memmap_raw, np.float32, 'r', shape=(8619, 2056, 2464))

    print('start calculate mean and std of raw img', datetime.datetime.now())

    img_mean_raw, img_std_raw = \
        calc_img_mean_and_std_trimmed(memmap_raw, trim_ratio,
                                      calc_std=True,
                                      effective_index=idx_effective_data)
    # TODO for debug. not calculate raw
    # img_mean_raw = 0
    # img_std_raw = 0

    dirpath_img_mean_raw = dir_path_image_crr_params + '/bayer_img_mean_raw'
    dirpath_img_std_raw = dir_path_image_crr_params + '/bayer_img_std_raw'
    np.save(dirpath_img_mean_raw, img_mean_raw)
    np.save(dirpath_img_std_raw, img_std_raw)

    list_dirpath = [dirpath_img_mean_raw, dirpath_img_std_raw]
    list_img = [img_mean_raw, img_std_raw]
    for i_img in range(len(list_img)):
        save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

    # calculate regression parameters for all pixels and channels
    print('start attenuation correction parameter calculation.')

    # 3 is number of parameter in exp_curve other than x
    atn_crr_params = np.zeros([a, b, 3])
    atn_crr_params = atn_crr_params.reshape([a * b, 3])

    hist_bounds = np.arange(altitude_min, altitude_max, bin_band)
    idxs = np.digitize(altitudes_all, hist_bounds)
    altitudes_ret = []
    each_bin_image_list = []
    tmp_altitude_sample = 0.0
    message = 'start calculating histogram ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx_bin in trange(1, hist_bounds.size, ascii=True, desc=message):
        tmp_altitudes = altitudes_all[np.where(idxs == idx_bin)]
        if len(tmp_altitudes) > min_sample_per_bin:
            # calculate sample image in this bin
            tmp_idx = np.where(idxs == idx_bin)[0]
            if len(tmp_idx) > max_sample_per_bin:
                tmp_idx = random.sample(list(tmp_idx), max_sample_per_bin)
                tmp_altitudes = altitudes_all[tmp_idx]

            tmp_bin_imgs = memmap_raw[tmp_idx]
            # calculate sample image of current bin
            tmp_bin_img_sample = np.zeros((a, b), np.float32)

            if sampling_method == 'mean':
                tmp_bin_img_sample = np.mean(tmp_bin_imgs, axis=0)
                tmp_altitude_sample = np.mean(tmp_altitudes)

            elif sampling_method == 'median':
                tmp_bin_img_sample = np.median(tmp_bin_imgs, axis=0)
                tmp_altitude_sample = np.mean(
                    tmp_altitudes)
                # altitude value is calculated as mean because it has less varieance.

            elif sampling_method == 'mean_trimmed':
                #     TOOD implement trimmed mean and std
                tmp_bin_img_sample, dummy = calc_img_mean_and_std_trimmed(tmp_bin_imgs, trim_ratio, calc_std=False,
                                                                          effective_index=-1)
                tmp_altitude_sample = np.mean(tmp_altitudes)

            del tmp_bin_imgs

            each_bin_image_list.append(tmp_bin_img_sample)
            altitudes_ret.append(tmp_altitude_sample)

    imgs_for_calc_atn = np.array(each_bin_image_list)
    # TODO for debug
    # print('len of each_bin_image_list', len(each_bin_image_list))
    # print('shape of images for calc atn', imgs_for_calc_atn.shape)

    imgs_for_calc_atn = imgs_for_calc_atn.reshape([len(each_bin_image_list), a * b])
    altitudes_for_calc_atn = altitudes_ret

    print('start curve fitting', datetime.datetime.now())
    if init_atn_crr_params_path is not None:
        initial_atn_crr_params = np.load(init_atn_crr_params_path)
        initial_atn_crr_params = initial_atn_crr_params.reshape([a * b, 3])

        # all pixels
        results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
            [joblib.delayed(optim_exp_curve_param_with_init)(
                altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel], initial_atn_crr_params[i_pixel, :]) for i_pixel
                in
                range(a * b)])
        atn_crr_params = np.array(results)

        # TODO for debug only 100 pixels
        # results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
        #     [joblib.delayed(optim_exp_curve_param_with_init)(
        #         altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel], initial_atn_crr_params[i_pixel, :]) for i_pixel
        #         in
        #         range(100)])
        # atn_crr_params = np.zeros([a * b, 3])
        # atn_crr_params[0:100, :] = np.array(results)

    else:
        #     auto initialisation
        results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
            [joblib.delayed(optim_exp_curve_param_auto_init)(
                altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel], curve_fit_trial_num) for i_pixel in
                range(a * b)])
        atn_crr_params = np.array(results)

        # TODO for debug only 100 pixels
        # results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
        #     [joblib.delayed(optim_exp_curve_param_auto_init)(
        #         altitudes_for_calc_atn, imgs_for_calc_atn[:,i_pixel], curve_fit_trial_num) for i_pixel in
        #         range(100)])
        # atn_crr_params=np.zeros([a*b,3])
        # atn_crr_params[0:100,:]=np.array(results)

    atn_crr_params = atn_crr_params.reshape([a, b, 3])

    if calculated_atn_crr_params_path is None:
        calculated_atn_crr_params_path = dir_path_image_crr_params + '/atn_crr_params.npy'

    # visualise attenuation parameters
    outpath = calculated_atn_crr_params_path[:(len(calculated_atn_crr_params_path) - 4)]
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    np.save(calculated_atn_crr_params_path, atn_crr_params)
    print('atn_crr_params has been saved to', calculated_atn_crr_params_path, datetime.datetime.now())

    save_atn_crr_params_png(outpath, atn_crr_params)

    # apply median filter to attenuation parameter
    #if median_filter_kernel_size != 1:
        #atn_crr_params = filter_atn_parm_median(atn_crr_params, median_filter_kernel_size)

    # apply attenuation correction parameters to raw images in memmap
    gain = calc_attenuation_correction_gain(target_altitude, atn_crr_params)
    message = 'applying attenuation correction to raw ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i_img in trange(memmap_raw.shape[0], ascii=True, desc=message):
        # memmap data can not be updated in joblib .
        memmap_raw[i_img, ...] = apply_atn_crr_2_img(memmap_raw[i_img, ...], altitudes_all[i_img], atn_crr_params, gain)

    print('start calculating mean and std of attenuation corrected images', datetime.datetime.now(), flush=True)
    img_mean_atn_crr, img_std_atn_crr = calc_img_mean_and_std_trimmed(memmap_raw, trim_ratio, calc_std=True,
                                                                      effective_index=idx_effective_data)

    # dirpath_img_mean_raw = dir_path_image_crr_params + '/bayer_img_mean_raw'
    # dirpath_img_std_raw = dir_path_image_crr_params + '/bayer_img_std_raw'
    dirpath_img_mean_atn_crr = dir_path_image_crr_params + '/bayer_img_mean_atn_crr'
    dirpath_img_std_atn_crr = dir_path_image_crr_params + '/bayer_img_std_atn_crr'
    # np.save(dirpath_img_mean_raw, img_mean_raw)
    # np.save(dirpath_img_std_raw, img_std_raw)
    np.save(dirpath_img_mean_atn_crr, img_mean_atn_crr)
    np.save(dirpath_img_std_atn_crr, img_std_atn_crr)

    # visualize mean and std images
    list_dirpath = [dirpath_img_mean_atn_crr, dirpath_img_std_atn_crr]
    list_img = [img_mean_atn_crr, img_std_atn_crr]
    for i_img in range(len(list_img)):
        save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

    # convert bayer_file_path from absolute path to relative path
    for i_bayer_file in range(len(bayer_filelist)):
        bayer_filelist[i_bayer_file] = bayer_filelist[i_bayer_file].replace(dirpath, '..')

    # save file list includes altitude and filepath of bayer image
    file_list_name = dir_path_image_crr_params + '/filelist.csv'
    df_all = pd.concat([df_all, pd.DataFrame(bayer_filelist, columns=['bayer file'])], axis=1)
    df_all.to_csv(file_list_name)

    dict_cfg = {
        'src_filelist_path': src_filelist_path,
        'label_raw_file': label_raw_file,
        'label_altitude': label_altitude,
        'altitude_min': altitude_min,
        'altitude_max': altitude_max,
        'calculated_atn_crr_params_path': calculated_atn_crr_params_path,
        'median_filter_kernel_size': median_filter_kernel_size,
        'sampling_method': sampling_method,
        'dst_file_format': dst_file_format,
        # 'apply_attenuation_correction': apply_attenuation_correction,
        'target_altitude': target_altitude,
        'curve_fit_trial_num': curve_fit_trial_num,
        # 'target_mean': target_mean,
        # 'target_std': target_std,
        # 'apply_gamma_correction': apply_gamma_correction,
        # 'apply_distortion_correction': apply_distortion_correction,
        # 'camera_parameter_file_path': camera_parameter_file_path,
        'src_file_format': src_file_format,
        # 'attenuation_correction_parameter_file_path': attenuation_correction_parameter_file_path,
        'bin_band': bin_band,
        'min_sample_per_bin': min_sample_per_bin,
        'max_sample_per_bin': max_sample_per_bin
    }

    cfg_filepath = dir_path_image_crr_params + '/config.yaml'
    with open(cfg_filepath, 'w') as cfg_file:
        yaml.dump(dict_cfg, cfg_file, default_flow_style=False)
    print('Done. Configurations are saved to ', cfg_filepath, datetime.datetime.now())

    del memmap_raw
    os.remove(file_name_memmap_raw)


def load_xviii_bayer_from_binary(xviii_binary_data):
    """
    Load bayer data of Xviii camera image from raw binary data.
    :param xviii_binary_data: raw binary of xviii image. Should be loaded by 'np.fromfile('path_of_xviii_raw_data(.raw)')'
    :return: bayer data of xviii image
    """
    img_h = 1024
    img_w = 1280
    bayer_img = np.zeros((img_h, img_w), dtype=np.uint32)

    # https://github.com/ocean-perception/image_conversion/blob/master/src/xviii_demosaic.cpp
    # read raw data and put them into bayer patttern.
    count = 0
    for i in range(0, img_h, 1):
        for j in range(0, img_w, 4):
            work = xviii_binary_data[count:count + 12]
            bayer_img[i, j] = (work[3] & 0xff) << 16 | ((work[2] & 0xff) << 8) | (work[1] & 0xff)
            bayer_img[i, j + 1] = ((work[0] & 0xff) << 16) | ((work[7] & 0xff) << 8) | (work[6] & 0xff)
            bayer_img[i, j + 2] = ((work[5] & 0xff) << 16) | ((work[4] & 0xff) << 8) | (work[11] & 0xff)
            bayer_img[i, j + 3] = ((work[10] & 0xff) << 16) | ((work[9] & 0xff) << 8) | (work[8] & 0xff)
            count += 12

    return bayer_img


def calc_img_mean_and_std_trimmed(src_imgs, ratio_trimming, calc_std=True, effective_index=-1):
    """
    calc trimmed mean and standard deviation images of input images.
    The rate of tirimming should be designated
    :param src_imgs: numpy array of source images. The size should be [number_of_source_images, height fo each image, width of each image, channel of each image]
    :param ratio_trimming: The ratio of trimming for outlier removal. If 0.2 is given, the function remove the pixel values in the range lower than 10% and higher than 90%.
    :param calc_std: if False, this function calculate only mean value.
    :return: [ret_mean, ret_std]
    """

    [n, a, b] = src_imgs.shape
    ret_mean = np.zeros((a, b), np.float32)
    ret_std = np.zeros((a, b), np.float32)

    if effective_index == -1:
        effective_index = [list(range(0, n))]

    message = 'calculating trimmed mean and std of images ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx_a in trange(a, ascii=True, desc=message):
        results = joblib.Parallel(n_jobs=-2, verbose=0)([joblib.delayed(calc_mean_and_std_trimmed)(
            src_imgs[effective_index, idx_a, idx_b][0], ratio_trimming, calc_std) for idx_b in
            range(b)])
        ret_mean[idx_a, :] = np.array(results)[:, 0]
        ret_std[idx_a, :] = np.array(results)[:, 1]

    return ret_mean, ret_std


def calc_mean_and_std_trimmed(src_values, rate_trimming, calc_std=True):
    sorted_values = np.sort(src_values)
    idx_left_limit = int(len(src_values) * rate_trimming / 2.0)
    idx_right_limit = int(len(src_values) * (1.0 - rate_trimming / 2.0))

    mean = np.mean(sorted_values[idx_left_limit:idx_right_limit])
    std = 0

    if calc_std:
        std = np.std(sorted_values[idx_left_limit:idx_right_limit])
    return np.array([mean, std])


def save_atn_crr_params_png(dst_dirpath, atn_crr_params):
    """
    save the visualised image of attenuation correction parameters. 4 (number of channels in bayer images) * 3 (number of parameters in each attenuation curve) = 12 png images are saved.
    :param dst_dirpath: png images are saved in this path
    :param atn_crr_params: The attenuation correction parameters for visualisation.
    :return: None
    """
    # visualise attenuation parameters
    params_g1 = atn_crr_params[0::2, 0::2, :]
    params_r = atn_crr_params[0::2, 1::2, :]
    params_b = atn_crr_params[1::2, 0::2, :]
    params_g2 = atn_crr_params[1::2, 1::2, :]

    result_table_vis = np.zeros((4, params_g1.shape[0], params_g2.shape[1], 3))
    result_table_vis[0:1, :, :, :] = params_g1
    result_table_vis[1:2, :, :, :] = params_r
    result_table_vis[2:3, :, :, :] = params_b
    result_table_vis[3:4, :, :, :] = params_g2
    for i_ch in range(4):
        for i_param in range(3):
            cax = plt.matshow(result_table_vis[i_ch, :, :, i_param])
            plt.colorbar(cax)
            filename = 'ch_' + str(i_ch) + '_param_' + str(i_param) + '.png'
            plt.title(filename[:len(filename) - 4])
            plt.savefig(os.path.join(dst_dirpath, filename))
            plt.close()


def save_bayer_array_png(dst_dirpath, bayer_img_array):
    """
    save the visualised image of bayer data array.
    :param dst_dirpath: png images are saved in this path
    :param bayer_img_array: the bayer image
    :return:
    """
    # visualise attenuation parameters
    params_g1 = bayer_img_array[0::2, 0::2, ...]
    params_r = bayer_img_array[0::2, 1::2, ...]
    params_b = bayer_img_array[1::2, 0::2, ...]
    params_g2 = bayer_img_array[1::2, 1::2, ...]

    result_table_vis = np.zeros((params_g1.shape[0], params_g2.shape[1], 4))
    result_table_vis[:, :, 0] = params_g1[:, :, ...]
    result_table_vis[:, :, 1] = params_r[:, :, ...]
    result_table_vis[:, :, 2] = params_b[:, :, ...]
    result_table_vis[:, :, 3] = params_g2[:, :, ...]

    if not os.path.exists(dst_dirpath):
        os.makedirs(dst_dirpath)

    for i_ch in range(4):
        cax = plt.matshow(result_table_vis[:, :, i_ch, ...])
        plt.colorbar(cax)
        plt.title('ch_' + str(i_ch))
        plt.savefig(dst_dirpath + '/ch_' + str(i_ch) + '.png')
        plt.close()


def filter_atn_parm_median(src_atn_param, kernel_size):
    params_g1 = src_atn_param[0::2, 0::2, :, :]
    params_r = src_atn_param[0::2, 1::2, :, :]
    params_b = src_atn_param[1::2, 0::2, :, :]
    params_g2 = src_atn_param[1::2, 1::2, :, :]

    list_params = [params_g1, params_r, params_b, params_g2]
    params_g1_fil = np.zeros(params_g1.shape, params_g1.dtype)
    params_r_fil = np.zeros(params_g1.shape, params_g1.dtype)
    params_b_fil = np.zeros(params_g1.shape, params_g1.dtype)
    params_g2_fil = np.zeros(params_g1.shape, params_g1.dtype)

    list_params_fil = [params_g1_fil, params_r_fil, params_b_fil, params_g2_fil]

    params = np.zeros((params_g1.shape[0], params_g1.shape[1], 4, 3))

    for i_mos in range(len(list_params)):
        tmp_params_fil = np.zeros(list_params[i_mos].shape, list_params[i_mos].dtype)
        for i in range(np.size(list_params[i_mos], axis=2)):
            for j in range(np.size(list_params[i_mos], axis=3)):
                list_params_fil[i_mos][:, :, i, j] = filters.median_filter(list_params[i_mos][:, :, i, j],
                                                                           (kernel_size, kernel_size))

    ret = np.zeros(src_atn_param.shape, src_atn_param.dtype)
    ret[0::2, 0::2, :, :] = params_g1_fil
    ret[0::2, 1::2, :, :] = params_r_fil
    ret[1::2, 0::2, :, :] = params_b_fil
    ret[1::2, 1::2, :, :] = params_g2_fil

    return ret


def load_memmap_from_npy_filelist(list_raw_files):
    filename_images_map = 'memmap_raw_img_' + str(uuid.uuid4()) + '.map'

    I = np.load(os.path.expanduser(list_raw_files[0]))
    list_shape = [len(list_raw_files)]
    list_shape = list_shape + list(I.shape)
    # memmap = np.memmap(filename=filename_images_map, mode='w+', shape=tuple(list_shape),dtype=I.dtype)

    if 'red' in socket.gethostname():
        #     if executed on HPC, the raw data are loaded on RAM
        try:
            memmap = np.zeros(shape=tuple(list_shape), dtype=np.float32)
            dummy_memmap = np.memmap(filename=filename_images_map, mode='w+', shape=(1, 1),
                                     dtype=np.float32)
            message = 'loading raw data to RAM ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except MemoryError:
            # use memmap instead of RAM when MemoryError is caught.
            memmap = np.memmap(filename=filename_images_map, mode='w+', shape=tuple(list_shape),
                               dtype=np.float32)
            message = 'loading raw data to memmap ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    else:
        memmap = np.memmap(filename=filename_images_map, mode='w+', shape=tuple(list_shape),
                           dtype=np.float32)  # loaded as float32 because this memmap used for restoring attenuation corrected data later.
        message = 'loading raw data to memmap ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    memmap[0, ...] = I.astype(np.float32)

    for i_file in trange(1, len(list_raw_files), ascii=True, desc=message):
        memmap[i_file, ...] = np.load(list_raw_files[i_file])

    return filename_images_map, memmap


def exp_curve(x, a, b, c):
    return a * np.exp(b * x) + c


def residual_exp_curve(params, x, y):
    residual = exp_curve(x, params[0], params[1], params[2]) - y
    return residual


def optim_exp_curve_param_with_init(altitudes, intensities, init_params):
    loss = 'soft_l1'
    # loss='linear'
    method = 'trf'
    # method='lm'
    altitudes = np.array(altitudes)
    intensities = np.array(intensities)
    init_params = np.array(init_params)
    tmp_params = optimize.least_squares(residual_exp_curve, init_params, loss=loss,
                                        method=method, args=(altitudes, intensities),
                                        bounds=([1, -np.inf, 0], [np.inf, 0, np.inf]))
    return tmp_params.x


def optim_exp_curve_param_auto_init(altitudes, intensities, num_of_trials=1):
    loss = 'soft_l1'
    # loss='linear'
    method = 'trf'
    # method='lm'
    bound_lower = [1, -np.inf, 0]
    bound_upper = [np.inf, 0, np.inf]

    altitudes = np.array(altitudes)
    intensities = np.array(intensities)

    flag_already_calculated = False
    min_cost = float('inf')
    for i_calc in range(num_of_trials):
        c = 0
        if i_calc == 0:
            idx_0 = int(len(intensities) * 0.3)
            idx_1 = int(len(intensities) * 0.7)
        else:
            [idx_0, idx_1] = random.sample(range(len(intensities)), k=2)

        b = (np.log((intensities[idx_0] - c) / (intensities[idx_1] - c))) / (altitudes[idx_0] - altitudes[idx_1])
        a = (intensities[idx_1] - c) / np.exp(b * altitudes[idx_1])
        if a < 1 or b > 0 or np.isnan(a) or np.isnan(b):
            if flag_already_calculated == True:
                continue
            else:
                flag_already_calculated = True
            a = 1.01
            b = -0.01

        init_params = np.array([a, b, c])
        # print(init_params)
        # tmp_params=None
        try:
            tmp_params = optimize.least_squares(residual_exp_curve, init_params, loss=loss,
                                                method=method, args=(altitudes, intensities),
                                                bounds=(bound_lower, bound_upper))
            if tmp_params.cost < min_cost:
                min_cost = tmp_params.cost
                ret_params = tmp_params.x

        except (ValueError, UnboundLocalError) as e:
            print('Value Error', a, b, c)

    return ret_params


def jacob_exp_curve(params, x, y):
    # prepared as Jacobian for optim_exp_curbe_param_with_init
    # It is not used currently, because numerical method (default) is faster in this case..
    da = -np.exp(params[1] * x)
    db = -params[0] * x * np.exp(params[1] * x)
    dc = -np.ones(db.shape)
    ret = np.array([da, db, dc]).transpose()
    return ret


def calc_attenuation_correction_gain(target_altitude, atn_crr_params):
    atn_crr_params = atn_crr_params.squeeze()
    return atn_crr_params[:, :, 0] * np.exp(
        atn_crr_params[:, :, 1] * target_altitude) + atn_crr_params[:, :, 2]


def correct_attenuation(src_img_file_path, dst_img_file_path, src_altitude, gain, table_correct_params):
    src_img = np.array(imageio.imread(os.path.expanduser(src_img_file_path)))

    dst_img = gain / (
            table_correct_params[:, :, :, 0] * np.exp(table_correct_params[:, :, :, 1] * src_altitude)
            + table_correct_params[:, :, :, 2]) * src_img
    dst_img = dst_img.astype(np.uint8)
    imageio.imwrite(dst_img_file_path, dst_img)


def correct_attenuation_from_raw(mosaic_mat, src_altitude, gain, table_correct_params):
    return ((gain / (
            table_correct_params[:, :, :, 0] * np.exp(table_correct_params[:, :, :, 1] * src_altitude)
            + table_correct_params[:, :, :, 2])) * mosaic_mat).astype(np.float32)


def correct_distortion(src_img, map_x, map_y):
    src_img = np.clip(src_img, 0, 255).astype(np.uint8)
    dst_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR)
    return dst_img


def apply_atn_crr_2_img(img, altitude, atn_crr_params, gain):
    atn_crr_params = atn_crr_params.squeeze()
    img = ((gain / (
            atn_crr_params[:, :, 0] * np.exp(atn_crr_params[:, :, 1] * altitude)
            + atn_crr_params[:, :, 2])) * img).astype(np.float32)
    return img
