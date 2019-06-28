import datetime
import os
import sys

import cv2
import imageio
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import yaml
from tqdm import tqdm

from auv_nav.tools.console import Console
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from correct_images.calculate_correction_parameters import calc_attenuation_correction_gain, apply_atn_crr_2_img
from correct_images.read_mission import read_params


def develop_corrected_image(path, force):
    '''

    :param path_mission: Path to 'mission.yaml'.
    :param path_correct: Path to 'correct_images.yaml'
    :return: None. Result image files and configurations are saved as files.
    '''

    path_correct = get_config_folder(path) / "correct_images.yaml"
    if not path_correct.exists():
        Console.warn('Config File does not exist. Did you parse first this dive?')
        Console.quit('run correct_images parse first.')
    path_mission = get_raw_folder(path) / "mission.yaml"
    path_processed = get_processed_folder(path)

    # load configuration from mission.yaml, correct_images.yaml files
    Console.info('Lading', path_mission, datetime.datetime.now())
    Console.info('Lading', path_correct, datetime.datetime.now())

    mission = read_params(path_mission, 'mission')  # read_params(path to file, type of file: mission/correct_config)
    config_ = read_params(path_correct, 'correct')

    # load src_file_dirpath from correct_images.yaml
    # sfd = config_.config.path_0.get('src_file_dirpath') # load_data.get('src_file_dirpath', None)
    img_path = mission.image.cameras_1.get('path')
    # params_dir_path = './processed/' + sfd + '/' + img_path + '/attenuation_correction/params'

    params_dir_path = path_processed / img_path / 'attenuation_correction/params'

    # load filelist
    filelist_path = params_dir_path / 'filelist.csv'
    if filelist_path.exists():
        df_filelist = pd.read_csv(filelist_path)
    else:
        Console.warn('Code will quit now - filelist.csv not found in target folder.')
        Console.warn('Run correct_images [parse] before [process].')
        Console.warn(filelist_path)
        sys.exit()

    # load config.yaml
    path_config = params_dir_path / 'config.yaml'
    with path_config.open('r') as stream:
        load_data_config = yaml.safe_load(stream)

    # load from correct_images.yaml
    target_mean = config_.normalization.target_mean # load_data.get('target_mean', 30)
    target_std = config_.normalization.target_std # load_data.get('target_std', 5)
    src_img_index = config_.config.src_img_index # load_data.get('src_img_index', -1)
    apply_attenuation_correction = config_.flags.apply_attenuation_correction # load_data.get('apply_attenuation_correction', True)
    apply_gamma_correction = config_.flags.apply_gamma_correction # load_data.get('apply_gamma_correction', True)
    apply_distortion_correction = config_.flags.apply_distortion_correction # load_data.get('apply_distortion_correction', True)
    camera_parameter_file_path = config_.flags.camera_parameter_file_path # load_data.get('camera_parameter_file_path', None)
    dst_dir_path = None # load_data.get('dst_dir_path', None)
    dst_img_format = config_.output.dst_file_format # load_data.get('dst_img_format', 'png')
    median_filter_kernel_size = config_.attenuation_correction.median_filter_kernel_size # load_data.get('median_filter_kernel_size', 1)
    debayer_option = config_.normalization.debayer_option # load_data.get('debayer_option', 'linear')

    # load from config.yaml
    label_altitude = load_data_config['label_altitude']
    target_altitude = load_data_config['target_altitude']
    src_file_format = load_data_config['src_file_format']
    label_raw_file = load_data_config['label_raw_file']

    # load .npy files
    pdp = str(params_dir_path)
    atn_crr_params = np.load(pdp + '/atn_crr_params.npy')
    bayer_img_mean = np.load(pdp + '/bayer_img_mean_raw.npy')
    bayer_img_std = np.load(pdp + '/bayer_img_std_raw.npy')
    bayer_img_corrected_mean = np.load(pdp + '/bayer_img_mean_atn_crr.npy')
    bayer_img_corrected_std = np.load(pdp + '/bayer_img_std_atn_crr.npy')

    # load values from file list
    list_altitude = df_filelist[label_altitude].values
    list_bayer_file = df_filelist['bayer file'].values
    # convert from relative path to real path
    for i_bayer_file in range(len(list_bayer_file)):
        list_bayer_file[i_bayer_file] = params_dir_path / list_bayer_file[i_bayer_file]

    # get image size
    bayer_sample = np.load(str(list_bayer_file[0]))
    a = bayer_sample.shape[0]
    b = bayer_sample.shape[1]

    # identify debayer params
    # for opencv
    if src_file_format == 'raw':
        bayer_pattern = 'GRBG'
        if debayer_option == 'linear':
            code = cv2.COLOR_BAYER_GR2BGR
        elif debayer_option == 'ea':
            code = cv2.COLOR_BAYER_GR2BGR_EA
        elif debayer_option == 'vng':
            code = cv2.COLOR_BAYER_GR2BGR_VNG

    elif src_file_format == 'tif' or src_file_format == 'tiff':
        bayer_pattern = 'RGGB'
        if debayer_option == 'linear':
            code = cv2.COLOR_BAYER_RG2BGR
        elif debayer_option == 'ea':
            code = cv2.COLOR_BAYER_RG2BGR_EA
        elif debayer_option == 'vng':
            code = cv2.COLOR_BAYER_RG2BGR_VNG

    # calculate distortion correction paramters
    if apply_distortion_correction:
        if camera_parameter_file_path is None:
            if src_file_format == 'raw':
                #         load AE2000 camera param
                camera_parameter_file_path = 'camera_params/camera_parameters_20171221_200233_xviii51707923_1_of_100.yml'
            elif src_file_format == 'tif' or src_file_format == 'tiff':
                camera_parameter_file_path = 'camera_params/camera_parameters_unagi6k.yml'
        map_x, map_y = calc_distortion_mapping(camera_parameter_file_path, a, b)

    # if developing target are not designated, develop all files in filelist.csv
    if src_img_index == -1:
        src_img_index = range(len(df_filelist))

    # determine destination path
    if dst_dir_path is None:
        # new directory is created in the same directory of 'params_dir_path'
        dst_dir_path = params_dir_path / 'developed'

    if not dst_dir_path.exists():
        dst_dir_path.mkdir(parents=True)

    message = 'developing images ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    list_dst_name = []
    for i_img in tqdm(src_img_index, ascii=True, desc=message):
        # attenuation correction or only pixel stat
        if apply_attenuation_correction:
            corrected_bayer_img = attenuation_correction_bayer(np.load(str(list_bayer_file[i_img])),
                                                               bayer_img_corrected_mean, bayer_img_corrected_std,
                                                               target_mean, target_std, atn_crr_params,
                                                               list_altitude[i_img], target_altitude, True, 8)

        else:
            corrected_bayer_img = pixel_stat_bayer(np.load(str(list_bayer_file[i_img])), bayer_img_mean, bayer_img_std,
                                                   target_mean, target_std, 8)

        # debayer image
        # demosaicing
        # corrected_rgb_img = demosaicing_CFA_Bayer_bilinear(corrected_bayer_img, bayer_pattern)
        # opencv
        corrected_rgb_img = cv2.cvtColor(corrected_bayer_img.astype(np.uint8), code)

        if apply_distortion_correction:
            corrected_rgb_img = correct_distortion(corrected_rgb_img, map_x, map_y, 8)

        if apply_gamma_correction:
            corrected_rgb_img = gamma_correct(corrected_rgb_img, 8)

        corrected_rgb_img = corrected_rgb_img.astype(np.uint8)

        dst_path = dst_dir_path / list_bayer_file[i_img].with_suffix('.' + dst_img_format)
        imageio.imwrite(dst_path, corrected_rgb_img)
        list_dst_name.append(dst_path.name)

    df_dst_filelist = df_filelist.iloc[src_img_index].copy()
    df_dst_filelist['image file name'] = list_dst_name
    dst_filelist_path = dst_dir_path / 'filelist.csv'
    df_dst_filelist.to_csv(dst_filelist_path)

    dict_cfg = {
        'target_mean': target_mean,
        'target_std': target_std,
        'src_img_index': src_img_index,
        'apply_attenuation_correction': apply_attenuation_correction,
        'apply_gamma_correction': apply_gamma_correction,
        'apply_distortion_correction': apply_distortion_correction,
        'camera_parameter_file_path': camera_parameter_file_path,
        'dst_dir_path': dst_dir_path,
        'dst_img_format': dst_img_format,
        'median_filter_kernel_size': median_filter_kernel_size,
        'params_dir_path': params_dir_path,
        'debayer_option': debayer_option
    }
    cfg_filepath = dst_dir_path / 'config.yaml'
    with cfg_filepath.open('w') as cfg_file:
        yaml.dump(dict_cfg, cfg_file, default_flow_style=False)
    Console.info('Done. Configurations are saved to ', cfg_filepath, datetime.datetime.now())


def filter_atn_parm_median(src_atn_param, kernel_size):
    #  0 1
    #  2 3
    params_0 = src_atn_param[0::2, 0::2, :, :]
    params_1 = src_atn_param[0::2, 1::2, :, :]
    params_2 = src_atn_param[1::2, 0::2, :, :]
    params_3 = src_atn_param[1::2, 1::2, :, :]

    list_params = [params_0, params_1, params_2, params_3]
    params_0_fil = np.zeros(params_0.shape, params_0.dtype)
    params_1_fil = np.zeros(params_0.shape, params_0.dtype)
    params_2_fil = np.zeros(params_0.shape, params_0.dtype)
    params_3_fil = np.zeros(params_0.shape, params_0.dtype)

    list_params_fil = [params_0_fil, params_1_fil, params_2_fil, params_3_fil]

    for i_mos in range(len(list_params)):
        for i in range(np.size(list_params[i_mos], axis=2)):
            for j in range(np.size(list_params[i_mos], axis=3)):
                list_params_fil[i_mos][:, :, i, j] = filters.median_filter(list_params[i_mos][:, :, i, j],
                                                                           (kernel_size, kernel_size))

    ret = np.zeros(src_atn_param.shape, src_atn_param.dtype)
    ret[0::2, 0::2, :, :] = params_0_fil
    ret[0::2, 1::2, :, :] = params_1_fil
    ret[1::2, 0::2, :, :] = params_2_fil
    ret[1::2, 1::2, :, :] = params_3_fil

    return ret


def calc_distortion_mapping(camera_parameter_file_path, a, b):
    fs = cv2.FileStorage(camera_parameter_file_path, cv2.FILE_STORAGE_READ)
    fn = fs.getNode('camera_matrix')
    camera_matrix = fn.mat()
    fn = fs.getNode('distortion_coefficients')
    distortion_coefficients = fn.mat()
    cam_mat, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (b, a), 0)
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, cam_mat, (b, a), 5)
    return map_x, map_y


def attenuation_correction_bayer(bayer_img, bayer_img_mean, bayer_img_std, target_mean, target_std, atn_crr_params,
                                 src_altitude, target_altitude, apply_attenuation_correction=True, dst_bit=8):
    gain = calc_attenuation_correction_gain(target_altitude, atn_crr_params)
    ret = apply_atn_crr_2_img(bayer_img, src_altitude, atn_crr_params, gain)
    ret = pixel_stat_bayer(ret, bayer_img_mean, bayer_img_std, target_mean, target_std, dst_bit)
    ret = np.clip(ret, 0, 2 ** dst_bit - 1)
    return ret


def pixel_stat_bayer(bayer_img, bayer_img_mean, bayer_img_std, target_mean, target_std, dst_bit=8):
    # target_mean and target std should be given in 0 - 100 scale
    target_mean_in_bitdeph = target_mean / 100.0 * (2.0 ** dst_bit - 1.0)
    target_std_in_bitdeph = target_std / 100.0 * (2.0 ** dst_bit - 1.0)
    ret = (bayer_img - bayer_img_mean) / bayer_img_std * target_std_in_bitdeph + target_mean_in_bitdeph
    ret = np.clip(ret, 0, 2 ** dst_bit - 1)
    return ret


def correct_distortion(src_img, map_x, map_y, dst_bit=8):
    src_img = np.clip(src_img, 0, 2 ** dst_bit - 1)
    dst_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR)
    return dst_img


def gamma_correct(colour, bitdepth):
    # translated to python by Jenny Walker jw22g14@soton.ac.uk

    # MATLAB original code:
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Author: B.Thornton@soton.ac.uk
    # %
    # % gamma correction to account for non-linear perception of intensity by
    # % humans
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    colour = np.divide(colour, ((2 ** bitdepth - 1)))

    if all(i < 0.0031308 for i in colour.flatten()):
        colour = 12.92 * colour
    else:
        colour = 1.055 * np.power(colour, (1 / 1.5)) - 0.055

    colour = np.multiply(np.array(colour), np.array(2 ** bitdepth - 1))

    # added by takaki
    colour = np.clip(colour, 0, 2 ** bitdepth - 1)

    return colour
