import datetime
import sys
import shutil

import cv2
import imageio
import joblib
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import yaml
from tqdm import tqdm
from pathlib import Path

from auv_nav.tools.console import Console
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from correct_images.calculate_correction_parameters import \
    calc_attenuation_correction_gain, apply_atn_crr_2_img
from correct_images.read_mission import read_params
from correct_images.utilities import *
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from auv_nav.correct_images.utilities import MonoCamera


def develop_corrected_image(path, force):
    '''

    :param path_mission: Path to 'mission.yaml'.
    :param path_correct: Path to 'correct_images.yaml'
    :return: None. Result image files and configurations are saved as files.
    '''
    path = Path(path).resolve()
    path_correct = get_config_folder(path) / "correct_images.yaml"
    if not path_correct.exists():
        Console.warn(
            'Config File does not exist. Did you parse first this dive?')
        Console.quit('run correct_images parse first.')
    path_mission = get_raw_folder(path) / "mission.yaml"
    path_processed = get_processed_folder(path)

    # load configuration from mission.yaml, correct_images.yaml files
    Console.info('Loading', path_mission, datetime.datetime.now())
    Console.info('Loading', path_correct, datetime.datetime.now())

    mission = read_params(path_mission,
                          'mission')  # read_params(path to file, type of file: mission/correct_config)

    print('mission:', path_mission)

    config_ = read_params(path_correct, 'correct')
    camera_format = mission.image.format

    if camera_format == 'biocam':
        camera_iterations = 2
        for i in range(camera_iterations):
            if i is 0:
                camera = config_.config.camera_1
                img_p = mission.image.cameras_0.get('path')
            elif i is 1:
                camera = config_.config.camera_2
                img_p = mission.image.cameras_1.get('path')

            src_file_format = 'tif'
            # read image path from mission.yaml
            # img_p = mission.image.cameras_0.get('path')
            print('camera', camera, ', type', type(camera))
            print('img_p', img_p, ', type', type(img_p))
            print('Bool camera in img_p', (camera in img_p))
            if camera in img_p:
                img_path = img_p
            else:
                img_p = mission.image.cameras_1.get('path')

                # TODO skip if
                img_path = img_p

                # if camera in img_p:
                #     img_path = img_p
                # else:
                #     print('Mission yaml file does not have path to camera: ',
                #           camera)
                #     continue

            print('path to image', Path(path_processed / img_path))
            if not Path(path_processed / img_path).exists():
                Console.info('Image path does not exist for camera', camera)
                continue

            # find path to attenuation parameters
            params_folder = 'attenuation_correction/params_' + camera
            params_dir_path = path_processed / img_path
            params_dir_path = params_dir_path / params_folder
            bayer_folder = 'bayer_' + camera
            bayer_path = params_dir_path.parents[0] / bayer_folder

            # load filelist.csv to read bayer files' paths
            filelist_path = params_dir_path / 'filelist.csv'
            if filelist_path.exists():
                df_filelist = pd.read_csv(filelist_path)
            else:
                Console.warn(
                    'filelist.csv not found in target folder for camera ',
                    camera)
                Console.warn('Run correct_images [parse] before [process].')
                Console.warn(filelist_path)
                continue

            # destination folder for corrected images 
            dst_folder = 'developed_' + camera
            dst_dir_path = params_dir_path.parents[0] / dst_folder

            # check if images will be overwritten or written for first time
            if not dst_dir_path.exists():
                dst_dir_path.mkdir(parents=True)
                Console.info(
                    'Code will write images for the first time for camera ',
                    camera)
            else:
                if force is True:
                    Console.warn('Processed images already exist for camera ',
                                 camera)
                    Console.warn('Code will overwrite existing images.')

                else:
                    Console.warn('Processed images already exist for camera ',
                                 camera)
                    Console.warn(
                        'Run correct_images with [process] [-F] option for overwriting existing processed images.')
                    continue

            # load config.yaml
            path_config = params_dir_path / 'config.yaml'
            with path_config.open('r') as stream:
                load_data_config = yaml.safe_load(stream)

            label_altitude = load_data_config['label_altitude']
            target_altitude = load_data_config['target_altitude']
            src_file_format = load_data_config['src_file_format']

            # load from correct_images.yaml
            target_mean = config_.normalization.target_mean
            target_std = config_.normalization.target_std
            src_img_index = config_.config.src_img_index
            apply_attenuation_correction = config_.flags.apply_attenuation_correction
            apply_gamma_correction = config_.flags.apply_gamma_correction
            apply_distortion_correction = config_.flags.apply_distortion_correction
            camera_parameter_file_path = config_.flags.camera_parameter_file_path

            debayer_option = config_.output.debayer_option
            bayer_pattern = config_.output.bayer_pattern

            # TODO should be read from yaml file
            if i is 1:
                bayer_pattern = 'greyscale'

            dst_img_format = config_.output.dst_file_format
            median_filter_kernel_size = config_.attenuation_correction.median_filter_kernel_size

            # load .npy files
            pdp = str(params_dir_path)

            atn_crr_params = np.load(pdp + '/atn_crr_params.npy')
            bayer_img_mean = np.load(pdp + '/bayer_img_mean_raw.npy')
            bayer_img_std = np.load(pdp + '/bayer_img_std_raw.npy')
            bayer_img_corrected_mean = np.load(
                pdp + '/bayer_img_mean_atn_crr.npy')
            bayer_img_corrected_std = np.load(
                pdp + '/bayer_img_std_atn_crr.npy')

            # load values from file list

            list_altitude = df_filelist[label_altitude].values
            list_bayer_file = df_filelist['bayer file'].values

            list_bayer_file_path = list_bayer_file

            # convert from relative to absolute path
            for i_bayer_file in range(len(list_bayer_file)):
                list_bayer_file_path[i_bayer_file] = bayer_path / \
                                                     list_bayer_file[
                                                         i_bayer_file]

            # get image size
            bayer_sample = np.load(str(list_bayer_file_path[0]))
            a = bayer_sample.shape[0]
            b = bayer_sample.shape[1]

            # identify debayer parameters for opencv
            if debayer_option == 'linear':
                code = cv2.COLOR_BAYER_GR2BGR
            elif debayer_option == 'ea':
                code = cv2.COLOR_BAYER_GR2BGR_EA
            elif debayer_option == 'vng':
                code = cv2.COLOR_BAYER_GR2BGR_VNG

            # calculate distortion correction paramters
            map_x = None
            map_y = None
            if apply_distortion_correction:
                if camera_parameter_file_path == 'None':
                    Console.quit(
                        'Camera parameters path not provided with distortion correction flag set to True')

                else:
                    camera_calib_name = '/mono_' + camera + '.yaml'
                    camera_parameter_file_path = camera_parameter_file_path + camera_calib_name
                    camera_params_path = Path(
                        camera_parameter_file_path).resolve()

                    if not Path(camera_params_path).exists():
                        print('Path to camera parameters does not exist')
                        sys.exit()
                    else:
                        map_x, map_y = calc_distortion_mapping(
                            camera_params_path, b, a)

            # if developing target are not designated, develop all files in filelist.csv
            if src_img_index == -1:
                src_img_index = range(len(df_filelist))

            message = 'developing images ' + datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            list_dst_name = []
            # img_index = 0

            list_dst_name = joblib.Parallel(n_jobs=-2,
                                            verbose=3)(
                [joblib.delayed(process_img)(
                    apply_attenuation_correction,
                    apply_distortion_correction,
                    apply_gamma_correction,
                    atn_crr_params,
                    bayer_img_corrected_mean,
                    bayer_img_corrected_std,
                    bayer_img_mean,
                    bayer_img_std, bayer_pattern,
                    dst_dir_path, dst_img_format,
                    i_img,
                    list_altitude, list_bayer_file,
                    list_bayer_file_path, map_x,
                    map_y,
                    target_altitude, target_mean,
                    target_std) for i_img in src_img_index])


            # for i_img in tqdm(src_img_index, ascii=True, desc=message):
            #     # attenuation correction or only pixel stat
            #     dst_path = process_img()
            #
            #     list_dst_name.append(dst_path.name)
            #     # img_index = img_index + 1

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
            Console.info('Done. Configurations are saved to ', cfg_filepath,
                         datetime.datetime.now())
            Console.info('#### ------ Process completed ------ #####')

            # TODO discuss whether bayer files should be deleted after
            #  developing. If they are deleted, developing with different
            #  parameter (mainly mean and std) gets difficult.
            # remove the bayer folder containing npy files
            # shutil.rmtree(bayer_path)

    if camera_format == 'seaxerocks_3':
        camera_iterations = 2
        for i in range(camera_iterations):
            if i is 0:
                camera = config_.config.camera_1
            elif i is 1:
                camera = config_.config.camera_2

            src_file_format = 'raw'

            # read image path from mission.yaml
            img_p = mission.image.cameras_0.get('path')
            if camera in img_p:
                img_path = img_p
            else:
                img_p = mission.image.cameras_1.get('path')
                if camera in img_p:
                    img_path = img_p
                else:
                    img_p = mission.image.cameras_2.get('path')
                    if camera in img_p:
                        img_path = img_p
                    else:
                        print(
                            'Mission yaml file does not have path to camera: ',
                            camera)
                        continue

            if not Path(path_processed / img_path).exists():
                Console.warn('Image path does not exist for camera', camera)
                continue

            # find the path to parameters folders
            params_folder = 'attenuation_correction/params_' + camera
            params_dir_path = path_processed / img_path
            params_dir_path = params_dir_path / params_folder
            bayer_folder = 'bayer' + camera
            bayer_path = params_dir_path.parents[0] / bayer_folder

            # load bayer files paths from filelist.csv
            filelist_path = params_dir_path / 'filelist.csv'
            if filelist_path.exists():
                df_filelist = pd.read_csv(filelist_path)
            else:
                Console.warn(
                    'filelist.csv not found in target folder for camera ',
                    camera)
                Console.warn('Run correct_images [parse] before [process].')
                Console.warn(filelist_path)
                continue

            # destination folder for corrected images 
            dst_folder = 'developed_' + camera
            dst_dir_path = params_dir_path.parents[0] / dst_folder

            # check if the images are going to be overwritten or newly written 
            if not dst_dir_path.exists():
                dst_dir_path.mkdir(parents=True)
                Console.info(
                    'Code will write images for the first time for camera ',
                    camera)
            else:
                if force is True:
                    Console.warn('Processed images already exist for camera ',
                                 camera)
                    Console.warn('Code will overwrite existing images.')

                else:
                    Console.warn('Processed images already exist for camera ',
                                 camera)
                    Console.warn(
                        'Run correct_images with [process] [-F] option for overwriting existing processed images.')
                    continue

            # load config.yaml
            path_config = params_dir_path / 'config.yaml'
            with path_config.open('r') as stream:
                load_data_config = yaml.safe_load(stream)

            label_altitude = load_data_config['label_altitude']
            target_altitude = load_data_config['target_altitude']
            src_file_format = load_data_config['src_file_format']

            # load from correct_images.yaml
            target_mean = config_.normalization.target_mean
            target_std = config_.normalization.target_std
            src_img_index = config_.config.src_img_index
            apply_attenuation_correction = config_.flags.apply_attenuation_correction
            apply_gamma_correction = config_.flags.apply_gamma_correction
            apply_distortion_correction = config_.flags.apply_distortion_correction
            camera_parameter_file_path = config_.flags.camera_parameter_file_path
            dst_img_format = config_.output.dst_file_format
            median_filter_kernel_size = config_.attenuation_correction.median_filter_kernel_size
            debayer_option = config_.output.debayer_option
            bayer_pattern = config_.output.bayer_pattern

            # load .npy files
            pdp = str(params_dir_path)
            atn_crr_params = np.load(pdp + '/atn_crr_params.npy')
            bayer_img_mean = np.load(pdp + '/bayer_img_mean_raw.npy')
            bayer_img_std = np.load(pdp + '/bayer_img_std_raw.npy')
            bayer_img_corrected_mean = np.load(
                pdp + '/bayer_img_mean_atn_crr.npy')
            bayer_img_corrected_std = np.load(
                pdp + '/bayer_img_std_atn_crr.npy')

            # load values from file list
            list_altitude = df_filelist[label_altitude].values
            list_bayer_file = df_filelist['bayer file'].values

            list_bayer_file_path = list_bayer_file

            # convert from relative path to real path
            for i_bayer_file in range(len(list_bayer_file)):
                # list_bayer_file[i_bayer_file] = params_dir_path / list_bayer_file[i_bayer_file]
                list_bayer_file_path[i_bayer_file] = bayer_path / \
                                                     list_bayer_file[
                                                         i_bayer_file]

            # get image size
            bayer_sample = np.load(str(list_bayer_file_path[0]))
            a = bayer_sample.shape[0]
            b = bayer_sample.shape[1]

            if debayer_option == 'linear':
                code = cv2.COLOR_BAYER_GR2BGR
            elif debayer_option == 'ea':
                code = cv2.COLOR_BAYER_GR2BGR_EA
            elif debayer_option == 'vng':
                code = cv2.COLOR_BAYER_GR2BGR_VNG

            # calculate distortion correction paramters
            if apply_distortion_correction:
                if camera_parameter_file_path == 'None':
                    Console.quit(
                        'Camera parameters path not provided with distortion correction flag set to True')
                else:
                    camera_calib_name = '/mono_' + camera + '.yaml'
                    camera_parameter_file_path = camera_parameter_file_path + camera_calib_name
                    camera_params_path = Path(
                        camera_parameter_file_path).resolve()

                    if not Path(camera_params_path).exists():
                        Console.quit(
                            'Path to camera parameters does not exist')
                    else:
                        map_x, map_y = calc_distortion_mapping(
                            camera_params_path, b, a)

            # if developing target are not designated, develop all files in filelist.csv
            if src_img_index == -1:
                src_img_index = range(len(df_filelist))

            message = 'developing images ' + datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            list_dst_name = []
            img_index = 0
            for i_img in tqdm(src_img_index, ascii=True, desc=message):

                # attenuation correction or only pixel stat
                if apply_attenuation_correction:
                    corrected_bayer_img = attenuation_correction_bayer(
                        np.load(str(list_bayer_file_path[i_img])),
                        bayer_img_corrected_mean, bayer_img_corrected_std,
                        target_mean, target_std, atn_crr_params,
                        list_altitude[i_img], target_altitude, True, 8)

                else:
                    corrected_bayer_img = pixel_stat_bayer(
                        np.load(str(list_bayer_file_path[i_img])),
                        bayer_img_mean, bayer_img_std,
                        target_mean, target_std, 8)

                # Debayer image
                # corrected_rgb_img = demosaicing_CFA_Bayer_bilinear(corrected_bayer_img, bayer_pattern)
                corrected_rgb_img = cv2.cvtColor(
                    corrected_bayer_img.astype(np.uint8), code)

                if apply_distortion_correction:
                    corrected_rgb_img = correct_distortion(corrected_rgb_img,
                                                           map_x, map_y, 8)

                if apply_gamma_correction:
                    corrected_rgb_img = gamma_correct(corrected_rgb_img, 8)

                corrected_rgb_img = corrected_rgb_img.astype(np.uint8)

                image_name = list_bayer_file[img_index].stem
                image_name_str = str(image_name + '.' + dst_img_format)
                dst_path = dst_dir_path / image_name_str
                imageio.imwrite(dst_path, corrected_rgb_img)
                list_dst_name.append(dst_path.name)
                img_index = img_index + 1

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
            Console.info('Done. Configurations are saved to ', cfg_filepath,
                         datetime.datetime.now())

            Console.info('#### ------ Process completed ------ #####')

            # remove the bayer folder containing npy files 
            shutil.rmtree(bayer_path)

    if camera_format == 'acfr_standard' or camera_format == 'unnagi':

        camera_iterations = 2
        for i in range(camera_iterations):
            if i is 0:
                camera = config_.config.camera_1
            elif i is 1:
                camera = config_.config.camera_2

            src_file_format = 'tif'

            # read image path from mission.yaml
            img_path = mission.image.cameras_0.get('path')
            if not Path(path_processed / img_path).exists():
                Console.warn('Image path does not exist for camera', camera)
                continue

            # find parameters folders
            params_folder = 'attenuation_correction/params_' + camera
            params_dir_path = path_processed / img_path
            params_dir_path = params_dir_path / params_folder
            bayer_folder = 'bayer' + camera
            bayer_path = params_dir_path.parents[0] / bayer_folder

            # load filelist
            filelist_path = params_dir_path / 'filelist.csv'
            if filelist_path.exists():
                df_filelist = pd.read_csv(filelist_path)
            else:
                Console.warn(
                    'Filelist.csv not found in target folder for camera ',
                    camera)
                Console.warn('Run correct_images [parse] before [process].')
                Console.warn(filelist_path)
                continue

            # output folder for corrected images
            dst_folder = 'developed_' + camera
            dst_dir_path = params_dir_path.parents[0] / dst_folder

            # check if images will be overwritten or newly written
            if not dst_dir_path.exists():
                dst_dir_path.mkdir(parents=True)
                Console.info(
                    'Code will write images for the first time for camera ',
                    camera)
            else:
                if force is True:
                    Console.warn('Processed images already exist for camera ',
                                 camera)
                    Console.warn('Code will overwrite existing images.')

                else:
                    Console.warn('Processed Images already exist for camera ',
                                 camera)
                    Console.warn(
                        'Run correct_images with [process] [-F] option for overwriting existing processed images.')
                    continue

            # load config.yaml
            path_config = params_dir_path / 'config.yaml'
            with path_config.open('r') as stream:
                load_data_config = yaml.safe_load(stream)

            label_altitude = load_data_config['label_altitude']
            target_altitude = load_data_config['target_altitude']
            src_file_format = load_data_config['src_file_format']

            # load from correct_images.yaml
            target_mean = config_.normalization.target_mean  # load_data.get('target_mean', 30)
            target_std = config_.normalization.target_std  # load_data.get('target_std', 5)
            src_img_index = config_.config.src_img_index  # load_data.get('src_img_index', -1)
            apply_attenuation_correction = config_.flags.apply_attenuation_correction  # load_data.get('apply_attenuation_correction', True)
            apply_gamma_correction = config_.flags.apply_gamma_correction  # load_data.get('apply_gamma_correction', True)
            apply_distortion_correction = config_.flags.apply_distortion_correction  # load_data.get('apply_distortion_correction', True)
            camera_parameter_path = config_.flags.camera_parameter_file_path  # load_data.get('camera_parameter_file_path', None)
            camera_parameter_file_path = Path(camera_parameter_path).resolve()
            #dst_dir_path = None # load_data.get('dst_dir_path', None)
            dst_img_format = config_.output.dst_file_format # load_data.get('dst_img_format', 'png')
            median_filter_kernel_size = config_.attenuation_correction.median_filter_kernel_size # load_data.get('median_filter_kernel_size', 1)
            debayer_option = config_.output.debayer_option # load_data.get('debayer_option', 'linear')


            # load .npy files
            pdp = str(params_dir_path)
            atn_crr_params = np.load(pdp + '/atn_crr_params.npy')
            bayer_img_mean = np.load(pdp + '/bayer_img_mean_raw.npy')
            bayer_img_std = np.load(pdp + '/bayer_img_std_raw.npy')
            bayer_img_corrected_mean = np.load(
                pdp + '/bayer_img_mean_atn_crr.npy')
            bayer_img_corrected_std = np.load(
                pdp + '/bayer_img_std_atn_crr.npy')

            # load values from file list

            list_altitude = df_filelist[label_altitude].values
            list_bayer_file = df_filelist['bayer file'].values

            list_bayer_file_path = list_bayer_file

            # convert from relative path to real path
            for i_bayer_file in range(len(list_bayer_file)):
                # list_bayer_file[i_bayer_file] = params_dir_path / list_bayer_file[i_bayer_file]
                list_bayer_file_path[i_bayer_file] = bayer_path / \
                                                     list_bayer_file[
                                                         i_bayer_file]

            # get image size
            bayer_sample = np.load(str(list_bayer_file_path[0]))
            a = bayer_sample.shape[0]
            b = bayer_sample.shape[1]

            # identify debayer params for opencv
            if debayer_option == 'linear':
                code = cv2.COLOR_BAYER_GR2BGR
            elif debayer_option == 'ea':
                code = cv2.COLOR_BAYER_GR2BGR_EA
            elif debayer_option == 'vng':
                code = cv2.COLOR_BAYER_GR2BGR_VNG

            # calculate distortion correction paramters
            map_x = None
            map_y = None
            if apply_distortion_correction:
                if camera_parameter_file_path == 'None':
                    Console.quit(
                        'Camera parameters path not provided with distortion correction flag set to True')
                else:
                    camera_calib_name = '/mono_' + camera + '.yaml'
                    camera_parameter_file_path = camera_parameter_file_path + camera_calib_name
                    camera_params_path = Path(
                        camera_parameter_file_path).resolve()

                    if not Path(camera_params_path).exists():
                        Console.quit(
                            'Path to camera parameters does not exist')
                    else:
                        map_x, map_y = calc_distortion_mapping(
                            camera_params_path, b, a)

            # if developing target are not designated, develop all files in filelist.csv
            if src_img_index == -1:
                src_img_index = range(len(df_filelist))

            message = 'developing images ' + datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
            list_dst_name = []
            img_index = 0
            for i_img in tqdm(src_img_index, ascii=True, desc=message):

                # attenuation correction or only pixel stat
                if apply_attenuation_correction:
                    corrected_bayer_img = attenuation_correction_bayer(
                        np.load(str(list_bayer_file_path[i_img])),
                        bayer_img_corrected_mean, bayer_img_corrected_std,
                        target_mean, target_std, atn_crr_params,
                        list_altitude[i_img], target_altitude, True, 8)

                else:
                    corrected_bayer_img = pixel_stat_bayer(
                        np.load(str(list_bayer_file_path[i_img])),
                        bayer_img_mean, bayer_img_std,
                        target_mean, target_std, 8)

                # debayer image
                # corrected_rgb_img = demosaicing_CFA_Bayer_bilinear(corrected_bayer_img, bayer_pattern)
                corrected_rgb_img = cv2.cvtColor(
                    corrected_bayer_img.astype(np.uint8), code)

                if apply_distortion_correction:
                    corrected_rgb_img = correct_distortion(corrected_rgb_img,
                                                           map_x, map_y, 8)

                if apply_gamma_correction:
                    corrected_rgb_img = gamma_correct(corrected_rgb_img, 8)

                corrected_rgb_img = corrected_rgb_img.astype(np.uint8)

                image_name = list_bayer_file[img_index].stem
                image_name_str = str(image_name + '.' + dst_img_format)
                dst_path = dst_dir_path / image_name_str
                imageio.imwrite(dst_path, corrected_rgb_img)
                list_dst_name.append(dst_path.name)
                img_index = img_index + 1

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
            Console.info('Done. Configurations are saved to ', cfg_filepath,
                         datetime.datetime.now())

            Console.info('#### ------ Process completed ------ #####')

            # remove the bayer folder containing npy files 
            shutil.rmtree(bayer_path)


def process_img(apply_attenuation_correction, apply_distortion_correction,
                apply_gamma_correction, atn_crr_params,
                bayer_img_corrected_mean, bayer_img_corrected_std,
                bayer_img_mean, bayer_img_std, bayer_pattern, dst_dir_path,
                dst_img_format, i_img, list_altitude, list_bayer_file,
                list_bayer_file_path, map_x, map_y, target_altitude,
                target_mean, target_std):
    if apply_attenuation_correction:
        corrected_bayer_img = attenuation_correction_bayer(
            np.load(str(list_bayer_file_path[i_img])),
            bayer_img_corrected_mean, bayer_img_corrected_std,
            target_mean, target_std, atn_crr_params,
            list_altitude[i_img], target_altitude, True, 8)

    else:
        corrected_bayer_img = pixel_stat_bayer(
            np.load(str(list_bayer_file_path[i_img])),
            bayer_img_mean, bayer_img_std,
            target_mean, target_std, 8)
    # Debayer image
    if bayer_pattern == 'greyscale':
        corrected_rgb_img = corrected_bayer_img
    else:
        corrected_rgb_img = demosaicing_CFA_Bayer_bilinear(
            corrected_bayer_img, bayer_pattern)
    # corrected_rgb_img = cv2.cvtColor(corrected_bayer_img.astype(np.uint8), code)
    if apply_distortion_correction:
        corrected_rgb_img = correct_distortion(corrected_rgb_img,
                                               map_x, map_y, 8)
    if apply_gamma_correction:
        corrected_rgb_img = gamma_correct(corrected_rgb_img, 8)
    corrected_rgb_img = corrected_rgb_img.astype(np.uint8)
    image_name = list_bayer_file[i_img].stem
    # image_name = list_bayer_file[img_index].stem
    image_name_str = str(image_name + '.' + dst_img_format)
    dst_path = dst_dir_path / image_name_str
    imageio.imwrite(dst_path, corrected_rgb_img)
    return dst_path


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
                list_params_fil[i_mos][:, :, i, j] = filters.median_filter(
                    list_params[i_mos][:, :, i, j],
                    (kernel_size, kernel_size))

    ret = np.zeros(src_atn_param.shape, src_atn_param.dtype)
    ret[0::2, 0::2, :, :] = params_0_fil
    ret[0::2, 1::2, :, :] = params_1_fil
    ret[1::2, 0::2, :, :] = params_2_fil
    ret[1::2, 1::2, :, :] = params_3_fil

    return ret


def calc_distortion_mapping(camera_parameter_file_path, a, b):
    # MonoCamera is a Class to read camera parameters: defined in utilities.py
    mono_cam = MonoCamera(camera_parameter_file_path)
    cam_mat, _ = cv2.getOptimalNewCameraMatrix(mono_cam.K, mono_cam.d, (a, b),
                                               0)
    map_x, map_y = cv2.initUndistortRectifyMap(mono_cam.K, mono_cam.d, None,
                                               cam_mat, (b, a), 5)
    return map_x, map_y


def attenuation_correction_bayer(bayer_img, bayer_img_mean, bayer_img_std,
                                 target_mean, target_std, atn_crr_params,
                                 src_altitude, target_altitude,
                                 apply_attenuation_correction=True, dst_bit=8):
    gain = calc_attenuation_correction_gain(target_altitude, atn_crr_params)
    ret = apply_atn_crr_2_img(bayer_img, src_altitude, atn_crr_params, gain)
    ret = pixel_stat_bayer(ret, bayer_img_mean, bayer_img_std, target_mean,
                           target_std, dst_bit)
    ret = np.clip(ret, 0, 2 ** dst_bit - 1)
    return ret


def pixel_stat_bayer(bayer_img, bayer_img_mean, bayer_img_std, target_mean,
                     target_std, dst_bit=8):
    # target_mean and target std should be given in 0 - 100 scale
    target_mean_in_bitdeph = target_mean / 100.0 * (2.0 ** dst_bit - 1.0)
    target_std_in_bitdeph = target_std / 100.0 * (2.0 ** dst_bit - 1.0)
    ret = (
                  bayer_img - bayer_img_mean) / bayer_img_std * target_std_in_bitdeph + target_mean_in_bitdeph
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
