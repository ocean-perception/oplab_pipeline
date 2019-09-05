import datetime
import sys
import random
import socket
import uuid
import os

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
from pathlib import Path

from auv_nav.tools.console import Console
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_processed_folder
from auv_nav.tools.folder_structure import get_config_folder
from correct_images.read_mission import read_params

from numpy.linalg import inv


def calculate_correction_parameters(path, force):
    '''

    :param path_mission: Path to 'mission.yaml'.
    :param path_correct: Path to 'correct_images.yaml'
    :return: None. Result image files and configurations are saved as files.
    '''
    path = Path(path).resolve()
    path_mission = get_raw_folder(path) / "mission.yaml"
    path_raw = get_raw_folder(path)
    path_processed = get_processed_folder(path)

    
    Console.info('loading', path_mission, datetime.datetime.now())
    # load parameters from mission.yaml and correct_images.yaml
    # read_params(path to file, type of file: mission/correct_config)
    mission = read_params(path_mission, 'mission')
    camera_format = mission.image.format
    path_correct = get_config_folder(path) / "correct_images.yaml"

    # load default correct_images.yaml if it does not exist in configuration folder structure
    if not path_correct.exists():
        root = Path(__file__).parents[1]
        default_file = root / 'correct_images/default_yaml' / 'correct_images.yaml'
        Console.warn("Cannot find {}, generating default from {}".format(
            path_correct, default_file))
        
        default_file.copy(path_correct)
        Console.info('loading', path_correct, datetime.datetime.now())

        # discovering default parameters from mission.yaml for default correct_images.yaml

        for entry in os.scandir(path_processed):
            # print(entry)
            if entry.is_dir():
                if 'json_renav_' in entry.path:
                    json_path = entry.path
        print('JSON:', json_path)
        with path_correct.open('r') as f:
            params = yaml.safe_load(f)
            params['config']['auv_nav_path'] = json_path
            params['config']['format'] = camera_format
            if camera_format == 'seaxerocks_3':
                params['config']['camera1'] = 'Cam51707925'
                params['config']['camera2'] = 'Cam51707923'
                #params['config']['camera3'] = 'LM165'
            elif camera_format == 'acfr_standard' or camera_format == 'unaggi':
                params['config']['camera1'] = 'LC'
                params['config']['camera2'] = 'RC'
            elif camera_format == 'biocam':
                params['config']['camera1'] = 'cam61003146_strobe'
                params['config']['camera2'] = 'cam61004444_strobe'
        # dump default parameters camera and auv_nav_path to correct_images.yaml file
        with path_correct.open('w') as f:
            yaml.dump(params, f)

    # load parameters from correct_images.yaml if it already exists
    config_ = read_params(path_correct, 'correct')
    label_raw_file = 'raw file'
    label_altitude = ' Altitude [m]'
    altitude_max = config_.attenuation_correction.altitude_max.get('max')
    altitude_min = config_.attenuation_correction.altitude_min.get('min')
    sampling_method = config_.attenuation_correction.sampling_method
    format_ = config_.config.format


    #check for image format: biocam, seaxerocks_3, acfr_standard
    if format_ == 'biocam':
        camera_iterations = 2
        for i in range(camera_iterations):
            if i is 0:
                camera = config_.config.camera_1
            elif i is 1:
                camera = config_.config.camera_2
            
            src_file_format = 'tif'
            calculated_atn_crr_params_path = None
            
            dst_file_format = config_.output.dst_file_format
            joblib_verbose = 3
            init_atn_crr_params_path = None
            trim_ratio = 0.2

            # load src file or target files to data frame
            
            src_filelist_path = None
            
            # read path to raw images from mission.yaml file
            img_p = mission.image.cameras_0.get('path')

            if camera in img_p:
                img_path = img_p
                camera_serial = mission.image.cameras_0.get('name')
            else:
                img_p = mission.image.cameras_1.get('path')
                if camera in img_p:
                    img_path = img_p
                    camera_serial = mission.image.cameras_1.get('name')
                else:
                    print('Mission yaml file does not have path to camera: ', camera)
                    continue
                
            # read path to CSV file
            auv_nav_filepath = Path(config_.config.auv_nav_path).resolve()
            
            src_file_dirpath = path_raw / img_path
                        
            csv_path = 'csv/dead_reckoning/auv_dr_' + camera_serial + '.csv'
            
            auv_nav_filepath = auv_nav_filepath / csv_path


            df_all = pd.read_csv(auv_nav_filepath,
                                 dtype={'Imagenumber': object})
            raw_file_list = [None] * len(df_all)
            for i_file in range(len(raw_file_list)):
                if src_file_format == 'raw':
                    raw_file_list[i_file] = src_file_dirpath / str(
                        df_all['Imagenumber'][i_file].zfill(7) + '.raw')
                elif src_file_format == 'tif':
                    raw_file_list[i_file] = src_file_dirpath / \
                                            df_all['Imagenumber'][i_file]
                else:
                    Console.error('src_file_format:', src_file_format,
                                  'is incorrect.')
                    return
            df_all = pd.concat([df_all, pd.DataFrame(
                raw_file_list, columns=[label_raw_file])], axis=1)
            src_filelist = df_all[label_raw_file]

            # set up parameters for attenuation correction
            target_altitude = None  
            curve_fit_trial_num = 1 
            
            bin_band = 0.1  
            min_sample_per_bin = 5  
            max_sample_per_bin = 100  
            median_filter_kernel_size = 1

            # remove too low or too high altitude file and too small file size file
            altitudes_all = df_all[label_altitude].values
            match_count = 0
            
            # check if altitudes match with min and max provided in correct_images.yaml
            for i in range(len(altitudes_all)):
                if altitudes_all[i] <= altitude_max and altitudes_all[i] >= altitude_min:
                    match_count = match_count + 1
            if match_count < 1:
                Console.quit(
                    'altitude values in dive dataset do not match with minimum and maximum altitude provided in correct_images.yaml')
            else:
                idx_effective_data = np.where(
                    (altitudes_all >= altitude_min) & (
                            altitudes_all <= altitude_max))
            
            # configure output file path
            
            dirpath = src_filelist[0].parent

            dirpath = get_processed_folder(dirpath)
            print(dirpath)
            
            dirpath = dirpath / 'attenuation_correction'
            if not dirpath.exists():
                dirpath.mkdir(parents=True)
            dirpath_atn_crr = dirpath / 'tmp_atn_crr'
            bayer_folder_name = 'bayer' + camera
            dirpath_bayer = dirpath / bayer_folder_name
            if not dirpath_bayer.exists():
                dirpath_bayer.mkdir(parents=True)

            # file path of output image data
            dst_filelist = [None] * len(df_all)
            bayer_filelist = [None] * len(df_all)
            atn_crr_filelist = [None] * len(df_all)
            for i_dst_file in range(len(dst_filelist)):
                tmp_filepath = src_filelist[i_dst_file]
                file_stem = get_processed_folder(tmp_filepath).stem
                dst_filelist[i_dst_file] = dirpath / str(
                    file_stem + '.' + dst_file_format)
                bayer_filelist[i_dst_file] = dirpath_bayer / str(
                    file_stem + '.npy')
                atn_crr_filelist[i_dst_file] = dirpath_atn_crr / str(
                    file_stem + '.npy')

            # file path of metadata
            params_folder_name = 'params_' + camera
            dir_path_image_crr_params = dst_filelist[
                                            0].parent / params_folder_name
            
            if not dir_path_image_crr_params.exists():
                dir_path_image_crr_params.mkdir(parents=True)
                Console.info(
                    'code will compute correction parameters for this Camera for first time.')
            else:
                print(dir_path_image_crr_params)
                if force is True:
                    Console.warn(
                        'Attenuation correction parameters already exist.')
                    Console.warn('Code will overwrite existing parameters.')

                else:
                    Console.warn(
                        'Code will quit - correction parameters already exist.')
                    Console.warn(
                        'Run correct_images with [parse] [-F] option for overwriting existing correction parameters.')
                    sys.exit()

            if calculated_atn_crr_params_path is None:
                calculated_atn_crr_params_path = dir_path_image_crr_params / 'atn_crr_params.npy'

            file_list_raw = df_all[label_raw_file].values.tolist()

            if src_file_format == 'raw':
                # xviii camera
                a, b = 1024, 1280
                # developing .raw data to bayer data of uint32 numpy array.
                Console.info('start loading bayer images', len(file_list_raw),
                             'files to', dirpath_bayer,
                             datetime.datetime.now())
                src_file_list_not_exist = []
                bayer_file_list_not_exsit = []
                for idx_raw in range(len(file_list_raw)):
                    if not bayer_filelist[idx_raw].exists():
                        src_file_list_not_exist.append(file_list_raw[idx_raw])
                        bayer_file_list_not_exsit.append(
                            bayer_filelist[idx_raw])

                Console.info(
                    len(file_list_raw) - len(bayer_file_list_not_exsit),
                    'files have already existed.')

                task_num = 100
                num_loop = int(len(bayer_file_list_not_exsit) / task_num) + 1
                start_idx = 0
                idx_total = start_idx
                end_idx = 0
                while start_idx < len(bayer_file_list_not_exsit):
                    # for debug
                    #     break

                    Console.info(
                        'processing load_xviii_bayer_from_binary',
                        int(start_idx / task_num) + 1, '/', num_loop,
                        'of total',
                        len(bayer_file_list_not_exsit), 'files',
                        datetime.datetime.now(), flush=True)

                    end_idx = start_idx + task_num
                    if end_idx > len(bayer_file_list_not_exsit):
                        end_idx = len(bayer_file_list_not_exsit)

                    raw_img_for_size = np.fromfile(str(
                        src_file_list_not_exist[start_idx]), dtype=np.uint8)
                    arg_bayer_img = np.zeros(
                        (end_idx - start_idx, raw_img_for_size.shape[0]),
                        dtype=raw_img_for_size.dtype)
                    for idx_raw in range(start_idx, end_idx):
                        arg_bayer_img[idx_raw - start_idx, :] = np.fromfile(
                            str(src_file_list_not_exist[idx_raw]),
                            dtype=raw_img_for_size.dtype)

                    results = joblib.Parallel(n_jobs=-2,
                                              verbose=joblib_verbose)(
                        [joblib.delayed(load_xviii_bayer_from_binary)(
                            arg_bayer_img[idx_arg, :]) for idx_arg in
                            range(end_idx - start_idx)])

                    for idx_raw in range(start_idx, end_idx):
                        np.save(bayer_file_list_not_exsit[idx_raw],
                                results[idx_raw - start_idx])

                    start_idx = end_idx

            elif src_file_format == 'tif' or src_file_format == 'tiff':
                # unaggi camera
                Console.info('start loading tif images', len(file_list_raw),
                             'files to', dirpath_bayer,
                             datetime.datetime.now())
                src_file_list_not_exist = []
                bayer_file_list_not_exsit = []
                for idx_raw in range(len(file_list_raw)):
                    if not bayer_filelist[idx_raw].exists():
                        src_file_list_not_exist.append(file_list_raw[idx_raw])
                        bayer_file_list_not_exsit.append(
                            bayer_filelist[idx_raw])

                Console.info(
                    len(file_list_raw) - len(bayer_file_list_not_exsit),
                    'files have already existed.')

                tmp_tif_for_size = imageio.imread(file_list_raw[0])
                a = tmp_tif_for_size.shape[0]
                b = tmp_tif_for_size.shape[1]

                for i_file_not_exist in range(len(src_file_list_not_exist)):
                    tmp_tif = imageio.imread(
                        src_file_list_not_exist[i_file_not_exist])
                    tmp_npy = np.zeros([a, b], np.uint16)
                    tmp_npy[:, :] = np.array(tmp_tif, np.uint16)
                    np.save(bayer_file_list_not_exsit[i_file_not_exist],
                            tmp_npy)

            # caluculate attenuation correction parameter
            if target_altitude is None:
                target_altitude = float(
                    np.mean(altitudes_all[idx_effective_data]))

            # memmap is created at local directory
            file_name_memmap_raw, memmap_raw = load_memmap_from_npy_filelist(
                bayer_filelist)
            print('Memmap directory: ', file_name_memmap_raw)

            Console.info('start calculate mean and std of raw img',
                         datetime.datetime.now())

            img_mean_raw, img_std_raw = \
                calc_img_mean_and_std_trimmed(memmap_raw, trim_ratio,
                                              calc_std=True,
                                              effective_index=idx_effective_data)

            dirpath_img_mean_raw = dir_path_image_crr_params / 'bayer_img_mean_raw'
            dirpath_img_std_raw = dir_path_image_crr_params / 'bayer_img_std_raw'
            np.save(str(dirpath_img_mean_raw), img_mean_raw)
            np.save(str(dirpath_img_std_raw), img_std_raw)

            list_dirpath = [dirpath_img_mean_raw, dirpath_img_std_raw]
            list_img = [img_mean_raw, img_std_raw]
            for i_img in range(len(list_img)):
                save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

            # calculate regression parameters for all pixels and channels
            Console.info('start attenuation correction parameter calculation.')

            # 3 is number of parameter in exp_curve other than x
            atn_crr_params = np.zeros([a, b, 3])
            atn_crr_params = atn_crr_params.reshape([a * b, 3])

            hist_bounds = np.arange(altitude_min, altitude_max, bin_band)
            idxs = np.digitize(altitudes_all, hist_bounds)
            altitudes_ret = []
            each_bin_image_list = []
            tmp_altitude_sample = 0.0
            message = 'start calculating histogram ' + \
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # print('sampling_method', sampling_method)
            for idx_bin in trange(1, hist_bounds.size, ascii=True,
                                  desc=message):
                tmp_altitudes = altitudes_all[np.where(idxs == idx_bin)]
                if len(tmp_altitudes) > min_sample_per_bin:
                    # calculate sample image in this bin
                    tmp_idx = np.where(idxs == idx_bin)[0]
                    if len(tmp_idx) > max_sample_per_bin:
                        tmp_idx = random.sample(list(tmp_idx),
                                                max_sample_per_bin)
                        tmp_altitudes = altitudes_all[tmp_idx]

                    tmp_bin_imgs = memmap_raw[tmp_idx]
                    
                    # calculate sample image of current bin
                    tmp_bin_img_sample = np.zeros((a, b), np.float32)

                    if sampling_method == 'mean':
                        tmp_bin_img_sample = np.mean(tmp_bin_imgs, axis=0)
                        tmp_altitude_sample = np.mean(tmp_altitudes)


                    elif sampling_method == 'mean_trimmed':
                        #     TOOD implement trimmed mean and std
                        tmp_bin_img_sample, dummy = calc_img_mean_and_std_trimmed(
                            tmp_bin_imgs, trim_ratio, calc_std=False,
                            effective_index=-1)
                        tmp_altitude_sample = np.mean(tmp_altitudes)


                    elif sampling_method == 'median':
                    # else:
                        tmp_bin_img_sample = np.median(tmp_bin_imgs, axis=0)
                        tmp_altitude_sample = np.mean(
                            tmp_altitudes)
                        # altitude value is calculated as mean because it has less varieance.


                    del tmp_bin_imgs

                    each_bin_image_list.append(tmp_bin_img_sample)

                    # print('added altitude value', tmp_altitude_sample)
                    altitudes_ret.append(tmp_altitude_sample)

            imgs_for_calc_atn = np.array(each_bin_image_list)

            imgs_for_calc_atn = imgs_for_calc_atn.reshape(
                [len(each_bin_image_list), a * b])
            altitudes_for_calc_atn = altitudes_ret

            Console.info('start curve fitting', datetime.datetime.now())
            if init_atn_crr_params_path is not None:
                initial_atn_crr_params = np.load(init_atn_crr_params_path)
                initial_atn_crr_params = initial_atn_crr_params.reshape(
                    [a * b, 3])

                # all pixels
                results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
                    [joblib.delayed(optim_exp_curve_param_with_init)(
                        altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel],
                        initial_atn_crr_params[i_pixel, :]) for i_pixel
                        in
                        range(a * b)])
                atn_crr_params = np.array(results)


            else:
                results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
                    [joblib.delayed(optim_exp_curve_param_log_transform)(
                        altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel])
                        for i_pixel in range(a * b)])

                atn_crr_params = np.array(results)


            atn_crr_params = atn_crr_params.reshape([a, b, 3])

            if calculated_atn_crr_params_path is None:
                calculated_atn_crr_params_path = dir_path_image_crr_params / 'atn_crr_params.npy'

            # visualise attenuation parameters
            outpath = calculated_atn_crr_params_path.parent
            if not outpath.exists():
                outpath.mkdir(parents=True)

            np.save(str(calculated_atn_crr_params_path), atn_crr_params)
            Console.info('atn_crr_params has been saved to',
                         calculated_atn_crr_params_path,
                         datetime.datetime.now())

            save_atn_crr_params_png(outpath, atn_crr_params)

            # apply attenuation correction parameters to raw images in memmap
            gain = calc_attenuation_correction_gain(target_altitude,
                                                    atn_crr_params)
            message = 'applying attenuation correction to raw ' + \
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i_img in trange(memmap_raw.shape[0], ascii=True, desc=message):
                # memmap data can not be updated in joblib .
                memmap_raw[i_img, ...] = apply_atn_crr_2_img(
                    memmap_raw[i_img, ...], altitudes_all[i_img],
                    atn_crr_params, gain)

            Console.info(
                'start calculating mean and std of attenuation corrected images',
                datetime.datetime.now(), flush=True)
            img_mean_atn_crr, img_std_atn_crr = calc_img_mean_and_std_trimmed(
                memmap_raw, trim_ratio, calc_std=True,
                effective_index=idx_effective_data)

            dirpath_img_mean_atn_crr = dir_path_image_crr_params / 'bayer_img_mean_atn_crr'
            dirpath_img_std_atn_crr = dir_path_image_crr_params / 'bayer_img_std_atn_crr'
            np.save(str(dirpath_img_mean_atn_crr), img_mean_atn_crr)
            np.save(str(dirpath_img_std_atn_crr), img_std_atn_crr)

            # visualize mean and std images
            list_dirpath = [dirpath_img_mean_atn_crr, dirpath_img_std_atn_crr]
            list_img = [img_mean_atn_crr, img_std_atn_crr]
            for i_img in range(len(list_img)):
                save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

            # convert bayer_file_path from absolute path to filename
            for i_bayer_file in range(len(bayer_filelist)):
                tmp_filepath = src_filelist[i_bayer_file]
                file_stem = get_processed_folder(tmp_filepath).stem
                bayer_filelist[i_bayer_file] = str(file_stem + '.npy')
                # bayer_filelist[i_bayer_file] = '..' / bayer_filelist[i_bayer_file].relative_to(dirpath.parent)

            # save file list includes altitude and filepath of bayer image
            file_list_name = dir_path_image_crr_params / 'filelist.csv'
            df_all = pd.concat([df_all, pd.DataFrame(
                bayer_filelist, columns=['bayer file'])], axis=1)
            df_all.to_csv(file_list_name)

            dict_cfg = {
                'src_filelist_path': src_filelist_path,
                'label_raw_file': label_raw_file,
                'label_altitude': label_altitude,
                'altitude_min': altitude_min,
                'altitude_max': altitude_max,
                'calculated_atn_crr_params_path': str(
                    calculated_atn_crr_params_path.resolve()),
                'median_filter_kernel_size': median_filter_kernel_size,
                'sampling_method': sampling_method,
                'dst_file_format': dst_file_format,
                'target_altitude': target_altitude,
                'src_file_format': src_file_format,
                'bin_band': bin_band,
                'min_sample_per_bin': min_sample_per_bin,
                'max_sample_per_bin': max_sample_per_bin
            }

            cfg_filepath = dir_path_image_crr_params / 'config.yaml'
            with cfg_filepath.open('w') as cfg_file:
                yaml.dump(dict_cfg, cfg_file, default_flow_style=False)
            Console.info('Done. Configurations are saved to ',
                         cfg_filepath, datetime.datetime.now())

            del memmap_raw
            
            path_parent = Path(path).parents[4]

            for file_name in Path(path_parent).glob('*.map'):
                Path(file_name).unlink()
            

            Console.info('#########.......Parse is completed ........#########')


    if format_ == 'seaxerocks_3':
        camera_iterations = 2
        for i in range(camera_iterations):
            if i is 0:
                camera = config_.config.camera_1
            elif i is 1:
                camera = config_.config.camera_2
            
            src_file_format = 'raw'
           
            calculated_atn_crr_params_path = None

            dst_file_format = config_.output.dst_file_format
            joblib_verbose = 3
            init_atn_crr_params_path = None
            trim_ratio = 0.2

            # load src file or target files to data frame
            src_filelist_path = None
            if src_filelist_path is not None:
                df_all = pd.read_csv(str(src_filelist_path))
                if src_file_format == 'tif' or src_file_format == 'tif':
                    # for tunasand camera, left camera (LC) or right camera (RC) should be selected
                    if camera_lr == 'LC':
                        df_all = df_all.query(
                            'Imagenumber.str.contains("LC")', engine='python')
                    elif camera_lr == 'RC':
                        df_all = df_all.query(
                            'Imagenumber.str.contains("RC")', engine='python')

            else:
                if camera_format == 'seaxerocks_3':
                    img_p = mission.image.cameras_0.get('path')

                    if camera in img_p:
                        img_path = img_p
                        camera_serial = mission.image.cameras_0.get('name')
                    else:
                        img_p = mission.image.cameras_1.get('path')
                        if camera in img_p:
                            img_path = img_p
                            camera_serial = mission.image.cameras_1.get('name')
                        else:
                            img_p = mission.image.cameras_2.get('path')
                            if camera in img_p:
                                img_path = img_p
                                camera_serial = mission.image.cameras_2.get(
                                    'name')
                            else:
                                print(
                                    'Mission yaml file does not have path to camera: ',
                                    camera)
                                continue
                else:
                    img_path = mission.image.cameras_0.get('path')
                auv_nav_filepath = Path(config_.config.auv_nav_path).resolve()
                src_file_dirpath = path_raw / img_path
                print('SRC:', src_file_dirpath)
                if camera_format == 'seaxerocks_3':
                    csv_path = 'csv/dead_reckoning/auv_dr_' + camera_serial + '.csv'
                else:
                    csv_path = 'csv/dead_reckoning/auv_dr_' + camera_lr + '.csv'
                # auv_nav_filepath = path_processed / anf
                auv_nav_filepath = auv_nav_filepath / csv_path

                df_all = pd.read_csv(auv_nav_filepath,
                                     dtype={'Imagenumber': object})
                raw_file_list = [None] * len(df_all)
                for i_file in range(len(raw_file_list)):
                    if src_file_format == 'raw':
                        raw_file_list[i_file] = src_file_dirpath / str(
                            df_all['Imagenumber'][i_file].zfill(7) + '.raw')
                    elif src_file_format == 'tif':
                        raw_file_list[i_file] = src_file_dirpath / \
                                                df_all['Imagenumber'][i_file]
                    else:
                        Console.error('src_file_format:', src_file_format,
                                      'is incorrect.')
                        return
                df_all = pd.concat([df_all, pd.DataFrame(
                    raw_file_list, columns=[label_raw_file])], axis=1)
            src_filelist = df_all[label_raw_file]

            # for attenuation correction
            target_altitude = None  # load_data.get('target_altitude', None)
            curve_fit_trial_num = 1  # load_data.get('curve_fit_trial_num', 1)
            # attenuation_correction_parameter_file_path = load_data.get('attenuation_correction_parameter_file_path', None)
            bin_band = 0.1  # load_data.get('bin_band', 0.1)  # 0.1m for AE2000
            min_sample_per_bin = 5  # load_data.get('min_sample_per_bin', 5)
            max_sample_per_bin = 100  # load_data.get('max_sample_per_bin', 100)
            # load_data.get('median_filter_kernel_size', 1)
            median_filter_kernel_size = 1

            # remove too low or too high altitude file and too small file size file
            altitudes_all = df_all[label_altitude].values
            match_count = 0
            
            # check if altitudes match with min and max provided in correct_images.yaml
            for i in range(len(altitudes_all)):
                if altitudes_all[i] <= altitude_max and altitudes_all[i] >= altitude_min:
                    match_count = match_count + 1
            if match_count < 1:
                Console.quit(
                    'altitude values in dive dataset do not match with minimum and maximum altitude provided in correct_images.yaml')
            else:
                idx_effective_data = np.where(
                (altitudes_all >= altitude_min) & (
                        altitudes_all <= altitude_max))
            
                # configure output file path
            dirpath = src_filelist[0].parent
            dirpath = get_processed_folder(dirpath)
         
            dirpath = dirpath / 'attenuation_correction'
            if not dirpath.exists():
                dirpath.mkdir(parents=True)
            dirpath_atn_crr = dirpath / 'tmp_atn_crr'
            bayer_folder_name = 'bayer' + camera
            dirpath_bayer = dirpath / bayer_folder_name
            if not dirpath_bayer.exists():
                dirpath_bayer.mkdir(parents=True)

            # file path of output image data
            dst_filelist = [None] * len(df_all)
            bayer_filelist = [None] * len(df_all)
            atn_crr_filelist = [None] * len(df_all)
            for i_dst_file in range(len(dst_filelist)):
                tmp_filepath = src_filelist[i_dst_file]
                file_stem = get_processed_folder(tmp_filepath).stem
                dst_filelist[i_dst_file] = dirpath / str(
                    file_stem + '.' + dst_file_format)
                bayer_filelist[i_dst_file] = dirpath_bayer / str(
                    file_stem + '.npy')
                atn_crr_filelist[i_dst_file] = dirpath_atn_crr / str(
                    file_stem + '.npy')

            # file path of metadata
            params_folder_name = 'params_' + camera
            dir_path_image_crr_params = dst_filelist[
                                            0].parent / params_folder_name
            if not dir_path_image_crr_params.exists():
                dir_path_image_crr_params.mkdir(parents=True)
                Console.info(
                    'code will compute correction parameters for this Camera for first time.')
            else:
                print(dir_path_image_crr_params)
                if force is True:
                    Console.warn(
                        'Attenuation correction parameters already exist.')
                    Console.warn('Code will overwrite existing parameters.')

                else:
                    Console.warn(
                        'Code will quit - correction parameters already exist.')
                    Console.warn(
                        'Run correct_images with [parse] [-F] option for overwriting existing correction parameters.')
                    sys.exit()

            if calculated_atn_crr_params_path is None:
                calculated_atn_crr_params_path = dir_path_image_crr_params / 'atn_crr_params.npy'

            file_list_raw = df_all[label_raw_file].values.tolist()

            if src_file_format == 'raw':
                # xviii camera
                a, b = 1024, 1280
                # developing .raw data to bayer data of uint32 numpy array.
                Console.info('start loading bayer images', len(file_list_raw),
                             'files to', dirpath_bayer,
                             datetime.datetime.now())
                src_file_list_not_exist = []
                bayer_file_list_not_exsit = []
                for idx_raw in range(len(file_list_raw)):
                    if not bayer_filelist[idx_raw].exists():
                        src_file_list_not_exist.append(file_list_raw[idx_raw])
                        bayer_file_list_not_exsit.append(
                            bayer_filelist[idx_raw])

                Console.info(
                    len(file_list_raw) - len(bayer_file_list_not_exsit),
                    'files have already existed.')

                task_num = 100
                num_loop = int(len(bayer_file_list_not_exsit) / task_num) + 1
                start_idx = 0
                idx_total = start_idx
                end_idx = 0
                while start_idx < len(bayer_file_list_not_exsit):
                    # for debug
                    #     break

                    Console.info(
                        'processing load_xviii_bayer_from_binary',
                        int(start_idx / task_num) + 1, '/', num_loop,
                        'of total',
                        len(bayer_file_list_not_exsit), 'files',
                        datetime.datetime.now(), flush=True)

                    end_idx = start_idx + task_num
                    if end_idx > len(bayer_file_list_not_exsit):
                        end_idx = len(bayer_file_list_not_exsit)

                    raw_img_for_size = np.fromfile(str(
                        src_file_list_not_exist[start_idx]), dtype=np.uint8)
                    arg_bayer_img = np.zeros(
                        (end_idx - start_idx, raw_img_for_size.shape[0]),
                        dtype=raw_img_for_size.dtype)
                    for idx_raw in range(start_idx, end_idx):
                        arg_bayer_img[idx_raw - start_idx, :] = np.fromfile(
                            str(src_file_list_not_exist[idx_raw]),
                            dtype=raw_img_for_size.dtype)

                    results = joblib.Parallel(n_jobs=-2,
                                              verbose=joblib_verbose)(
                        [joblib.delayed(load_xviii_bayer_from_binary)(
                            arg_bayer_img[idx_arg, :]) for idx_arg in
                            range(end_idx - start_idx)])

                    for idx_raw in range(start_idx, end_idx):
                        np.save(bayer_file_list_not_exsit[idx_raw],
                                results[idx_raw - start_idx])

                    start_idx = end_idx

            elif src_file_format == 'tif' or src_file_format == 'tiff':
                # unaggi camera
                Console.info('start loading tif images', len(file_list_raw),
                             'files to', dirpath_bayer,
                             datetime.datetime.now())
                src_file_list_not_exist = []
                bayer_file_list_not_exsit = []
                for idx_raw in range(len(file_list_raw)):
                    if not bayer_filelist[idx_raw].exists():
                        src_file_list_not_exist.append(file_list_raw[idx_raw])
                        bayer_file_list_not_exsit.append(
                            bayer_filelist[idx_raw])

                Console.info(
                    len(file_list_raw) - len(bayer_file_list_not_exsit),
                    'files have already existed.')

                tmp_tif_for_size = imageio.imread(file_list_raw[0])
                a = tmp_tif_for_size.shape[0]
                b = tmp_tif_for_size.shape[1]

                for i_file_not_exist in range(len(src_file_list_not_exist)):
                    tmp_tif = imageio.imread(
                        src_file_list_not_exist[i_file_not_exist])
                    tmp_npy = np.zeros([a, b], np.uint16)
                    tmp_npy[:, :] = np.array(tmp_tif, np.uint16)
                    np.save(bayer_file_list_not_exsit[i_file_not_exist],
                            tmp_npy)

            # caluculate attenuation correction parameter
            if target_altitude is None:
                target_altitude = float(
                    np.mean(altitudes_all[idx_effective_data]))

            # memmap is created at local directory
            file_name_memmap_raw, memmap_raw = load_memmap_from_npy_filelist(
                bayer_filelist)
            # TODO for debug. read existing file.
            # file_name_memmap_raw = '/home/ty1u18/PycharmProjects/correct_images/memmap_raw_img_d953693d-47aa-403d-9ece-f8f3b19d8b98.map'
            # memmap_raw = np.memmap(file_name_memmap_raw, np.float32, 'r', shape=(8619, 2056, 2464))

            Console.info('start calculate mean and std of raw img',
                         datetime.datetime.now())

            img_mean_raw, img_std_raw = \
                calc_img_mean_and_std_trimmed(memmap_raw, trim_ratio,
                                              calc_std=True,
                                              effective_index=idx_effective_data)
            # TODO for debug. not calculate raw
            # img_mean_raw = 0
            # img_std_raw = 0

            dirpath_img_mean_raw = dir_path_image_crr_params / 'bayer_img_mean_raw'
            dirpath_img_std_raw = dir_path_image_crr_params / 'bayer_img_std_raw'
            np.save(str(dirpath_img_mean_raw), img_mean_raw)
            np.save(str(dirpath_img_std_raw), img_std_raw)

            list_dirpath = [dirpath_img_mean_raw, dirpath_img_std_raw]
            list_img = [img_mean_raw, img_std_raw]
            for i_img in range(len(list_img)):
                save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

            # calculate regression parameters for all pixels and channels
            Console.info('start attenuation correction parameter calculation.')

            # 3 is number of parameter in exp_curve other than x
            atn_crr_params = np.zeros([a, b, 3])
            atn_crr_params = atn_crr_params.reshape([a * b, 3])

            hist_bounds = np.arange(altitude_min, altitude_max, bin_band)
            idxs = np.digitize(altitudes_all, hist_bounds)
            altitudes_ret = []
            each_bin_image_list = []
            tmp_altitude_sample = 0.0
            message = 'start calculating histogram ' + \
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # print('sampling_method', sampling_method)
            for idx_bin in trange(1, hist_bounds.size, ascii=True,
                                  desc=message):
                tmp_altitudes = altitudes_all[np.where(idxs == idx_bin)]
                if len(tmp_altitudes) > min_sample_per_bin:
                    # calculate sample image in this bin
                    tmp_idx = np.where(idxs == idx_bin)[0]
                    if len(tmp_idx) > max_sample_per_bin:
                        tmp_idx = random.sample(list(tmp_idx),
                                                max_sample_per_bin)
                        tmp_altitudes = altitudes_all[tmp_idx]

                    tmp_bin_imgs = memmap_raw[tmp_idx]
                    # calculate sample image of current bin
                    tmp_bin_img_sample = np.zeros((a, b), np.float32)

                    if sampling_method == 'mean':
                        tmp_bin_img_sample = np.mean(tmp_bin_imgs, axis=0)
                        tmp_altitude_sample = np.mean(tmp_altitudes)


                    elif sampling_method == 'mean_trimmed':
                        #     TOOD implement trimmed mean and std
                        tmp_bin_img_sample, dummy = calc_img_mean_and_std_trimmed(
                            tmp_bin_imgs, trim_ratio, calc_std=False,
                            effective_index=-1)
                        tmp_altitude_sample = np.mean(tmp_altitudes)


                    elif sampling_method == 'median':
                    # else:
                        tmp_bin_img_sample = np.median(tmp_bin_imgs, axis=0)
                        tmp_altitude_sample = np.mean(
                            tmp_altitudes)
                        # altitude value is calculated as mean because it has less varieance.


                    del tmp_bin_imgs

                    each_bin_image_list.append(tmp_bin_img_sample)

                    # print('added altitude value', tmp_altitude_sample)
                    altitudes_ret.append(tmp_altitude_sample)

            imgs_for_calc_atn = np.array(each_bin_image_list)

            imgs_for_calc_atn = imgs_for_calc_atn.reshape(
                [len(each_bin_image_list), a * b])
            altitudes_for_calc_atn = altitudes_ret

            Console.info('start curve fitting', datetime.datetime.now())
            if init_atn_crr_params_path is not None:
                initial_atn_crr_params = np.load(init_atn_crr_params_path)
                initial_atn_crr_params = initial_atn_crr_params.reshape(
                    [a * b, 3])

                # all pixels
                results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
                    [joblib.delayed(optim_exp_curve_param_with_init)(
                        altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel],
                        initial_atn_crr_params[i_pixel, :]) for i_pixel
                        in
                        range(a * b)])
                atn_crr_params = np.array(results)

            else:
                results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
                    [joblib.delayed(optim_exp_curve_param_log_transform)(
                        altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel])
                        for i_pixel in range(a * b)])

                atn_crr_params = np.array(results)

            atn_crr_params = atn_crr_params.reshape([a, b, 3])

            if calculated_atn_crr_params_path is None:
                calculated_atn_crr_params_path = dir_path_image_crr_params / 'atn_crr_params.npy'

            # visualise attenuation parameters
            outpath = calculated_atn_crr_params_path.parent
            if not outpath.exists():
                outpath.mkdir(parents=True)

            np.save(str(calculated_atn_crr_params_path), atn_crr_params)
            Console.info('atn_crr_params has been saved to',
                         calculated_atn_crr_params_path,
                         datetime.datetime.now())

            save_atn_crr_params_png(outpath, atn_crr_params)

            # apply median filter to attenuation parameter
            # if median_filter_kernel_size != 1:
            # atn_crr_params = filter_atn_parm_median(atn_crr_params, median_filter_kernel_size)

            # apply attenuation correction parameters to raw images in memmap
            gain = calc_attenuation_correction_gain(target_altitude,
                                                    atn_crr_params)
            message = 'applying attenuation correction to raw ' + \
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i_img in trange(memmap_raw.shape[0], ascii=True, desc=message):
                # memmap data can not be updated in joblib .
                memmap_raw[i_img, ...] = apply_atn_crr_2_img(
                    memmap_raw[i_img, ...], altitudes_all[i_img],
                    atn_crr_params, gain)

            Console.info(
                'start calculating mean and std of attenuation corrected images',
                datetime.datetime.now(), flush=True)
            img_mean_atn_crr, img_std_atn_crr = calc_img_mean_and_std_trimmed(
                memmap_raw, trim_ratio, calc_std=True,
                effective_index=idx_effective_data)

            dirpath_img_mean_atn_crr = dir_path_image_crr_params / 'bayer_img_mean_atn_crr'
            dirpath_img_std_atn_crr = dir_path_image_crr_params / 'bayer_img_std_atn_crr'
            np.save(str(dirpath_img_mean_atn_crr), img_mean_atn_crr)
            np.save(str(dirpath_img_std_atn_crr), img_std_atn_crr)

            # visualize mean and std images
            list_dirpath = [dirpath_img_mean_atn_crr, dirpath_img_std_atn_crr]
            list_img = [img_mean_atn_crr, img_std_atn_crr]
            for i_img in range(len(list_img)):
                save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

            # convert bayer_file_path from absolute path to filename
            for i_bayer_file in range(len(bayer_filelist)):
                tmp_filepath = src_filelist[i_bayer_file]
                file_stem = get_processed_folder(tmp_filepath).stem
                bayer_filelist[i_bayer_file] = str(file_stem + '.npy')
                # bayer_filelist[i_bayer_file] = '..' / bayer_filelist[i_bayer_file].relative_to(dirpath.parent)

            # save file list includes altitude and filepath of bayer image
            file_list_name = dir_path_image_crr_params / 'filelist.csv'
            df_all = pd.concat([df_all, pd.DataFrame(
                bayer_filelist, columns=['bayer file'])], axis=1)
            df_all.to_csv(file_list_name)

            dict_cfg = {
                'src_filelist_path': src_filelist_path,
                'label_raw_file': label_raw_file,
                'label_altitude': label_altitude,
                'altitude_min': altitude_min,
                'altitude_max': altitude_max,
                'calculated_atn_crr_params_path': str(
                    calculated_atn_crr_params_path.resolve()),
                'median_filter_kernel_size': median_filter_kernel_size,
                'sampling_method': sampling_method,
                'dst_file_format': dst_file_format,
                'target_altitude': target_altitude,
                'curve_fit_trial_num': curve_fit_trial_num,
                'src_file_format': src_file_format,
                'bin_band': bin_band,
                'min_sample_per_bin': min_sample_per_bin,
                'max_sample_per_bin': max_sample_per_bin
            }

            cfg_filepath = dir_path_image_crr_params / 'config.yaml'
            with cfg_filepath.open('w') as cfg_file:
                yaml.dump(dict_cfg, cfg_file, default_flow_style=False)
            Console.info('Done. Configurations are saved to ',
                         cfg_filepath, datetime.datetime.now())

            del memmap_raw
            
            path_parent = Path(path).parents[4]

            for file_name in Path(path_parent).glob('*.map'):
                Path(file_name).unlink()
            

            Console.info('#########.......Parse is completed ........#########')
    
    if format_ == 'acfr_standard':
        camera_iterations = 2
        for i in range(camera_iterations):
            if i is 0:
                camera = config_.config.camera_1
            elif i is 1:
                camera = config_.config.camera_2
            
            src_file_format = 'tif'
            
            camera_lr = camera
            calculated_atn_crr_params_path = None
            dst_file_format = config_.output.dst_file_format
            joblib_verbose = 3
            init_atn_crr_params_path = None
            trim_ratio = 0.2

            # load src file or target files to data frame
            src_filelist_path = None
            if src_filelist_path is not None:
                df_all = pd.read_csv(str(src_filelist_path))
                if src_file_format == 'tif' or src_file_format == 'tif':
                    # for tunasand camera, left camera (LC) or right camera (RC) should be selected
                    if camera_lr == 'LC':
                        df_all = df_all.query(
                            'Imagenumber.str.contains("LC")', engine='python')
                    elif camera_lr == 'RC':
                        df_all = df_all.query(
                            'Imagenumber.str.contains("RC")', engine='python')

            else:
                if camera_format == 'seaxerocks_3':
                    img_p = mission.image.cameras_0.get('path')

                    if camera in img_p:
                        img_path = img_p
                        camera_serial = mission.image.cameras_0.get('name')
                    else:
                        img_p = mission.image.cameras_1.get('path')
                        if camera in img_p:
                            img_path = img_p
                            camera_serial = mission.image.cameras_1.get('name')
                        else:
                            img_p = mission.image.cameras_2.get('path')
                            if camera in img_p:
                                img_path = img_p
                                camera_serial = mission.image.cameras_2.get(
                                    'name')
                            else:
                                print(
                                    'Mission yaml file does not have path to camera: ',
                                    camera)
                                sys.exit()
                else:
                    img_path = mission.image.cameras_0.get('path')
                auv_nav_filepath = Path(config_.config.auv_nav_path).resolve()
                src_file_dirpath = path_raw / img_path
                if camera_format == 'seaxerocks_3':
                    csv_path = 'csv/dead_reckoning/auv_dr_' + camera_serial + '.csv'
                else:
                    csv_path = 'csv/dead_reckoning/auv_dr_' + camera_lr + '.csv'
                # auv_nav_filepath = path_processed / anf
                auv_nav_filepath = auv_nav_filepath / csv_path

                df_all = pd.read_csv(auv_nav_filepath,
                                     dtype={'Imagenumber': object})
                raw_file_list = [None] * len(df_all)
                for i_file in range(len(raw_file_list)):
                    if src_file_format == 'raw':
                        raw_file_list[i_file] = src_file_dirpath / str(
                            df_all['Imagenumber'][i_file].zfill(7) + '.raw')
                    elif src_file_format == 'tif':
                        raw_file_list[i_file] = src_file_dirpath / \
                                                df_all['Imagenumber'][i_file]
                    else:
                        Console.error('src_file_format:', src_file_format,
                                      'is incorrect.')
                        return
                df_all = pd.concat([df_all, pd.DataFrame(
                    raw_file_list, columns=[label_raw_file])], axis=1)
            src_filelist = df_all[label_raw_file]

            # for attenuation correction
            target_altitude = None  # load_data.get('target_altitude', None)
            curve_fit_trial_num = 1  # load_data.get('curve_fit_trial_num', 1)
            # attenuation_correction_parameter_file_path = load_data.get('attenuation_correction_parameter_file_path', None)
            bin_band = 0.1  # load_data.get('bin_band', 0.1)  # 0.1m for AE2000
            min_sample_per_bin = 5  # load_data.get('min_sample_per_bin', 5)
            max_sample_per_bin = 100  # load_data.get('max_sample_per_bin', 100)
            # load_data.get('median_filter_kernel_size', 1)
            median_filter_kernel_size = 1

            # remove too low or too high altitude file and too small file size file
            altitudes_all = df_all[label_altitude].values
            match_count = 0
            
            # check if altitudes match with min and max provided in correct_images.yaml
            for i in range(len(altitudes_all)):
                if altitudes_all[i] <= altitude_max and altitudes_all[i] >= altitude_min:
                    match_count = match_count + 1
            if match_count < 1:
                Console.quit(
                    'altitude values in dive dataset do not match with minimum and maximum altitude provided in correct_images.yaml')
            else:
                idx_effective_data = np.where(
                (altitudes_all >= altitude_min) & (
                        altitudes_all <= altitude_max))

            # configure output file path
            dirpath = src_filelist[0].parent
            dirpath = get_processed_folder(dirpath)
            dirpath = dirpath / 'attenuation_correction'
            if not dirpath.exists():
                dirpath.mkdir(parents=True)
            dirpath_atn_crr = dirpath / 'tmp_atn_crr'
            bayer_folder_name = 'bayer' + camera
            dirpath_bayer = dirpath / bayer_folder_name
            if not dirpath_bayer.exists():
                dirpath_bayer.mkdir(parents=True)

            # file path of output image data
            dst_filelist = [None] * len(df_all)
            bayer_filelist = [None] * len(df_all)
            atn_crr_filelist = [None] * len(df_all)
            for i_dst_file in range(len(dst_filelist)):
                tmp_filepath = src_filelist[i_dst_file]
                file_stem = get_processed_folder(tmp_filepath).stem
                dst_filelist[i_dst_file] = dirpath / str(
                    file_stem + '.' + dst_file_format)
                bayer_filelist[i_dst_file] = dirpath_bayer / str(
                    file_stem + '.npy')
                atn_crr_filelist[i_dst_file] = dirpath_atn_crr / str(
                    file_stem + '.npy')

            # file path of metadata
            params_folder_name = 'params_' + camera
            dir_path_image_crr_params = dst_filelist[
                                            0].parent / params_folder_name
            if not dir_path_image_crr_params.exists():
                dir_path_image_crr_params.mkdir(parents=True)
                Console.info(
                    'code will compute correction parameters for this Camera for first time.')
            else:
                print(dir_path_image_crr_params)
                if force is True:
                    Console.warn(
                        'Attenuation correction parameters already exist.')
                    Console.warn('Code will overwrite existing parameters.')

                else:
                    Console.warn(
                        'Code will quit - correction parameters already exist.')
                    Console.warn(
                        'Run correct_images with [parse] [-F] option for overwriting existing correction parameters.')
                    sys.exit()

            if calculated_atn_crr_params_path is None:
                calculated_atn_crr_params_path = dir_path_image_crr_params / 'atn_crr_params.npy'

            file_list_raw = df_all[label_raw_file].values.tolist()

            if src_file_format == 'raw':
                # xviii camera
                a, b = 1024, 1280
                # developing .raw data to bayer data of uint32 numpy array.
                Console.info('start loading bayer images', len(file_list_raw),
                             'files to', dirpath_bayer,
                             datetime.datetime.now())
                src_file_list_not_exist = []
                bayer_file_list_not_exsit = []
                for idx_raw in range(len(file_list_raw)):
                    if not bayer_filelist[idx_raw].exists():
                        src_file_list_not_exist.append(file_list_raw[idx_raw])
                        bayer_file_list_not_exsit.append(
                            bayer_filelist[idx_raw])

                Console.info(
                    len(file_list_raw) - len(bayer_file_list_not_exsit),
                    'files have already existed.')

                task_num = 100
                num_loop = int(len(bayer_file_list_not_exsit) / task_num) + 1
                start_idx = 0
                idx_total = start_idx
                end_idx = 0
                while start_idx < len(bayer_file_list_not_exsit):

                    Console.info(
                        'processing load_xviii_bayer_from_binary',
                        int(start_idx / task_num) + 1, '/', num_loop,
                        'of total',
                        len(bayer_file_list_not_exsit), 'files',
                        datetime.datetime.now(), flush=True)

                    end_idx = start_idx + task_num
                    if end_idx > len(bayer_file_list_not_exsit):
                        end_idx = len(bayer_file_list_not_exsit)

                    raw_img_for_size = np.fromfile(str(
                        src_file_list_not_exist[start_idx]), dtype=np.uint8)
                    arg_bayer_img = np.zeros(
                        (end_idx - start_idx, raw_img_for_size.shape[0]),
                        dtype=raw_img_for_size.dtype)
                    for idx_raw in range(start_idx, end_idx):
                        arg_bayer_img[idx_raw - start_idx, :] = np.fromfile(
                            str(src_file_list_not_exist[idx_raw]),
                            dtype=raw_img_for_size.dtype)

                    results = joblib.Parallel(n_jobs=-2,
                                              verbose=joblib_verbose)(
                        [joblib.delayed(load_xviii_bayer_from_binary)(
                            arg_bayer_img[idx_arg, :]) for idx_arg in
                            range(end_idx - start_idx)])

                    for idx_raw in range(start_idx, end_idx):
                        np.save(bayer_file_list_not_exsit[idx_raw],
                                results[idx_raw - start_idx])

                    start_idx = end_idx

            elif src_file_format == 'tif' or src_file_format == 'tiff':
                # unaggi camera
                Console.info('start loading tif images', len(file_list_raw),
                             'files to', dirpath_bayer,
                             datetime.datetime.now())
                src_file_list_not_exist = []
                bayer_file_list_not_exsit = []
                for idx_raw in range(len(file_list_raw)):
                    if not bayer_filelist[idx_raw].exists():
                        src_file_list_not_exist.append(file_list_raw[idx_raw])
                        bayer_file_list_not_exsit.append(
                            bayer_filelist[idx_raw])

                Console.info(
                    len(file_list_raw) - len(bayer_file_list_not_exsit),
                    'files have already existed.')

                tmp_tif_for_size = imageio.imread(file_list_raw[0])
                a = tmp_tif_for_size.shape[0]
                b = tmp_tif_for_size.shape[1]

                for i_file_not_exist in range(len(src_file_list_not_exist)):
                    tmp_tif = imageio.imread(
                        src_file_list_not_exist[i_file_not_exist])
                    tmp_npy = np.zeros([a, b], np.uint16)
                    tmp_npy[:, :] = np.array(tmp_tif, np.uint16)
                    np.save(bayer_file_list_not_exsit[i_file_not_exist],
                            tmp_npy)

            # calculate attenuation correction parameter
            if target_altitude is None:
                target_altitude = float(
                    np.mean(altitudes_all[idx_effective_data]))

            # memmap is created at local directory
            file_name_memmap_raw, memmap_raw = load_memmap_from_npy_filelist(
                bayer_filelist)

            Console.info('start calculate mean and std of raw img',
                         datetime.datetime.now())

            img_mean_raw, img_std_raw = \
                calc_img_mean_and_std_trimmed(memmap_raw, trim_ratio,
                                              calc_std=True,
                                              effective_index=idx_effective_data)

            dirpath_img_mean_raw = dir_path_image_crr_params / 'bayer_img_mean_raw'
            dirpath_img_std_raw = dir_path_image_crr_params / 'bayer_img_std_raw'
            np.save(str(dirpath_img_mean_raw), img_mean_raw)
            np.save(str(dirpath_img_std_raw), img_std_raw)

            list_dirpath = [dirpath_img_mean_raw, dirpath_img_std_raw]
            list_img = [img_mean_raw, img_std_raw]
            for i_img in range(len(list_img)):
                save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

            # calculate regression parameters for all pixels and channels
            Console.info('start attenuation correction parameter calculation.')

            # 3 is number of parameter in exp_curve other than x
            atn_crr_params = np.zeros([a, b, 3])
            atn_crr_params = atn_crr_params.reshape([a * b, 3])

            hist_bounds = np.arange(altitude_min, altitude_max, bin_band)
            idxs = np.digitize(altitudes_all, hist_bounds)
            altitudes_ret = []
            each_bin_image_list = []
            tmp_altitude_sample = 0.0
            message = 'start calculating histogram ' + \
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for idx_bin in trange(1, hist_bounds.size, ascii=True,
                                  desc=message):
                tmp_altitudes = altitudes_all[np.where(idxs == idx_bin)]
                if len(tmp_altitudes) > min_sample_per_bin:
                    # calculate sample image in this bin
                    tmp_idx = np.where(idxs == idx_bin)[0]
                    if len(tmp_idx) > max_sample_per_bin:
                        tmp_idx = random.sample(list(tmp_idx),
                                                max_sample_per_bin)
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
                        tmp_bin_img_sample, dummy = calc_img_mean_and_std_trimmed(
                            tmp_bin_imgs, trim_ratio, calc_std=False,
                            effective_index=-1)
                        tmp_altitude_sample = np.mean(tmp_altitudes)

                    del tmp_bin_imgs

                    each_bin_image_list.append(tmp_bin_img_sample)
                    altitudes_ret.append(tmp_altitude_sample)

            imgs_for_calc_atn = np.array(each_bin_image_list)

            imgs_for_calc_atn = imgs_for_calc_atn.reshape(
                [len(each_bin_image_list), a * b])
            altitudes_for_calc_atn = altitudes_ret

            Console.info('start curve fitting', datetime.datetime.now())
            if init_atn_crr_params_path is not None:
                initial_atn_crr_params = np.load(init_atn_crr_params_path)
                initial_atn_crr_params = initial_atn_crr_params.reshape(
                    [a * b, 3])

                # all pixels
                results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
                    [joblib.delayed(optim_exp_curve_param_with_init)(
                        altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel],
                        initial_atn_crr_params[i_pixel, :]) for i_pixel
                        in
                        range(a * b)])
                atn_crr_params = np.array(results)

            else:
                results = joblib.Parallel(n_jobs=-2, verbose=joblib_verbose)(
                    [joblib.delayed(optim_exp_curve_param_auto_init)(
                        altitudes_for_calc_atn, imgs_for_calc_atn[:, i_pixel],
                        curve_fit_trial_num) for i_pixel in
                        range(a * b)])
                atn_crr_params = np.array(results)

            atn_crr_params = atn_crr_params.reshape([a, b, 3])

            if calculated_atn_crr_params_path is None:
                calculated_atn_crr_params_path = dir_path_image_crr_params / 'atn_crr_params.npy'

            # visualise attenuation parameters
            outpath = calculated_atn_crr_params_path.parent
            if not outpath.exists():
                outpath.mkdir(parents=True)

            np.save(str(calculated_atn_crr_params_path), atn_crr_params)
            Console.info('atn_crr_params has been saved to',
                         calculated_atn_crr_params_path,
                         datetime.datetime.now())

            save_atn_crr_params_png(outpath, atn_crr_params)

            # apply median filter to attenuation parameter
            # if median_filter_kernel_size != 1:
            # atn_crr_params = filter_atn_parm_median(atn_crr_params, median_filter_kernel_size)

            # apply attenuation correction parameters to raw images in memmap
            gain = calc_attenuation_correction_gain(target_altitude,
                                                    atn_crr_params)
            message = 'applying attenuation correction to raw ' + \
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i_img in trange(memmap_raw.shape[0], ascii=True, desc=message):
                # memmap data can not be updated in joblib .
                memmap_raw[i_img, ...] = apply_atn_crr_2_img(
                    memmap_raw[i_img, ...], altitudes_all[i_img],
                    atn_crr_params, gain)

            Console.info(
                'start calculating mean and std of attenuation corrected images',
                datetime.datetime.now(), flush=True)
            img_mean_atn_crr, img_std_atn_crr = calc_img_mean_and_std_trimmed(
                memmap_raw, trim_ratio, calc_std=True,
                effective_index=idx_effective_data)

            dirpath_img_mean_atn_crr = dir_path_image_crr_params / 'bayer_img_mean_atn_crr'
            dirpath_img_std_atn_crr = dir_path_image_crr_params / 'bayer_img_std_atn_crr'
            np.save(str(dirpath_img_mean_atn_crr), img_mean_atn_crr)
            np.save(str(dirpath_img_std_atn_crr), img_std_atn_crr)

            # visualize mean and std images
            list_dirpath = [dirpath_img_mean_atn_crr, dirpath_img_std_atn_crr]
            list_img = [img_mean_atn_crr, img_std_atn_crr]
            for i_img in range(len(list_img)):
                save_bayer_array_png(list_dirpath[i_img], list_img[i_img])

            # convert bayer_file_path from absolute path to filename
            for i_bayer_file in range(len(bayer_filelist)):
                tmp_filepath = src_filelist[i_bayer_file]
                file_stem = get_processed_folder(tmp_filepath).stem
                bayer_filelist[i_bayer_file] = str(file_stem + '.npy')
                # bayer_filelist[i_bayer_file] = '..' / bayer_filelist[i_bayer_file].relative_to(dirpath.parent)

            # save file list includes altitude and filepath of bayer image
            file_list_name = dir_path_image_crr_params / 'filelist.csv'
            df_all = pd.concat([df_all, pd.DataFrame(
                bayer_filelist, columns=['bayer file'])], axis=1)
            df_all.to_csv(file_list_name)

            dict_cfg = {
                'src_filelist_path': src_filelist_path,
                'label_raw_file': label_raw_file,
                'label_altitude': label_altitude,
                'altitude_min': altitude_min,
                'altitude_max': altitude_max,
                'calculated_atn_crr_params_path': str(
                    calculated_atn_crr_params_path.resolve()),
                'median_filter_kernel_size': median_filter_kernel_size,
                'sampling_method': sampling_method,
                'dst_file_format': dst_file_format,
                'target_altitude': target_altitude,
                'curve_fit_trial_num': curve_fit_trial_num,
                'src_file_format': src_file_format,
                'bin_band': bin_band,
                'min_sample_per_bin': min_sample_per_bin,
                'max_sample_per_bin': max_sample_per_bin
            }

            cfg_filepath = dir_path_image_crr_params / 'config.yaml'
            with cfg_filepath.open('w') as cfg_file:
                yaml.dump(dict_cfg, cfg_file, default_flow_style=False)
            Console.info('Done. Configurations are saved to ',
                         cfg_filepath, datetime.datetime.now())

            del memmap_raw
            
            path_parent = Path(path).parents[4]

            for file_name in Path(path_parent).glob('*.map'):
                Path(file_name).unlink()
            

            Console.info('#########.......Parse is completed ........#########')


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
            bayer_img[i, j] = (work[3] & 0xff) << 16 | (
                    (work[2] & 0xff) << 8) | (work[1] & 0xff)
            bayer_img[i, j + 1] = ((work[0] & 0xff) <<
                                   16) | ((work[7] & 0xff) << 8) | (
                                          work[6] & 0xff)
            bayer_img[i, j + 2] = ((work[5] & 0xff) <<
                                   16) | ((work[4] & 0xff) << 8) | (
                                          work[11] & 0xff)
            bayer_img[i, j + 3] = ((work[10] & 0xff) <<
                                   16) | ((work[9] & 0xff) << 8) | (
                                          work[8] & 0xff)
            count += 12

    return bayer_img


def calc_img_mean_and_std_trimmed(src_imgs, ratio_trimming, calc_std=True,
                                  effective_index=-1):
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

    message = 'calculating trimmed mean and std of images ' + \
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx_a in trange(a, ascii=True, desc=message):
        results = joblib.Parallel(n_jobs=-2, verbose=0)(
            [joblib.delayed(calc_mean_and_std_trimmed)(
                src_imgs[effective_index, idx_a, idx_b][0], ratio_trimming,
                calc_std) for idx_b in
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
            plt.savefig(str(dst_dirpath / filename))
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

    if not dst_dirpath.exists():
        dst_dirpath.mkdir(parents=True)

    for i_ch in range(4):
        cax = plt.matshow(result_table_vis[:, :, i_ch, ...])
        plt.colorbar(cax)
        plt.title('ch_' + str(i_ch))
        filename = 'ch_' + str(i_ch) + '.png'
        plt.savefig(str(dst_dirpath / filename))
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

    list_params_fil = [params_g1_fil,
                       params_r_fil, params_b_fil, params_g2_fil]

    params = np.zeros((params_g1.shape[0], params_g1.shape[1], 4, 3))

    for i_mos in range(len(list_params)):
        tmp_params_fil = np.zeros(
            list_params[i_mos].shape, list_params[i_mos].dtype)
        for i in range(np.size(list_params[i_mos], axis=2)):
            for j in range(np.size(list_params[i_mos], axis=3)):
                list_params_fil[i_mos][:, :, i, j] = filters.median_filter(
                    list_params[i_mos][:, :, i, j],
                    (kernel_size, kernel_size))

    ret = np.zeros(src_atn_param.shape, src_atn_param.dtype)
    ret[0::2, 0::2, :, :] = params_g1_fil
    ret[0::2, 1::2, :, :] = params_r_fil
    ret[1::2, 0::2, :, :] = params_b_fil
    ret[1::2, 1::2, :, :] = params_g2_fil

    return ret


def load_memmap_from_npy_filelist(list_raw_files):
    filename_images_map = 'memmap_raw_img_' + str(uuid.uuid4()) + '.map'

    I = np.load(str(list_raw_files[0]))
    list_shape = [len(list_raw_files)]
    list_shape = list_shape + list(I.shape)
    # memmap = np.memmap(filename=filename_images_map, mode='w+', shape=tuple(list_shape),dtype=I.dtype)

    if 'red' in socket.gethostname():
        #     if executed on HPC, the raw data are loaded on RAM
        try:
            memmap = np.zeros(shape=tuple(list_shape), dtype=np.float32)
            dummy_memmap = np.memmap(filename=filename_images_map, mode='w+',
                                     shape=(1, 1),
                                     dtype=np.float32)
            message = 'loading raw data to RAM ' + \
                      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except MemoryError:
            # use memmap instead of RAM when MemoryError is caught.
            memmap = np.memmap(filename=filename_images_map, mode='w+',
                               shape=tuple(list_shape),
                               dtype=np.float32)
            message = 'loading raw data to memmap ' + \
                      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    else:
        memmap = np.memmap(filename=filename_images_map, mode='w+',
                           shape=tuple(list_shape),
                           dtype=np.float32)  # loaded as float32 because this memmap used for restoring attenuation corrected data later.
        message = 'loading raw data to memmap ' + \
                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
    tmp_params = optimize.least_squares(residual_exp_curve, init_params,
                                        loss=loss,
                                        method=method, args=(
            altitudes, intensities),
                                        bounds=(
                                            [1, -np.inf, 0],
                                            [np.inf, 0, np.inf]))
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

        b = (np.log((intensities[idx_0] - c) / (intensities[idx_1] - c))
             ) / (altitudes[idx_0] - altitudes[idx_1])
        a = (intensities[idx_1] - c) / np.exp(b * altitudes[idx_1])
        if a < 1 or b > 0 or np.isnan(a) or np.isnan(b):
            if flag_already_calculated == True:
                continue
            else:
                flag_already_calculated = True
            a = 1.01
            b = -0.01

        init_params = np.array([a, b, c])
        # tmp_params=None
        try:
            tmp_params = optimize.least_squares(residual_exp_curve,
                                                init_params, loss=loss,
                                                method=method, args=(
                    altitudes, intensities),
                                                bounds=(
                                                    bound_lower, bound_upper))
            if tmp_params.cost < min_cost:
                min_cost = tmp_params.cost
                ret_params = tmp_params.x

        except (ValueError, UnboundLocalError) as e:
            Console.error('Value Error', a, b, c)

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


def correct_attenuation(src_img_file_path, dst_img_file_path, src_altitude,
                        gain, table_correct_params):
    src_img = np.array(imageio.imread(str(src_img_file_path)))

    dst_img = gain / (
            table_correct_params[:, :, :, 0] *
            np.exp(table_correct_params[:, :, :, 1] * src_altitude)
            + table_correct_params[:, :, :, 2]) * src_img
    dst_img = dst_img.astype(np.uint8)
    imageio.imwrite(dst_img_file_path, dst_img)


def correct_attenuation_from_raw(mosaic_mat, src_altitude, gain,
                                 table_correct_params):
    return ((gain / (
            table_correct_params[:, :, :, 0] *
            np.exp(table_correct_params[:, :, :, 1] * src_altitude)
            + table_correct_params[:, :, :, 2])) * mosaic_mat).astype(
        np.float32)


def correct_distortion(src_img, map_x, map_y):
    src_img = np.clip(src_img, 0, 255).astype(np.uint8)
    dst_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR)
    return dst_img


def apply_atn_crr_2_img(img, altitude, atn_crr_params, gain):
    atn_crr_params = atn_crr_params.squeeze()
    img = ((gain / (
            atn_crr_params[:, :, 0] *
            np.exp(atn_crr_params[:, :, 1] * altitude)
            + atn_crr_params[:, :, 2])) * img).astype(np.float32)
    return img


def optim_exp_curve_param_log_transform(altitudes, intensities):
    altitudes = np.array(altitudes)
    intensities = np.clip(np.array(intensities), 1,
                          np.inf)  # avoid np.log(0) warning

    try:
        intensities_log = np.log(intensities)
    except Warning:
        print()

    c = 1

    altitudes_with1 = np.ones((altitudes.shape[0], 2), dtype=np.float)
    altitudes_with1[:, 1] = altitudes

    # singularity check
    #assert np.linalg.det(altitudes_with1.transpose().dot(altitudes_with1)) != 0, altitudes_with1

    
    #theta = np.inv(altitudes_with1.transpose().dot(altitudes_with1)).dot(
        #altitudes_with1.transpose()).dot(intensities_log)
    
    # pinv is used to remove singularity when working with a small set of images. when
    # working with a large dataset uncomment code in lines 1827 and 1828
    theta = np.linalg.pinv(altitudes_with1.transpose().dot(altitudes_with1)).dot(
        altitudes_with1.transpose()).dot(intensities_log)


    a = np.exp(theta[0])
    b = theta[1]

    return np.array([a, b, c])
