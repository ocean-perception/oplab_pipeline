from auv_nav.tools.console import Console
from pathlib import Path
import pandas as pd

import numpy as np

def write_csv(csv_filepath, data_list, csv_filename, csv_flag):
    if csv_flag is True:
        # check the relvant folders exist and if note create them
        csv_file = Path(csv_filepath)
        if not csv_file.exists():
            csv_file.mkdir(parents=True, exist_ok=True)

        Console.info("Writing outputs to {}.csv ...".format(csv_filename))
        file = csv_file / '{}.csv'.format(csv_filename)
        str_to_write = ''
        str_to_write += 'Timestamp, Northing [m], Easting [m], Depth [m], ' \
                        'Roll [deg], Pitch [deg], Heading [deg], Altitude ' \
                        '[m], Latitude [deg], Longitude [deg]'
        if len(data_list) > 0:
            if data_list[0].covariance is not None:
                cov = ['x', 'y', 'z',
                       'roll', 'pitch', 'yaw',
                       'vx', 'vy', 'vz',
                       'vroll', 'vpitch', 'vyaw',
                       'ax', 'ay', 'az']
                for a in cov:
                    for b in cov:
                        str_to_write += ', cov_'+a+'_'+b
            str_to_write += '\n'
            for i in range(len(data_list)):
                try:
                    str_to_write += (
                        str(data_list[i].epoch_timestamp)+','
                        + str(data_list[i].northings)+','
                        + str(data_list[i].eastings)+','
                        + str(data_list[i].depth)+','
                        + str(data_list[i].roll)+','
                        + str(data_list[i].pitch)+','
                        + str(data_list[i].yaw)+','
                        + str(data_list[i].altitude)+','
                        + str(data_list[i].latitude)+','
                        + str(data_list[i].longitude))

                    if data_list[i].covariance is not None:
                        cov = data_list[i].covariance.flatten().tolist()
                        cov = [item for sublist in cov for item in sublist]
                        for c in cov:
                            str_to_write += ',' + str(c)
                    str_to_write += '\n'
                except IndexError:
                    break
            with file.open('w') as fileout:
                fileout.write(str_to_write)
        else:
            Console.warn('Empty data list {}'.format(str(csv_filename)))


def spp_csv(camera_list, camera_name, csv_filepath, csv_flag):
    if csv_flag is True and len(camera_list) > 1:
        csv_file = Path(csv_filepath)
        if csv_file.exists() is False:
            csv_file.mkdir(parents=True, exist_ok=True)

        Console.info("Writing outputs to {}.txt ...".format(camera_name))
        file = csv_file / '{}.txt'.format(camera_name)
        str_to_write = ''
        if len(camera_list) > 0:
            # With unwritted header: ['image_num_from', 'image_num_to',
            #                         'x', 'y', 'z', 'yaw', 'pitch', 'roll',
            #                         'inf_x_x', 'inf_x_y', 'inf_x_z',
            #                         'inf_x_yaw', 'inf_x_pitch',
            #                         'inf_x_roll', 'inf_y_y', 'inf_y_z',
            #                         'inf_y_yaw', 'inf_y_pitch',
            #                         'inf_y_roll', 'inf_z_z', 'inf_z_yaw',
            #                         ... ]
            # NOTE: matrix listed is just the upper corner of the diagonal
            # symetric information matrix, and order for SLAM++ input of
            # rotational variables is yaw, pitch, roll (reverse order).
            offset = 0
            for i in range(len(camera_list)):
                try:
                    imagenumber = camera_list[i].filename[-11:-4]
                    if imagenumber.isdigit():
                        # Ensure pose/node IDs start at zero.
                        if i == 0:
                            offset = int(imagenumber)
                        image_filename = int(imagenumber) - offset
                    else:
                        image_filename = camera_list[i].filename
                        Console.warn('image_filename for csv output has been'
                                     + ' set = camera_list[i].filename. If'
                                     + ' a failure has occurred, may be'
                                     + ' because this is not a number and'
                                     + ' cannot be turned into an "int", as'
                                     + ' needed for SLAM++ txt file output.')
                    str_to_write += (
                        'EDGE3' + ' '
                        + str(int(image_filename)) + ' '
                        + str(int(image_filename) + 1) + ' '
                        + str(np.sum(camera_list[i].northings)) + ' '
                        + str(np.sum(camera_list[i].eastings)) + ' '
                        + str(np.sum(camera_list[i].depth)) + ' '
                        + str(np.sum(camera_list[i].yaw)) + ' '
                        + str(np.sum(camera_list[i].pitch)) + ' '
                        + str(np.sum(camera_list[i].roll))
                        )
                    if camera_list[i].information is not None:
                        inf = camera_list[i].information.flatten().tolist()
                        inf = [item for sublist in inf for item in sublist]
                        # There are 12 state variables, we are only
                        # interested in the first 6. Hence the final 6x12
                        # elements in the information matrix can be
                        # deleted, as these have an unwanted primary variable.
                        inf = inf[:-72]
                        
                        for i in range(6):
                            # The rotationnal elements need to be switched
                            # around to be in SLAM++ (reverse) order.
                            j = inf[12*i + 3]
                            inf[12*i + 3] = inf[12*i + 5]
                            inf[12*i + 5] = j
                        j = inf[36:48]
                        inf[36:48] = inf[60:72]
                        inf[60:72] = j
                        
                        for i in range(6):
                            # Of the remaining 6x12 elements, half have unwanted
                            # secondary variables (the latter half of each
                            # primary variables chain of elements) and can be
                            # deleted. Duplicated elements (due to symmetry)
                            # can also be deleted.
                            inf += inf[i:6]
                            inf = inf[12:]
                        for c in inf:
                            str_to_write += ' ' + str(c)
                    str_to_write += '\n'
                except IndexError:
                    break
            with file.open('w') as fileout:
                fileout.write(str_to_write)
        else:
            Console.warn('Empty data list {}'.format(str(camera_name)))


# First column of csv file - image file naming step probably not very robust
# needs improvement
def camera_csv(camera_list, camera_name, csv_filepath, csv_flag):
    if csv_flag is True and len(camera_list) > 1:
        csv_file = Path(csv_filepath)
        if csv_file.exists() is False:
            csv_file.mkdir(parents=True, exist_ok=True)

        Console.info("Writing outputs to {}.csv ...".format(camera_name))
        file = csv_file / '{}.csv'.format(camera_name)
        str_to_write = ''
        str_to_write += 'Imagenumber, Northing [m], Easting [m], Depth [m], ' \
                        'Roll [deg], Pitch [deg], Heading [deg], Altitude '\
                        '[m], Timestamp, Latitude [deg], Longitude [deg]'
        if len(camera_list) > 0:
            if camera_list[0].covariance is not None:
                cov = ['x', 'y', 'z',
                       'roll', 'pitch', 'yaw',
                       'vx', 'vy', 'vz',
                       'vroll', 'vpitch', 'vyaw']
                for a in cov:
                    for b in cov:
                        str_to_write += ', cov_'+a+'_'+b
            str_to_write += '\n'
            for i in range(len(camera_list)):
                try:
                    imagenumber = camera_list[i].filename[-11:-4]
                    if imagenumber.isdigit():
                        image_filename = imagenumber
                    else:
                        image_filename = camera_list[i].filename
                    str_to_write += (
                        str(image_filename) + ','
                        + str(camera_list[i].northings) + ','
                        + str(camera_list[i].eastings) + ','
                        + str(camera_list[i].depth) + ','
                        + str(camera_list[i].roll) + ','
                        + str(camera_list[i].pitch) + ','
                        + str(camera_list[i].yaw) + ','
                        + str(camera_list[i].altitude) + ','
                        + str(camera_list[i].epoch_timestamp) + ','
                        + str(camera_list[i].latitude) + ','
                        + str(camera_list[i].longitude))
                    if camera_list[i].covariance is not None:
                        cov = camera_list[i].covariance.flatten().tolist()
                        cov = [item for sublist in cov for item in sublist]
                        for c in cov:
                            str_to_write += ',' + str(c)
                    str_to_write += '\n'
                except IndexError:
                    break
            with file.open('w') as fileout:
                fileout.write(str_to_write)
        else:
            Console.warn('Empty data list {}'.format(str(camera_name)))


# if this works make all follow this format!
def other_data_csv(data_list, data_name, csv_filepath, csv_flag):
    csv_file = Path(csv_filepath)
    if csv_file.exists() is False:
        csv_file.mkdir(parents=True, exist_ok=True)

    if csv_flag is True:
        Console.info("Writing outputs to {}.csv ...".format(data_name))
        # csv_header =
        csv_row_data_list = []
        for i in data_list:
            csv_row_data = {'epochtimestamp': i.epoch_timestamp, 'Northing [m]': i.northings, 'Easting [m]': i.eastings, 'Depth [m]': i.depth, 'Roll [deg]': i.roll,
                            'Pitch [deg]': i.pitch, 'Heading [deg]': i.yaw, 'Altitude [m]': i.altitude, 'Latitude [deg]': i.latitude, 'Longitude [deg]': i.longitude}
            for j in i.data:
                csv_row_data.update(
                    {'{} [{}]'.format(j['label'], j['units']): j['value']})
            csv_row_data_list.append(csv_row_data)
        df = pd.DataFrame(csv_row_data_list)
        # , na_rep='-') https://www.youtube.com/watch?v=hmYdzvmcTD8
        df.to_csv(csv_file / '{}.csv'.format(data_name),
                  header=True, index=False)


def write_raw_sensor_csv(csv_filepath, data_list, csv_filename, mutex=None):
    if len(data_list) > 0:
        # check the relvant folders exist and if note create them
        csv_file = Path(csv_filepath)
        if not csv_file.exists():
            if mutex is not None:
                mutex.acquire()
            csv_file.mkdir(parents=True, exist_ok=True)
            if mutex is not None:
                mutex.release()

        Console.info("Writing raw sensor to {}.csv ...".format(csv_filename))
        file = csv_file / '{}.csv'.format(csv_filename)
        str_to_write = ''
        str_to_write += data_list[0].write_csv_header()

        for d in data_list:
            str_to_write += d.to_csv()
        with file.open('w') as fileout:
            fileout.write(str_to_write)
    else:
        Console.warn('Empty data list {}'.format(str(csv_filename)))
