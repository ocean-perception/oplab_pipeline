from pathlib import Path
import pandas as pd


def write_csv(csv_filepath, data_list, csv_filename, csv_flag):
    if csv_flag is True:
        # check the relvant folders exist and if note create them
        csv_file = Path(csv_filepath)
        if not csv_file.exists():
            csv_file.mkdir(parents=True, exist_ok=True)

        print("Writing outputs to {}.csv ...".format(csv_filename))
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
            print('WARNING: empty data list ', str(csv_filename))


# First column of csv file - image file naming step probably not very robust
# needs improvement
def camera_csv(camera_list, camera_name, csv_filepath, csv_flag):
    if csv_flag is True and len(camera_list) > 1:
        csv_file = Path(csv_filepath)
        if csv_file.exists() is False:
            csv_file.mkdir(parents=True, exist_ok=True)

        print("Writing outputs to {}.csv ...".format(camera_name))
        file = csv_file / '{}.csv'.format(camera_name)
        str_to_write = ''
        str_to_write += 'Imagenumber, Northing [m], Easting [m], Depth [m], ' \
                        'Roll [deg], Pitch [deg], Heading [deg], Altitude '\
                        '[m], Timestamp, Latitude [deg], Longitude [deg]\n'
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
                    for c in camera_list[i].covariance:
                        str_to_write += ',' + str(c)
                str_to_write += '\n'
            except IndexError:
                break
        with file.open('w') as fileout:
            fileout.write(str_to_write)


# if this works make all follow this format!
def other_data_csv(data_list, data_name, csv_filepath, csv_flag):
    csv_file = Path(csv_filepath)
    if csv_file.exists() is False:
        csv_file.mkdir(parents=True, exist_ok=True)

    if csv_flag is True:
        print("Writing outputs to {}.csv ...".format(data_name))
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
