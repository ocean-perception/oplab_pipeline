from pathlib import Path
import os
import pandas as pd


def write_csv(csv_filepath, data_list, csv_filename, csv_flag):
    if csv_flag is True:
        # check the relvant folders exist and if note create them
        csv_file = Path(csv_filepath)
        if csv_file.exists() is False:
            os.makedirs(csv_filepath, exist_ok=True)

        print("Writing outputs to {}.csv ...".format(csv_filename))
        with open(os.path.join(csv_filepath,
                               '{}.csv'.format(csv_filename)),
                  'w') as fileout:
            fileout.write(
                'Timestamp, Northing [m], Easting [m], Depth [m], Roll [deg], \
                Pitch [deg], Heading [deg], Altitude [m], Latitude [deg], \
                Longitude [deg]\n')
        for i in range(len(data_list)):
            with open(os.path.join(csv_filepath,
                                   '{}.csv'.format(csv_filename)),
                      'a')as fileout:
                try:
                    fileout.write(
                        str(data_list[i].epoch_timestamp)+','
                        + str(data_list[i].northings)+','
                        + str(data_list[i].eastings)+','
                        + str(data_list[i].depth)+','
                        + str(data_list[i].roll)+','
                        + str(data_list[i].pitch)+','
                        + str(data_list[i].yaw)+','
                        + str(data_list[i].altitude)+','
                        + str(data_list[i].latitude)+','
                        + str(data_list[i].longitude)+'\n')
                    fileout.close()
                except IndexError:
                    break


# First column of csv file - image file naming step probably not very robust
# needs improvement
def camera_csv(camera_list, camera_name, csv_filepath, csv_flag):
    if csv_flag is True:
        csv_file = Path(csv_filepath)
        if csv_file.exists() is False:
            os.makedirs(csv_filepath, exist_ok=True)
        if len(camera_list) > 1:
            print("Writing outputs to {}.csv ...".format(camera_name))
            with open(os.path.join(csv_filepath, '{}.csv'.format(camera_name)), 'w') as fileout:
                fileout.write(
                    'Imagenumber, Northing [m], Easting [m], Depth [m], Roll [deg], Pitch [deg], Heading [deg], Altitude [m], Timestamp, Latitude [deg], Longitude [deg]\n')
            for i in range(len(camera_list)):
                with open(os.path.join(csv_filepath, '{}.csv'.format(camera_name)), 'a') as fileout:
                    try:
                        imagenumber = camera_list[i].filename[-11:-4]
                        if imagenumber.isdigit():
                            image_filename = imagenumber
                        else:
                            image_filename = camera_list[i].filename
                        fileout.write(str(image_filename)+','+str(camera_list[i].northings)+','+str(camera_list[i].eastings)+','+str(camera_list[i].depth)+','+str(camera_list[i].roll)+','+str(
                            camera_list[i].pitch)+','+str(camera_list[i].yaw)+','+str(camera_list[i].altitude)+','+str(camera_list[i].epoch_timestamp)+','+str(camera_list[i].latitude)+','+str(camera_list[i].longitude)+'\n')
                        fileout.close()
                    except IndexError:
                        break


# if this works make all follow this format!
def other_data_csv(data_list, data_name, csv_filepath, csv_flag):
    csv_file = Path(csv_filepath)
    if csv_file.exists() is False:
        os.makedirs(csv_filepath, exist_ok=True)

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
        df.to_csv(os.path.join(csv_filepath, '{}.csv'.format(
            data_name)), header=True, index=False)
