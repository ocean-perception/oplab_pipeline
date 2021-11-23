# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path

import numpy as np

from oplab import Console


def write_csv(csv_filepath, data_list, csv_filename, csv_flag=True, mutex=None):
    if csv_flag is True and len(data_list) > 1:
        csv_file = Path(csv_filepath)

        if not csv_file.exists():
            if mutex is not None:
                mutex.acquire()
            csv_file.mkdir(parents=True, exist_ok=True)
            if mutex is not None:
                mutex.release()

        Console.info("Writing outputs to {}.csv ...".format(csv_filename))
        file = csv_file / "{}.csv".format(csv_filename)
        covariance_file = csv_file / "{}_cov.csv".format(csv_filename)

        fileout = None
        fileout_cov = None
        if len(data_list) > 0:
            fileout = file.open("w")
            # write headers
            str_to_write = data_list[0].get_csv_header()
            fileout.write(str_to_write)
            if hasattr(data_list[0], "covariance") and hasattr(
                data_list[0], "get_csv_header_cov"
            ):
                if data_list[0].covariance is not None:
                    fileout_cov = covariance_file.open("w")
                    str_to_write_cov = data_list[0].get_csv_header_cov()
                    fileout_cov.write(str_to_write_cov)

            # Loop for each line in csv
            for i in range(len(data_list)):
                try:
                    str_to_write = data_list[i].to_csv_row()
                    if fileout_cov is not None:
                        str_to_write_cov = data_list[i].to_csv_cov_row()
                        fileout_cov.write(str_to_write_cov)
                    fileout.write(str_to_write)
                except IndexError:
                    Console.error(
                        "There is something wrong with camera filenames and \
                        indexing for the file",
                        csv_filename,
                    )
                    Console.quit("Check write_csv function.")
            fileout.close()
            if fileout_cov is not None:
                fileout_cov.close()
        else:
            Console.warn("Empty data list {}".format(str(csv_filename)))


def write_sidescan_csv(csv_filepath, data_list, csv_filename, csv_flag):
    if csv_flag:
        csv_file = Path(csv_filepath)
        if not csv_file.exists():
            csv_file.mkdir(parents=True, exist_ok=True)
        Console.info("Writing SSS outputs to {}.txt ...".format(csv_filename))
        file = csv_file / "{}.txt".format(csv_filename)
        if len(data_list) > 0:
            str_to_write = data_list[0].get_sidescan_header()
            with file.open("w") as fileout:
                fileout.write(str_to_write)
                for i in range(len(data_list)):
                    try:
                        str_to_write = data_list[i].to_sidescan_row()
                        fileout.write(str_to_write)
                    except IndexError:
                        break
        else:
            Console.warn("Empty data list {}".format(str(csv_filename)))


def spp_csv(camera_list, camera_name, csv_filepath, csv_flag):
    if csv_flag is True and len(camera_list) > 1:
        csv_file = Path(csv_filepath)
        if csv_file.exists() is False:
            csv_file.mkdir(parents=True, exist_ok=True)

        Console.info("Writing outputs to {}.txt ...".format(camera_name))
        file = csv_file / "{}.txt".format(camera_name)
        str_to_write = ""
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
                        Console.warn(
                            "image_filename for csv output has been"
                            + " set = camera_list[i].filename. If"
                            + " a failure has occurred, may be"
                            + " because this is not a number and"
                            + ' cannot be turned into an "int", as'
                            + " needed for SLAM++ txt file output."
                        )
                    str_to_write += (
                        "EDGE3"
                        + " "
                        + str(int(image_filename))
                        + " "
                        + str(int(image_filename) + 1)
                        + " "
                        + str(np.sum(camera_list[i].northings))
                        + " "
                        + str(np.sum(camera_list[i].eastings))
                        + " "
                        + str(np.sum(camera_list[i].depth))
                        + " "
                        + str(np.sum(camera_list[i].yaw))
                        + " "
                        + str(np.sum(camera_list[i].pitch))
                        + " "
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
                            j = inf[12 * i + 3]
                            inf[12 * i + 3] = inf[12 * i + 5]
                            inf[12 * i + 5] = j
                        j = inf[36:48]
                        inf[36:48] = inf[60:72]
                        inf[60:72] = j

                        for i in range(6):
                            # Of the remaining 6x12 elements, half have
                            # unwanted secondary variables (the latter half of
                            # each primary variables chain of elements) and can
                            # be deleted. Duplicated elements (due to symmetry)
                            # can also be deleted.
                            inf += inf[i:6]
                            inf = inf[12:]
                        for c in inf:
                            str_to_write += " " + str(c)
                    str_to_write += "\n"
                except IndexError:
                    break
            with file.open("w") as fileout:
                fileout.write(str_to_write)
        else:
            Console.warn("Empty data list {}".format(str(camera_name)))
