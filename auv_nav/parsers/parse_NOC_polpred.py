# parse_NOC_polpred
# Script to parse NOC tide data generated with POLPRED Continental Shelf Extended Model (CSX)
# Author: Jose Cappelletto
# Date: 11/09/2019
from auv_nav.sensors import Category
from auv_nav.sensors import Tide
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_file_list
from auv_nav.tools.time_conversions import date_time_to_epoch
from auv_nav.tools.time_conversions import read_timezone

def parse_NOC_polpred(mission,
                   vehicle,
                   category,
                   ftype,
                   outpath,
                   fileoutname):
    # parser meta data
    class_string = 'measurement'
    sensor_string = 'autosub'
    category = category
    output_format = ftype

    if category == Category.TIDE:
        filename = mission.tide.filename
        filepath = mission.tide.filepath
        timezone = mission.tide.timezone
        timeoffset = mission.tide.timeoffset
        timezone_offset = read_timezone(timezone)
        latitude_reference = mission.origin.latitude
        longitude_reference = mission.origin.longitude

        tide = Tide(mission.tide.std_offset,
                    latitude_reference,
                    longitude_reference)
        tide.sensor_string = sensor_string

        path = get_raw_folder(outpath / '..' / filepath)

        file_list = get_file_list(path)

        data_list = []

        # Data format sample
        #  Date      Time      Level     Speed    Direc'n
        #                         m        m/s       deg
        # 6/ 9/2019  00:00       0.74      0.14       51
        # 6/ 9/2019  01:00       0.58      0.15       56

        for file in file_list:
            with file.open('r', errors='ignore') as tide_file:
                for line in tide_file.readlines():
                    # we have to skip the first 5 rows of the file
                    for skip in range(5):
                        next(tide_file)

                    dd = int(line[0:2])
                    mm = int(line[3:5])
                    yyyy = int(line[6:10])

                    hour = int(line[12:14])
                    mins = int(line[15:17])
                    secs = 0        # current models only provide resolution in minutes
                    msec = 0
                    epoch_time = date_time_to_epoch(
                                    yyyy, mm, dd, hour, mins, secs, timezone_offset)
                    epoch_timestamp = epoch_time+msec/1000+timeoffset

                    tide.timestamp = epoch_timestamp
                    tide.height = float(line(22:28))

                    data_list.append(tide)
        return data_list
