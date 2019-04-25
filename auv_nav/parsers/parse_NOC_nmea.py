import pynmea2
from auv_nav.sensors import Category
from auv_nav.sensors import Usbl
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.folder_structure import get_file_list
from auv_nav.tools.time_conversions import date_time_to_epoch
from auv_nav.tools.time_conversions import read_timezone


def parse_NOC_nmea(mission,
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

    if category == Category.USBL:
        filename = mission.usbl.filename
        filepath = mission.usbl.filepath
        timezone = mission.usbl.timezone
        timeoffset = mission.usbl.timeoffset
        timezone_offset = read_timezone(timezone)
        latitude_reference = mission.origin.latitude
        longitude_reference = mission.origin.longitude

        usbl = Usbl(mission.usbl.std_offset,
                    latitude_reference,
                    longitude_reference)
        usbl.sensor_string = sensor_string

        path = get_raw_folder(outpath / '..' / filepath)

        file_list = get_file_list(path)

        data_list = []
        for file in file_list:
            with file.open('r', errors='ignore') as nmea_file:
                for line in nmea_file.readlines():
                    parts = line.split('\t')
                    msg = pynmea2.parse(parts[1])
                    date_str = line.split(' ')[0]
                    hour_str = str(parts[1]).split(',')[1]

                    yyyy = int(date_str[6:10])
                    mm = int(date_str[3:5])
                    dd = int(date_str[0:2])
                    hour = int(hour_str[0:2])
                    mins = int(hour_str[2:4])
                    secs = int(hour_str[4:6])
                    msec = int(hour_str[7:10])
                    epoch_time = date_time_to_epoch(
                                    yyyy, mm, dd, hour, mins, secs, timezone_offset)
                    epoch_timestamp = epoch_time+msec/1000+timeoffset
                    msg.timestamp = epoch_timestamp
                    usbl.from_nmea(msg)
                    data = usbl.export(output_format)
                    data_list.append(data)
        return data_list
