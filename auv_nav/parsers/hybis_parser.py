import os
import sys
import math
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import calendar


def date_time_to_epoch(yyyy, mm, dd, hh, mm1, ss, timezone_offset_to_utc=0):
    utc_date_time = (datetime(yyyy, mm, dd, hh, mm1, ss)
                     - timedelta(hours=timezone_offset_to_utc))
    epochtime = calendar.timegm(utc_date_time.timetuple())
    return epochtime


def interpolate(x_query, x_lower, x_upper, y_lower, y_upper):
    if x_upper == x_lower:
        y_query = y_lower
    else:
        y_query = (y_upper-y_lower)/(x_upper-x_lower)*(x_query-x_lower)+y_lower
    return y_query


def latlon_to_metres(latitude, longitude,
                     latitude_reference, longitude_reference):

    # average radius of earth, m
    R = 6378137.0

    latitude = latitude*math.pi/180.0
    latitude_reference = latitude_reference*math.pi/180.0

    longitude = longitude*math.pi/180.0
    longitude_reference = longitude_reference*math.pi/180.0

    a = (
        math.sin((latitude-latitude_reference)/2)
        * math.sin((latitude-latitude_reference)/2)
        + math.cos(latitude_reference)
        * math.cos(latitude)
        * math.sin((longitude-longitude_reference)/2)
        * math.sin((longitude-longitude_reference)/2))
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R*c

    y = math.sin(longitude-longitude_reference) * math.cos(latitude)
    x = (math.cos(latitude_reference)
         * math.sin(latitude)
         - math.sin(latitude_reference)
         * math.cos(latitude)
         * math.cos(longitude-longitude_reference))

    bearing = math.atan2(y, x)*180/math.pi

    while bearing > 360:
        bearing = bearing - 360
    while bearing < 0:
        bearing = bearing + 360

    return (distance, bearing)


class HyBisPos:
    def __init__(self, roll, pitch, heading,
                 depth, altitude, lon, lat,
                 date=0, timestr=0, stamp=0):
        if date is not 0:
            self.epoch_timestamp = self.convert(date, timestr)
        else:
            self.epoch_timestamp = stamp
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.heading = float(heading)
        self.depth = float(depth)
        self.altitude = float(altitude)
        self.longitude = self.convert_latlon(lon)
        self.latitude = self.convert_latlon(lat)

    def convert_latlon(self, latlon):
        # 5950.89694N, 00703.39736W
        final = 0
        if len(latlon) == 11:
            final = 1
        deg = float(latlon[0:3-final])
        minutes = float(latlon[3-final:11-final])
        zone = latlon[11-final]
        decdeg = deg + minutes/60.0
        if zone == 'W' or zone == 'S':
            return -decdeg
        else:
            return decdeg

    def convert(self, date, timestr):
        yyyy = int(date[6:10])
        mm = int(date[3:5])
        dd = int(date[0:2])
        hour = int(timestr[0:2])
        mins = int(timestr[3:5])
        secs = int(timestr[6:8])
        if hour < 0:
            hour = 0
            mins = 0
            secs = 0
        epoch_time = date_time_to_epoch(
            yyyy, mm, dd, hour, mins, secs, 0)
        return epoch_time


def parse_hybis(navigation_file,
                image_path,
                output_file,
                reference_lat=0,
                reference_lon=0):
    # extract data from files
    df = pd.read_csv(navigation_file, skipinitialspace=True)
    date = list(df['Date '])
    timestr = list(df['Time'])
    roll = list(df['Roll'])
    pitch = list(df['Pitch'])
    heading = list(df['Heading'])
    depth = list(df['Pressure'])
    altitude = list(df['Altitude'])
    lon = list(df['Hybis Long'])
    lat = list(df['Hybis Lat'])

    print('Found ' + str(len(df)) + ' navigation records!')

    hybis_vec = []
    for i in range(len(df)):
        if len(lon[i]) < 11:
            continue
        p = HyBisPos(roll[i], pitch[i], heading[i],
                     depth[i], altitude[i], lon[i], lat[i],
                     date[i], timestr[i])
        hybis_vec.append(p)

    for i in range(len(hybis_vec) - 1):
        if hybis_vec[i].altitude == 0:
            hybis_vec[i].altitude = interpolate(
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i-1].epoch_timestamp,
                hybis_vec[i+1].epoch_timestamp,
                hybis_vec[i-1].altitude,
                hybis_vec[i+1].altitude)
        if hybis_vec[i].depth == 0:
            hybis_vec[i].depth = interpolate(
                hybis_vec[i].epoch_timestamp,
                hybis_vec[i-1].epoch_timestamp,
                hybis_vec[i+1].epoch_timestamp,
                hybis_vec[i-1].depth,
                hybis_vec[i+1].depth)

    if reference_lon == 0 or reference_lat == 0:
        i = 0
        while hybis_vec[i].latitude == 0:
            i += 1
        latitude_ref = hybis_vec[i].latitude
        longitude_ref = hybis_vec[i].longitude
    else:
        latitude_ref = reference_lat
        longitude_ref = reference_lon

    data = ['Imagenumber, ', 'Northing [m], ', 'Easting [m], ', 'Depth [m], ',
            'Roll [deg], ', 'Pitch [deg], ', 'Heading [deg], ', 'Altitude [m], ',
            'Timestamp, ', 'Latitude [deg], ', 'Longitude [deg],', 'Filename\n']

    image_list = sorted(os.listdir(str(image_path)))
    print('Found ' + str(len(image_list)) + ' images!')
    print('Interpolating...')
    for k, filename in enumerate(image_list):
        modification_time = os.stat(image_path + filename).st_mtime

        i = 0
        while i < len(hybis_vec) - 2 and hybis_vec[i].epoch_timestamp < modification_time:
            i += 1

        latitude = interpolate(
            modification_time,
            hybis_vec[i].epoch_timestamp,
            hybis_vec[i+1].epoch_timestamp,
            hybis_vec[i].latitude,
            hybis_vec[i+1].latitude)
        longitude = interpolate(
            modification_time,
            hybis_vec[i].epoch_timestamp,
            hybis_vec[i+1].epoch_timestamp,
            hybis_vec[i].longitude,
            hybis_vec[i+1].longitude)
        depth = interpolate(
            modification_time,
            hybis_vec[i].epoch_timestamp,
            hybis_vec[i+1].epoch_timestamp,
            hybis_vec[i].depth,
            hybis_vec[i+1].depth)
        roll = interpolate(
            modification_time,
            hybis_vec[i].epoch_timestamp,
            hybis_vec[i+1].epoch_timestamp,
            hybis_vec[i].roll,
            hybis_vec[i+1].roll)
        pitch = interpolate(
            modification_time,
            hybis_vec[i].epoch_timestamp,
            hybis_vec[i+1].epoch_timestamp,
            hybis_vec[i].pitch,
            hybis_vec[i+1].pitch)
        heading = interpolate(
            modification_time,
            hybis_vec[i].epoch_timestamp,
            hybis_vec[i+1].epoch_timestamp,
            hybis_vec[i].heading,
            hybis_vec[i+1].heading)
        altitude = interpolate(
            modification_time,
            hybis_vec[i].epoch_timestamp,
            hybis_vec[i+1].epoch_timestamp,
            hybis_vec[i].altitude,
            hybis_vec[i+1].altitude)

        lateral_distance, bearing = latlon_to_metres(
                            latitude, longitude, latitude_ref, longitude_ref)
        eastings = math.sin(bearing*math.pi/180.0)*lateral_distance
        northings = math.cos(bearing*math.pi/180.0)*lateral_distance

        msg = (str(k) + ', ' + str(northings) + ', ' + str(eastings)
               + ', ' + str(depth) + ', ' + str(roll) + ', ' + str(pitch)
               + ', ' + str(heading) + ', ' + str(altitude) + ', '
               + str(modification_time) + ', ' + str(latitude) + ', ' + str(longitude)
               + ', ' + str(filename) + '\n')
        data.append(msg)
    print('Writing output to ' + output_file)
    output_file = Path(output_file)
    with output_file.open('w', encoding="utf-8") as fileout:
        for line in data:
            fileout.write(str(line).decode('utf-8'))
    print('DONE!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'navigation_file', help="Input navigation file (Hybis Dive_51.txt)")
    parser.add_argument(
        'image_path', help="Path to the folder containing Scorpio images")
    parser.add_argument(
        '-o', '--output', dest='output_file', default='hybis_dr_scorpion.csv', help="Name of the output CSV file")
    parser.add_argument(
        '--ref-lat', dest='reference_lat', default=0, help="Reference latitude. If none is provided, the first USBL fix will be used.")
    parser.add_argument(
        '--ref-lon', dest='reference_lon', default=0, help="Reference longitude. If none is provided, the first USBL fix will be used.")

    if len(sys.argv) == 1:
        print('Parser for HyBis ROV dives with Scorpio camera')
        print('Pass the location of the navigation file. Example: "/media/drive/Dive\ 51/Hybis\ Dive_51.txt.txt"')
        print('and the path to the scorpio images. Example: "/media/drive/Dive\ 51/scorpio/"')
        print('Example call:')
        print('')
        print('python3 hybis_parser.py /media/drive/Dive\ 51/Hybis\ Dive_51.txt.txt /media/drive/Dive\ 51/scorpio/')
        print('')
        print('By default, the first USBL fix will be used to convert navigation to meters. You can speficy one if you wish.')
        print('write python3 hybis_parser.py --help for more information.')
        print('')
        # Show help if no args provided
        #   parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    parse_hybis(args.navigation_file,
                args.image_path,
                args.output_file,
                args.reference_lat,
                args.reference_lon)
