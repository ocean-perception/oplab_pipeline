# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

import codecs
from auv_nav.auv_parsers.sensors import BodyVelocity, InertialVelocity
from auv_nav.auv_parsers.sensors import Orientation, Depth, Altitude
from auv_nav.auv_parsers.sensors import Category, Timestamp, PhinsHeaders


class PhinsParser():
    def __init__(self, filepath, filename, category, timezone, timeoffset,
                 heading_offset, ftype):
        # parser meta data
        self.class_string = 'measurement'
        self.sensor_string = 'phins'
        self.category = category

        self.filename = filename
        self.filepath = filepath
        self.output_format = ftype

        # phins std models
        depth_std_factor = 0.01/100  # from catalogue paroscientific
        velocity_std_factor = 0.001  # from catalogue rdi whn1200/600
        velocity_std_offset = 0.2    # from catalogue rdi whn1200/600
        altitude_std_factor = 1/100  # acoustic vel assumed 1% accurate

        # read in date from filename
        yyyy = int(filename[0:4])
        mm = int(filename[4:6])
        dd = int(filename[6:8])
        date = yyyy, mm, dd

        self.timestamp = Timestamp(date, timezone, timeoffset)

        self.body_velocity = BodyVelocity(velocity_std_factor,
                                          velocity_std_offset,
                                          heading_offset,
                                          self.timestamp)
        self.inertial_velocity = InertialVelocity()
        self.orientation = Orientation(heading_offset)
        self.depth = Depth(depth_std_factor, self.timestamp)
        self.altitude = Altitude(altitude_std_factor)

    def set_timestamp(self, epoch_timestamp):
        self.body_velocity.epoch_timestamp = epoch_timestamp
        self.inertial_velocity.epoch_timestamp = epoch_timestamp
        self.orientation.epoch_timestamp = epoch_timestamp
        self.depth.epoch_timestamp = epoch_timestamp
        self.altitude.epoch_timestamp = epoch_timestamp

    def line_is_valid(self, line, line_split):
        start_or_heading = (line[0] == PhinsHeaders.START
                            or line[0] == PhinsHeaders.HEADING)
        if (len(line_split) == 2
                and start_or_heading):
            # Get timestamp
            # Do a check sum as a lot of broken packets are found in phins data
            check_sum = str(line_split[1])

            # extract from $ to * as per phins manual
            string_to_check = ','.join(line)
            string_to_check = string_to_check[1:len(string_to_check)]
            string_sum = 0

            for i in range(len(string_to_check)):
                string_sum ^= ord(string_to_check[i])

            if str(hex(string_sum)[2:].zfill(2).upper()) == check_sum.upper():
                return True

            else:
                    print('Broken packet: ', line)
                    print('Check sum calculated ',
                          hex(string_sum).zfill(2).upper())
                    print('Does not match that provided', check_sum.upper())
                    print('Ignore and move on')
        return False

    def parse(self):
        # parse phins data
        print('...... parsing phins standard data')

        data_list = []

        with codecs.open(self.filepath + self.filename, 'r',
                         encoding='utf-8', errors='ignore') as filein:
            for complete_line in filein.readlines():
                line_and_md5 = complete_line.strip().split('*')
                line = line_and_md5[0].strip().split(',')
                if not self.line_is_valid(line, line_and_md5):
                    continue
                header = line[1]
                data = self.process_line(header, line)
                if data is not None:
                    data_list.append(data)
            return data_list

    def process_line(self, header, line):
        data = None
        if header == PhinsHeaders.TIME:
            epoch_timestamp = self.timestamp.get_time_from_phins(
                line)
            self.set_timestamp(epoch_timestamp)

        if self.category == Category.VELOCITY:

            if header == PhinsHeaders.DVL:
                self.body_velocity.from_phins(line)
                data = self.body_velocity.export(self.output_format)

            if (header == PhinsHeaders.VEL
                    or header == PhinsHeaders.VEL_STD):
                self.inertial_velocity.from_phins(line)
                data = self.inertial_velocity.export(self.output_format)

        if self.category == Category.ORIENTATION:
            self.orientation.from_phins(line)
            data = self.orientation.export(self.output_format)

        if (self.category == Category.DEPTH
                and header == PhinsHeaders.DEPTH):
            self.depth.from_phins(line)
            data = self.depth.export(self.output_format)

        if self.category == Category.ALTITUDE:
            if header == PhinsHeaders.ALTITUDE:
                self.altitude.from_phins(
                    line,
                    self.body_velocity.epoch_timestamp_dvl)
                data = self.altitude.export(self.output_format)
            if header == PhinsHeaders.DVL:
                self.body_velocity.from_phins(line)
                data = self.altitude.export(self.output_format)
        return data


def parse_phins(
        filepath, filename, category,
        timezone, timeoffset, heading_offset,
        output_format, outpath, out_filename):
    p = PhinsParser(filepath, filename, category, timezone,
                    timeoffset, heading_offset, output_format)
    return p.parse()
