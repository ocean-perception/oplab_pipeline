# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

from auv_nav.sensors import BodyVelocity, InertialVelocity
from auv_nav.sensors import Orientation, Depth, Altitude
from auv_nav.sensors import Category, Timestamp, PhinsHeaders
from auv_nav.tools.folder_structure import get_raw_folder
from auv_nav.tools.console import Console


class PhinsParser():
    def __init__(self, mission, vehicle, category, ftype, outpath, filename):
        # parser meta data
        self.class_string = 'measurement'
        self.sensor_string = 'phins'
        self.category = category

        self.outpath = outpath
        self.filename = mission.velocity.filename
        self.filepath = mission.velocity.filepath
        self.output_format = ftype
        self.headingoffset = vehicle.dvl.yaw

        # phins std models
        depth_std_factor = mission.depth.std_factor
        velocity_std_factor = mission.velocity.std_factor
        velocity_std_offset = mission.velocity.std_offset
        altitude_std_factor = mission.altitude.std_factor

        # read in date from filename
        yyyy = int(self.filename[0:4])
        mm = int(self.filename[4:6])
        dd = int(self.filename[6:8])
        date = yyyy, mm, dd

        self.timestamp = Timestamp(date,
                                   mission.velocity.timezone,
                                   mission.velocity.timeoffset)
        self.body_velocity = BodyVelocity(velocity_std_factor,
                                          velocity_std_offset,
                                          self.headingoffset,
                                          self.timestamp)
        self.inertial_velocity = InertialVelocity()
        self.orientation = Orientation(self.headingoffset)
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
                    Console.warn('Broken packet: ' + str(line))
                    Console.warn('Check sum calculated ' + str(hex(string_sum).zfill(2).upper()))
                    Console.warn('Does not match that provided ' + str(check_sum.upper()))
                    Console.warn('Ignore and move on')
        return False

    def parse(self):
        # parse phins data
        Console.info('... parsing phins standard data')

        data_list = []
        path = get_raw_folder(self.outpath / '..' / self.filepath / self.filename)
        with path.open('r', encoding='utf-8', errors='ignore') as filein:
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

    def build_rdi_acfr(self):
        data = ('RDI: ' + str(float(self.body_velocity.epoch_timestamp_dvl))
                + ' alt:' + str(float(self.altitude.altitude))
                + ' r1:0 r2:0 r3:0 r4:0'
                + ' h:' + str(float(self.orientation.yaw))
                + ' p:' + str(float(self.orientation.pitch))
                + ' r:' + str(float(self.orientation.roll))
                + ' vx:' + str(float(self.body_velocity.x_velocity))
                + ' vy:' + str(float(self.body_velocity.y_velocity))
                + ' vz:' + str(float(self.body_velocity.z_velocity))
                + ' nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:0 h_true:0 p_gimbal:0'
                + ' sv: ' + str(float(self.altitude.sound_velocity)) + '\n')
        self.body_velocity.clear()
        self.orientation.clear()
        self.altitude.clear()
        return data

    def process_line(self, header, line):
        data = None
        if header == PhinsHeaders.TIME:
            epoch_timestamp = self.timestamp.epoch_timestamp_from_phins(
                line)
            self.set_timestamp(epoch_timestamp)

        if self.category == Category.VELOCITY:
            if header == PhinsHeaders.DVL:
                self.body_velocity.from_phins(line)
                if self.output_format == 'oplab':
                    data = self.body_velocity.export(self.output_format)

            if (header == PhinsHeaders.VEL
                    or header == PhinsHeaders.VEL_STD):
                self.inertial_velocity.from_phins(line)
                if self.output_format == 'oplab':
                    data = self.inertial_velocity.export(self.output_format)

            if self.output_format == 'acfr':
                self.orientation.from_phins(line)
                if header == PhinsHeaders.ALTITUDE:
                    self.altitude.from_phins(
                        line,
                        self.body_velocity.epoch_timestamp_dvl)
                if (self.body_velocity.valid()
                   and self.altitude.valid()
                   and self.orientation.valid()):
                    data = self.build_rdi_acfr()

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
        mission,
        vehicle,
        category,
        ftype,
        outpath,
        fileoutname):
    p = PhinsParser(mission,
                    vehicle,
                    category,
                    ftype,
                    outpath,
                    fileoutname)
    return p.parse()
