# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""


from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.latlon_wgs84 import latlon_to_metres
from auv_nav.tools.time_conversions import (
    date_time_to_epoch,
    epoch_to_utctime,
)
from oplab import (
    Console,
    Mission,
    Vehicle,
    get_raw_folder,
)
from oplab.folder_structure import get_processed_folder


class RovCam:
    """Class to parse and store camera FileTime data"""
    def __init__(self,
        date_Cam,
        time_Cam,
        ms_Cam,
        index,
        stamp=0,
    ):
        self.depth=np.nan
        self.altitude=np.nan
        self.heading=np.nan
        self.pitch=np.nan
        self.roll=np.nan
        self.lat=np.nan
        self.lon=np.nan
        self.index=int(index)
        
        if date_Cam != 0:
            self.epoch_timestamp = self.convert_times_to_epoch(
                date_Cam,
                time_Cam,
                ms_Cam,
            )
        else:
            self.epoch_timestamp = stamp
    
    def add_lever_arms(self,
        target: str,
        positions: dict,
    ):
        """Add the lever arms to depth, altitude, lat, and lon
        measurements.
        """
        self.depth = self.depth + body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            positions[target][0] - positions["depth"][0],
            positions[target][1] - positions["depth"][1],
            positions[target][2] - positions["depth"][2],
        )[2]
        self.altitude = self.altitude - body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            positions[target][0] - positions["dvl"][0],
            positions[target][1] - positions["dvl"][1],
            positions[target][2] - positions["dvl"][2],
        )[2]
        (delta_lat, delta_lon) = body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            positions[target][0] - positions["dvl"][0],
            positions[target][1] - positions["dvl"][1],
            positions[target][2] - positions["dvl"][2],
        )[:2]
        self.lat = self.lat + delta_lat
        self.lon = self.lon + delta_lon
        return

    def convert_times_to_epoch(self, date_Cam, time_Cam, ms_Cam):
        date_Cam = str(date_Cam)
        yyyy = int(date_Cam[0:4])
        mm = int(date_Cam[4:6])
        dd = int(date_Cam[6:8])
        time_Cam = str(time_Cam)
        hour = int(time_Cam[0:2])
        mins = int(time_Cam[2:4])
        secs = int(time_Cam[4:6])
        usecs = ms_Cam*1000
        if hour < 0:
            hour = 0
            mins = 0
            secs = 0
        epoch_time = date_time_to_epoch(
            yyyy,
            mm,
            dd,
            hour,
            mins,
            secs,
            0,
            usecs,
        )
        return epoch_time
    
    def write_out_vis_cam(self):
        """
        Write out data to a csv nav file in vis cam format.
        """
        
        return
    
    def write_out_laser_cam(self):
        """
        Write out data to a csv nav file in laser cam format.
        """
        
        return


class RovRot:
    """Class to parse and store koyo20rov TCM data"""
    def __init__(self,
        heading,
        pitch,
        roll,
        rot_date=0,
        rot_time=0,
        timems=0,
        stamp=0,
    ):
        if rot_date != 0:
            self.epoch_timestamp = self.convert_times_to_epoch(
                rot_date,
                rot_time,
                timems,
            )
        else:
            self.epoch_timestamp = stamp
        self.heading = float(heading)
        self.pitch = float(pitch)
        self.roll = float(roll)
    
    def convert_times_to_epoch(self, rot_date, rot_time, timems):
        rot_date = str(rot_date)
        yyyy = int(rot_date[0:4])
        mm = int(rot_date[4:6])
        dd = int(rot_date[6:8])
        rot_time = str(rot_time)
        hour = int(rot_time[0:2])
        mins = int(rot_time[2:4])
        secs = int(rot_time[4:6])
        usecs = timems*1000
        if hour < 0:
            hour = 0
            mins = 0
            secs = 0
        epoch_time = date_time_to_epoch(
            yyyy,
            mm,
            dd,
            hour,
            mins,
            secs,
            0,
            usecs,
        )
        return epoch_time


class RovPos:
    """Class to parse and store koyo20rov ship_log data."""
    def __init__(self,
        depth=np.nan,
        altitude=np.nan,
        lon=np.nan,
        lat=np.nan,
        date=0,
        timestr=0,
        stamp=0,
    ):
        if date != 0:
            self.epoch_timestamp = self.convert_timestr_to_epoch(
                date,
                timestr,
            )
        else:
            self.epoch_timestamp = stamp
        self.depth = float(depth)
        self.altitude = float(altitude)
        self.lon = float(lon)
        self.lat = float(lat)
    
    def convert_timestr_to_epoch(self, date, timestr):
        while len(date) < 13:
            date = ' ' + date
        yyyy = 2000 + int(date[5:7])
        mm = int(date[8:10])
        dd = int(date[11:13])
        hour = int(timestr[0:2])
        mins = int(timestr[3:5])
        secs = int(timestr[6:8])
        if hour < 0:
            hour = 0
            mins = 0
            secs = 0
        epoch_time = date_time_to_epoch(
            yyyy,
            mm,
            dd,
            hour,
            mins,
            secs,
            0,
            0,
        )
        return epoch_time


def epoch_to_timestr(epoch: Union[int, float]) -> str:
    """
    Convert epoch number into time string.

    Args:
        epoch (int | float): time since epoch [s]

    Returns:
        str: date as YYYYMMDD_hhmmss
    """
    time = datetime.fromtimestamp(epoch)
    timestr = (
        f"{time.year:04}{time.month:02}{time.day:02}"
        f"_{time.hour:02}{time.minute:02}{time.second:02}"
    )
    return timestr


class RovParser:
    """
    Class to parse and write koyo20rov nav data.

    Attributes:
        processed_dive_path (PosixPath) : path to dive folder in processed
        filepath_pos1 (PosixPath) : path to the 1st ship_log file
        filepath_pos2 (PosixPath) : path to the 2nd ship_log file
        filepath_rot (PosixPath) : path to the TCM file
        filepath_LM165 (PosixPath) : path to the LM165 FileTime file
        filepath_Xviii (PosixPath) : path to the Xviii FileTime file
        filepath_mission_yaml (PosixPath) : path to the mission.yaml
        filepath_vehicle_yaml (PosixPath) : path to the vehicle.yaml

        ~ AFTER load_data IS CALLED ~
        sensor_positions (dict) : arrays for each sensor's vehicle position
        ref_lat (float) : reference latitude from the mission.yaml
        ref_lon (float) : reference longitude from the mission.yaml
        name_camA (str) : name of the first camera
        name_camB (str) : name of the second camera
        rel_dirpath_camA (str) : rel path to camA images from raw dive
        rel_dirpath_camB (str) : rel path to camB images from raw dive
        rel_dirpath_LM165 (str) : rel path to LM165 images from raw dive
        vector_pos (list) : RovPos objects
        vector_rot (list) : RovRot objects
        vector_LM165_at_DVL (list) : RovCam objects
        vector_Xviii (list) : RovCam objects

        ~ AFTER check_for_outputs IS CALLED ~
        outpath_camA (PosixPath) : output path to camA nav file
        outpath_camB (PosixPath) : output path to camB nav file
        outpath_LM165 (PosixPath) : output path to LM165@dvl nav file

        ~ AFTER add_lever_arms IS CALLED ~
        vector_camA (list) : RovCam objects
        vector_camB (list) : RovCam objects
    """

    def __init__(self,
        unspecified_dive_path: str,
    ):
        """
        Check all paths at initialisation, check for preexisting
        output files and whether force overwrite is enabled.

        Args:
            dive_path (str): dirpath to the target dive folder
        """
        # Obtain and check all input paths.
        unspecified_dive_path = Path(unspecified_dive_path).resolve()
        assert unspecified_dive_path.exists()
        raw_dive_path = get_raw_folder(unspecified_dive_path)
        self.processed_dive_path = get_processed_folder(
            unspecified_dive_path,
        )

        self.filepath_pos1 = (
            raw_dive_path
            / "nav/ship_log/201102.csv"
        ).resolve()
        assert self.filepath_pos1.exists()

        self.filepath_pos2 = (
            raw_dive_path
            / "nav/ship_log/201103.csv"
        ).resolve()
        assert self.filepath_pos2.exists()

        self.filepath_rot = (
            raw_dive_path
            / "nav/TCM/TCM.csv"
        ).resolve()
        assert self.filepath_rot.exists()

        self.filepath_LM165 = (
            raw_dive_path
            / "image/LM165/FileTime.csv"
        ).resolve()
        assert self.filepath_LM165.exists()

        self.filepath_Xviii = (
            raw_dive_path
            / "image/Xviii/FileTime.csv"
        ).resolve()
        assert self.filepath_Xviii.exists()

        self.vehicle_yaml_filepath = (
            raw_dive_path
            / "vehicle.yaml"
        ).resolve()
        assert self.vehicle_yaml_filepath.exists()

        self.mission_yaml_filepath = (
            raw_dive_path
            / "mission.yaml"
        )
        assert self.mission_yaml_filepath.exists()
        return

    def load_data(self):
        """
        Load in all data from .yaml files, image files, and navigation
        files.
        """
        # Read in all data, concatenate position files, and assign data
        # to variable names.
        Console.info(
            f"Loading vehicle.yaml at {self.vehicle_yaml_filepath}"
        )
        vehicle = Vehicle(self.vehicle_yaml_filepath)
        self.sensor_positions = {}
        for key in vehicle.data:
            x_pos = vehicle.data[key]["surge_m"]
            y_pos = vehicle.data[key]["sway_m"]
            z_pos = vehicle.data[key]["heave_m"]
            self.sensor_positions[key] = np.array([x_pos, y_pos, z_pos])

        Console.info(
            f"Loading mission.yaml at {self.mission_yaml_filepath}"
        )
        mission = Mission(self.mission_yaml_filepath)
        self.ref_lat = mission.origin.latitude
        self.ref_lon = mission.origin.longitude
        self.name_camA = mission.image.cameras[0].name
        self.name_camB = mission.image.cameras[1].name
        self.rel_dirpath_camA = mission.image.cameras[0].path
        self.rel_dirpath_camB = mission.image.cameras[1].path
        self.rel_dirpath_LM165 = mission.image.cameras[2].path

        dataframe_pos1 = pd.read_csv(
            self.filepath_pos1,
            encoding="shift_jis",
            header=None,
        )
        dataframe_pos2 = pd.read_csv(
            self.filepath_pos2,
            encoding="shift_jis",
            header=None,
        )
        dataframe_pos = pd.concat(
            [dataframe_pos1, dataframe_pos2],
            ignore_index=True,
        )
        Console.info(f"Found {len(dataframe_pos)} position records!")
        #  C :  2 : date
        #  D :  3 : timestr
        #  X : 23 : depth
        #  Y : 24 : altitude
        # AF : 31 : lat
        # AG : 32 : long
        date = list(dataframe_pos[2])
        timestr = list(dataframe_pos[3])
        depth = list(dataframe_pos[23])
        altitude = list(dataframe_pos[24])
        lat = list(dataframe_pos[31])
        lon = list(dataframe_pos[32])
        self.vector_pos = []
        for i, this_date in enumerate(date):
            if type(this_date) is not str:
                continue
            elif len(this_date) < 8:
                continue
            else:
                self.vector_pos.append(
                    RovPos(
                        depth[i],
                        altitude[i],
                        lon[i],
                        lat[i],
                        this_date,
                        timestr[i],
                    )
                )
        
        def return_epoch_timestamp(rovpos_obj):
            return rovpos_obj.epoch_timestamp
        # Sort by timestamp! (just in case)
        self.vector_pos = sorted(
            self.vector_pos,
            key=return_epoch_timestamp,
        )
        # Deal with missing values
        for i in range(len(self.vector_pos) - 1):
            if self.vector_pos[i].altitude == 0:
                self.vector_pos[i].altitude = interpolate(
                    self.vector_pos[i].epoch_timestamp,
                    self.vector_pos[i - 1].epoch_timestamp,
                    self.vector_pos[i + 1].epoch_timestamp,
                    self.vector_pos[i - 1].altitude,
                    self.vector_pos[i + 1].altitude,
                )
            if self.vector_pos[i].depth == 0:
                self.vector_pos[i].depth = interpolate(
                    self.vector_pos[i].epoch_timestamp,
                    self.vector_pos[i - 1].epoch_timestamp,
                    self.vector_pos[i + 1].epoch_timestamp,
                    self.vector_pos[i - 1].depth,
                    self.vector_pos[i + 1].depth,
                )

        dataframe_rot = pd.read_csv(self.filepath_rot)
        Console.info(f"Found {len(dataframe_rot)} rotation records!")
        rot_date = list(dataframe_rot["date(YYYYMMDD)"])
        rot_time = list(dataframe_rot["time(hhmmss)"])
        timems = list(dataframe_rot["time_ms(iiii)"])
        heading = list(dataframe_rot["heading(degrees)"])
        pitch = list(dataframe_rot["pitch(degrees)"])
        roll = list(dataframe_rot["roll(degrees)"])
        self.vector_rot = []
        for i, this_date in enumerate(rot_date):
            self.vector_rot.append(
                RovRot(
                    heading[i],
                    pitch[i],
                    roll[i],
                    this_date,
                    rot_time[i],
                    timems[i],
                )
            )

        dataframe_LM165 = pd.read_csv(self.filepath_LM165)
        Console.info(f"Found {len(dataframe_LM165)} LM165 image records!")
        date_LM165 = list(dataframe_LM165["date"])
        time_LM165 = list(dataframe_LM165["time"])
        index_LM165 = list(dataframe_LM165["index"])
        ms_LM165 = list(dataframe_LM165["ms"])
        self.vector_LM165_at_DVL = []
        for i, this_date in enumerate(date_LM165):
            self.vector_LM165_at_DVL.append(
                RovCam(
                    this_date,
                    time_LM165[i],
                    ms_LM165[i],
                    index_LM165[i],
                )
            )

        dataframe_Xviii = pd.read_csv(self.filepath_Xviii)
        Console.info(f"Found {len(dataframe_Xviii)} Xviii image records!")
        date_Xviii = list(dataframe_Xviii["date"])
        time_Xviii = list(dataframe_Xviii["time"])
        ms_Xviii = list(dataframe_Xviii["ms"])
        index_Xviii = list(dataframe_Xviii["index"])
        self.vector_Xviii = []
        for i, this_date in enumerate(date_Xviii):
            self.vector_Xviii.append(
                RovCam(
                    this_date,
                    time_Xviii[i],
                    ms_Xviii[i],
                    index_Xviii[i],
                )
            )
        return
    
    def check_for_outputs(self, forcing: bool):
        """
        Check for preexisting output files, exit if present and not
        forcing.

        Args:
            forcing (bool): whether forcing file overwrites or not
        """
        # Check load_data has been called
        assert hasattr(self, "vector_pos"), (
            "ERROR: RovParser.load_data() must be called first."
        )
        # Figure out beginning and end times
        epoch_start = min([
            self.vector_pos[0].epoch_timestamp,
            self.vector_rot[0].epoch_timestamp,
        ])
        epoch_end = max([
            self.vector_pos[-1].epoch_timestamp,
            self.vector_rot[-1].epoch_timestamp,
        ])
        timestr_start = epoch_to_timestr(epoch_start)
        timestr_end = epoch_to_timestr(epoch_end)
        # Check whether output files already exist
        dirpath_output = (
            self.processed_dive_path
            / f"json_renav_{timestr_start}_{timestr_end}"
            / "csv"
            / "dead_reckoning"
        ).resolve()
        dirpath_output.mkdir(parents=True, exist_ok=True)
        self.outpath_camA = (
            dirpath_output
            / f"auv_dr_{self.name_camA}.csv"
        )
        try:
            assert not self.outpath_camA.exists()
        except AssertionError:
            if not forcing:
                Console.error(
                    f"{self.outpath_camA.name} already exists at"
                    f" {self.outpath_camA.parent}. Use forcing to"
                    f" overwrite."
                )
                raise
            else:
                Console.warn(
                    f"{self.outpath_camA.name} will be overwritten at"
                    f" {self.outpath_camA.parent}"
                )
        self.outpath_camB = (
            dirpath_output
            / f"auv_dr_{self.name_camB}.csv"
        )
        try:
            assert not self.outpath_camB.exists()
        except AssertionError:
            if not forcing:
                Console.error(
                    f"{self.outpath_camB.name} already exists at"
                    f" {self.outpath_camB.parent}. Use forcing to"
                    f" overwrite."
                )
                raise
            else:
                Console.warn(
                    f"{self.outpath_camB.name} will be overwritten at"
                    f" {self.outpath_camB.parent}"
                )
        self.outpath_LM165_at_dvl = (
            dirpath_output
            / f"auv_dr_LM165_at_dvl.csv"
        )
        try:
            assert not self.outpath_LM165_at_dvl.exists()
        except AssertionError:
            if not forcing:
                Console.error(
                    f"{self.outpath_LM165_at_dvl.name} already exists at"
                    f" {self.outpath_LM165_at_dvl.parent}. Use forcing to"
                    f" overwrite."
                )
                raise
            else:
                Console.warn(
                    f"{self.outpath_LM165_at_dvl.name} will be overwritten at"
                    f" {self.outpath_LM165_at_dvl.parent}"
                )
        return

    def interpolate_vector_pos(self,
        vector_cam: list,
        name: str,
    ) -> list:
        """
        Interpolate positional data to camera image timestamps.

        Args:
            vector_cam (list): RovCam objects

        Returns:
            list: vector_cam with interpolated positional data
        """
        n = len(vector_cam)
        for j, image_object in enumerate(vector_cam):
            print(
                f"Interpolating {name} from position vectors"
                f" - {100*j/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_epoch = image_object.epoch_timestamp
            # If image later than all position data, just stop.
            if image_epoch > self.vector_pos[-1].epoch_timestamp:
                break
            # If image earlier than all position data, just move on.
            if image_epoch < self.vector_pos[0].epoch_timestamp:
                continue
            # Find the position data that immediately preceeds the image.
            i = 0
            while (image_epoch > self.vector_pos[i].epoch_timestamp):
                i+=1
            # Interpolate.
            image_object.lat = interpolate(
                image_epoch,
                self.vector_pos[i-1].epoch_timestamp,
                self.vector_pos[i].epoch_timestamp,
                self.vector_pos[i-1].lat,
                self.vector_pos[i].lat,
            )
            image_object.lon = interpolate(
                image_epoch,
                self.vector_pos[i-1].epoch_timestamp,
                self.vector_pos[i].epoch_timestamp,
                self.vector_pos[i-1].lon,
                self.vector_pos[i].lon,
            )
            image_object.depth = interpolate(
                image_epoch,
                self.vector_pos[i-1].epoch_timestamp,
                self.vector_pos[i].epoch_timestamp,
                self.vector_pos[i-1].depth,
                self.vector_pos[i].depth,
            )
            image_object.altitude = interpolate(
                image_epoch,
                self.vector_pos[i-1].epoch_timestamp,
                self.vector_pos[i].epoch_timestamp,
                self.vector_pos[i-1].altitude,
                self.vector_pos[i].altitude,
            )
        print(
            f"Interpolating {name} from position vectors"
            f" - {100*(j+1)/n:6.2f}%"
        )
        return vector_cam
    
    def interpolate_vector_rot(self,
        vector_cam: list,
        name: str,
    ) -> list:
        """
        Interpolate rotational data to camera image timestamps.

        Args:
            vector_cam (list): RovCam objects

        Returns:
            list: vector_cam with interpolated rotational data
        """
        n = len(vector_cam)
        for j, image_object in enumerate(vector_cam):
            print(
                f"Interpolating {name} from rotation vectors"
                f" - {100*j/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_epoch = image_object.epoch_timestamp
            # If image later than all rotation data, just stop.
            if image_epoch > self.vector_rot[-1].epoch_timestamp:
                break
            # If image earlier than all position data, just move on.
            if image_epoch < self.vector_rot[0].epoch_timestamp:
                continue
            # Find the position data that immediately preceeds the image.
            i = 0
            while (image_epoch > self.vector_rot[i].epoch_timestamp):
                i+=1
            # Interpolate.
            image_object.heading = interpolate(
                image_epoch,
                self.vector_rot[i-1].epoch_timestamp,
                self.vector_rot[i].epoch_timestamp,
                self.vector_rot[i-1].heading,
                self.vector_rot[i].heading,
            )
            image_object.pitch = interpolate(
                image_epoch,
                self.vector_rot[i-1].epoch_timestamp,
                self.vector_rot[i].epoch_timestamp,
                self.vector_rot[i-1].pitch,
                self.vector_rot[i].pitch,
            )
            image_object.roll = interpolate(
                image_epoch,
                self.vector_rot[i-1].epoch_timestamp,
                self.vector_rot[i].epoch_timestamp,
                self.vector_rot[i-1].roll,
                self.vector_rot[i].roll,
            )
        print(
            f"Interpolating {name} from rotation vectors"
            f" - {100*(j+1)/n:6.2f}%"
        )
        return vector_cam

    def interpolate_to_images(self):
        """
        Interpolate data to image timestamps.
        """
        # Now interpolate position vectors and rotation vectors to give
        # respective vectors at the timestamps of the LM165 images records
        # and then of the Xviii image records.
        Console.info("Interpolating data to image timestamps...")
        self.vector_LM165_at_DVL = self.interpolate_vector_pos(
            self.vector_LM165_at_DVL,
            "LM165",
        )
        self.vector_LM165_at_DVL = self.interpolate_vector_rot(
            self.vector_LM165_at_DVL,
            "LM165",
        )
        self.vector_Xviii = self.interpolate_vector_pos(
            self.vector_Xviii,
            "Xviii",
        )
        self.vector_Xviii = self.interpolate_vector_rot(
            self.vector_Xviii,
            "Xviii",
        )
        Console.info("...finished interpolating everything.")
        return
    
    def add_lever_arms(self):
        """
        Convert positional data of raw sensors to be that of the cameras
        (i.e. add lever arms).
        """
        Console.info("Adding lever arms...")
        # for LM165
        n = len(self.vector_LM165_at_DVL)
        for i, image_object in enumerate(self.vector_LM165_at_DVL):
            print(f" - for LM165 - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_object.add_lever_arms(
                "dvl",
                self.sensor_positions,
            )
        print(f" - for LM165 - {100*(i+1)/n:6.2f}%")
        # Create vectors for the two colour cameras
        self.vector_camA = deepcopy(self.vector_Xviii)
        self.vector_camB = deepcopy(self.vector_Xviii)
        # for camA
        n = len(self.vector_camA)
        for i, image_object in enumerate(self.vector_camA):
            print(f" - for camA - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_object.add_lever_arms(
                self.name_camA,
                self.sensor_positions,
            )
        print(f" - for camA - {100*(i+1)/n:6.2f}%")
        # for camB
        n = len(self.vector_camB)
        for i, image_object in enumerate(self.vector_camB):
            print(f" - for camB - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_object.add_lever_arms(
                self.name_camB,
                self.sensor_positions,
            )
        print(f" - for camB - {100*(i+1)/n:6.2f}%")
        Console.info("...finished adding lever arms.")
        return

    def add_northings_eastings(self):
        """
        Calculate northings and eastings.
        """
        Console.info("Calculating northings and eastings...")
        # for LM165
        n = len(self.vector_LM165_at_DVL)
        for i, image_object in enumerate(self.vector_LM165_at_DVL):
            print(
                f" - for LM165 - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            lateral_distance, bearing = latlon_to_metres(
                image_object.lat,
                image_object.lon,
                self.ref_lat,
                self.ref_lon,
            )
            image_object.northing = lateral_distance * np.cos(
                bearing * np.pi / 180.0
            )
            image_object.easting = lateral_distance * np.sin(
                bearing * np.pi / 180.0
            )
        print(f" - for LM165 - {100*(i+1)/n:6.2f}%")
        # for camA
        n = len(self.vector_camA)
        for i, image_object in enumerate(self.vector_camA):
            print(
                f" - for camA - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            lateral_distance, bearing = latlon_to_metres(
                image_object.lat,
                image_object.lon,
                self.ref_lat,
                self.ref_lon,
            )
            image_object.northing = lateral_distance * np.cos(
                bearing * np.pi / 180.0
            )
            image_object.easting = lateral_distance * np.sin(
                bearing * np.pi / 180.0
            )
        print(f" - for camA - {100*(i+1)/n:6.2f}%")
        # for camB
        n = len(self.vector_camB)
        for i, image_object in enumerate(self.vector_camB):
            print(
                f" - for camB - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            lateral_distance, bearing = latlon_to_metres(
                image_object.lat,
                image_object.lon,
                self.ref_lat,
                self.ref_lon,
            )
            image_object.northing = lateral_distance * np.cos(
                bearing * np.pi / 180.0
            )
            image_object.easting = lateral_distance * np.sin(
                bearing * np.pi / 180.0
            )
        print(f" - for camB - {100*(i+1)/n:6.2f}%")
        Console.info("...finished calculating northings and eastings.")
        return

    def write_outputs(self):
        """
        Create .csv file text and write output files.
        """
        Console.info("Generating .csv data strings...")
        headers = [
            "relative_path,",
            "northing [m],",
            "easting [m],",
            "depth [m],",
            "roll [deg],",
            "pitch [deg],",
            "heading [deg],",
            "altitude [m],",
            "timestamp [s],",
            "latitude [deg],",
            "longitude [deg]\n",
        ]
        self.data_for_LM165_at_DVL = deepcopy(headers)
        n = len(self.vector_LM165_at_DVL)
        for i, image_object in enumerate(self.vector_LM165_at_DVL):
            print(
                f" - for LM165_at_DVL - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_path = (
                f"{self.rel_dirpath_LM165}"
                + "/" + f"{image_object.index:07}"[1:4]
                + f"/image{image_object.index:07}.tif"
            )
            msg = (
                f"{image_path}, {image_object.northing},"
                f" {image_object.easting}, {image_object.depth},"
                f" {image_object.roll}, {image_object.pitch},"
                f" {image_object.heading}, {image_object.altitude},"
                f" {image_object.epoch_timestamp}, {image_object.lat},"
                f" {image_object.lon}\n"
            )
            self.data_for_LM165_at_DVL.append(msg)
        print(f" - for LM165_at_DVL - {100*(i+1)/n:6.2f}%")

        self.data_for_camA = deepcopy(headers)
        n = len(self.vector_camA)
        for i, image_object in enumerate(self.vector_camA):
            print(
                f" - for camA - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_path = (
                f"{self.rel_dirpath_camA}"
                + f"/{image_object.index:07}.raw"
            )
            msg = (
                f"{image_path}, {image_object.northing},"
                f" {image_object.easting}, {image_object.depth},"
                f" {image_object.roll}, {image_object.pitch},"
                f" {image_object.heading}, {image_object.altitude},"
                f" {image_object.epoch_timestamp}, {image_object.lat},"
                f" {image_object.lon}\n"
            )
            self.data_for_camA.append(msg)
        print(f" - for camA - {100*(i+1)/n:6.2f}%")
        
        self.data_for_camB = deepcopy(headers)
        n = len(self.vector_camB)
        for i, image_object in enumerate(self.vector_camB):
            print(
                f" - for camB - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_path = (
                f"{self.rel_dirpath_camB}"
                + f"/{image_object.index:07}.raw"
            )
            msg = (
                f"{image_path}, {image_object.northing},"
                f" {image_object.easting}, {image_object.depth},"
                f" {image_object.roll}, {image_object.pitch},"
                f" {image_object.heading}, {image_object.altitude},"
                f" {image_object.epoch_timestamp}, {image_object.lat},"
                f" {image_object.lon}\n"
            )
            self.data_for_camB.append(msg)
        print(f" - for camB - {100*(i+1)/n:6.2f}%")
        Console.info("...finished generating .csv data strings.")

        Console.info("Writing outputs...")
        n = len(self.vector_LM165_at_DVL)
        i = 0
        with self.outpath_LM165_at_dvl.open("w", encoding="utf-8") as fileout:
            for line in self.data_for_LM165_at_DVL:
                print(
                    f" - for LM165_at_DVL - {100*i/n:6.2f}%",
                    end='\r',
                    flush=True,
                )
                fileout.write(str(line))
                i += 1
        print(f" - for LM165_at_DVL - {100*i/n:6.2f}%")

        n = len(self.vector_camA)
        i = 0
        with self.outpath_camA.open("w", encoding="utf-8") as fileout:
            for line in self.data_for_camA:
                print(
                    f" - for camA - {100*i/n:6.2f}%",
                    end='\r',
                    flush=True,
                )
                fileout.write(str(line))
                i += 1
        print(f" - for camA - {100*i/n:6.2f}%")

        n = len(self.vector_camB)
        i = 0
        with self.outpath_camB.open("w", encoding="utf-8") as fileout:
            for line in self.data_for_camB:
                print(
                    f" - for camB - {100*i/n:6.2f}%",
                    end='\r',
                    flush=True,
                )
                fileout.write(str(line))
                i += 1
        print(f" - for camB - {100*i/n:6.2f}%")
        Console.info("...finished writing outputs.")
        return

