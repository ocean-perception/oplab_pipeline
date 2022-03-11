# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""


from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.latlon_wgs84 import latlon_to_metres
from auv_nav.tools.time_conversions import date_time_to_epoch
from oplab import Console, Mission, Vehicle, get_raw_folder
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
            (
                positions[target]["surge_m"]
                - positions["depth"]["surge_m"]
            ),
            (
                positions[target]["sway_m"]
                - positions["depth"]["sway_m"]
            ),
            (
                positions[target]["heave_m"]
                - positions["depth"]["heave_m"]
            ),
        )[2]
        self.altitude = self.altitude - body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            (
                positions[target]["surge_m"]
                - positions["dvl"]["surge_m"]
            ),
            (
                positions[target]["sway_m"]
                - positions["dvl"]["sway_m"]
            ),
            (
                positions[target]["heave_m"]
                - positions["dvl"]["heave_m"]
            ),
        )[2]
        (self.lat, self.lon) = (self.lat, self.lon) + body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            (
                positions[target]["surge_m"]
                - positions["ins"]["surge_m"]
            ),
            (
                positions[target]["sway_m"]
                - positions["ins"]["sway_m"]
            ),
            (
                positions[target]["heave_m"]
                - positions["ins"]["heave_m"]
            ),
        )[:2]
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
        # Check all output paths
        dirpath_output = (
            # [base]/raw/[yr]/[sys]/[veh]/[dive]/nav/
            # Need to take 6 steps back to get to [base].
            nav_dirpath.parents[5]
            # Step forward into "processed".
            / "processed"  # [base]/processed
            # Then need to take 4 of the 6 steps forward again.
            / nav_dirpath.parts[-5]  # [base]/processed/[yr]
            / nav_dirpath.parts[-4]  # [base]/processed/[yr]/[sys]
            / nav_dirpath.parts[-3]  # [base]/processed/[yr]/[sys]/[veh]
            / nav_dirpath.parts[-2]  # [base]/processed/[yr]/[sys]/[veh]/[dive]
            # Then the new bits.
            / "json_whatever_you_like"
            / "csv"
            / "dead_reckoning"
        ).resolve()
        # Check we did that properly.
        assert dirpath_output.parents[2].exists()
        # Make new folders if need to.
        dirpath_output.mkdir(parents=True, exist_ok=True)
        # Check image folders exist
        vis_cam_dirpath = (
            nav_dirpath
            / ".."
            / "image"
            / "Xviii"
            / "Cam303235077"
        ).resolve()
        assert vis_cam_dirpath.exists()
        laser_cam_dirpath = (
            nav_dirpath
            / ".."
            / "image"
            / "LM165"
        )
        assert laser_cam_dirpath.exists()
        # Check that output files we're going to make don't already exist
        cam_csv_filepath = (
            dirpath_output
            / "auv_dr_Cam303235077.csv"
        )
        assert not cam_csv_filepath.exists()
        laser_csv_filepath = (
            dirpath_output
            / "auv_dr_LM165_at_dvl.csv"
        )
        assert not laser_csv_filepath.exists()
        # ============================================================================================
        filepath = get_processed_folder(args.output_folder)
        start_datetime = epoch_to_datetime(parser.start_epoch)
        finish_datetime = epoch_to_datetime(parser.finish_epoch)

        # make path for processed outputs
        json_filename = (
            "json_renav_"
            + start_datetime[0:8]
            + "_"
            + start_datetime[8:14]
            + "_"
            + finish_datetime[0:8]
            + "_"
            + finish_datetime[8:14]
        )
        renavpath = filepath / json_filename
        if not renavpath.is_dir():
            try:
                renavpath.mkdir()
            except Exception as e:
                print("Warning:", e)
        elif renavpath.is_dir() and not force_overwite:
            # Check if dataset has already been processed
            Console.error(
                "It looks like this dataset has already been processed for the",
                "specified time span.",
            )
            Console.error("The following directory already exist: {}".format(renavpath))
            Console.error(
                "To overwrite the contents of this directory rerun auv_nav with",
                "the flag -F.",
            )
            Console.error("Example:   auv_nav process -F PATH")
            Console.quit("auv_nav process would overwrite json_renav files")

        # ====================================================================================================
        return

    def interpolate_to_images(self):
        """
        Interpolate data to image timestamps.

        SHOULD BE ABLE TO SPLIT OFF A FUNCTION TO CALL 4 TIMES.
        """
        # Now interpolate position vectors and rotation vectors to give
        # respective vectors at the timestamps of the LM165 images records
        # and then of the Xviii image records.

        # interpolate LM165 from position vectors
        n = len(self.vector_LM165_at_DVL)
        for j, image_object in enumerate(self.vector_LM165_at_DVL):
            print(
                f"Interpolating LM165 from position vectors"
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
            f"Interpolating LM165 from position vectors"
            f" - {100*(j+1)/n:6.2f}%"
        )

        # interpolate LM165 from rotation vectors
        n = len(self.vector_LM165_at_DVL)
        for j, image_object in enumerate(self.vector_LM165_at_DVL):
            print(
                f"Interpolating LM165 from rotation vectors"
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
            f"Interpolating LM165 from rotation vectors"
            f" - {100*(j+1)/n:6.2f}%"
        )
        
        # interpolate Xvii from position vectors
        n = len(self.vector_Xviii)
        for j, image_object in enumerate(self.vector_Xviii):
            print(
                f"Interpolating Xviii from position vectors"
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
            f"Interpolating Xviii from position vectors"
            f" - {100*(j+1)/n:6.2f}%"
        )

        # interpolate vector_Xvii from rotation vectors
        n = len(self.vector_Xviii)
        for j, image_object in enumerate(self.vector_Xviii):
            print(
                f"Interpolating Xviii from rotation vectors"
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
            f"Interpolating Xviii from rotation vectors"
            f" - {100*(j+1)/n:6.2f}%"
        )
        print("Finished interpolating everything.")
        return
    
    def add_lever_arms(self):
        """
        Convert positional data of raw sensors to be that of the cameras
        (i.e. add lever arms).
        """
        print("Adding lever arms...")
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
        self.vetcor_camB = deepcopy(self.vector_Xviii)
        # for camA
        n = len(self.vector_camA)
        for i, image_object in enumerate(self.vector_camA):
            print(f" - for camA - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_object.add_lever_arms(
                self.camA.name,
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
                self.camB.name,
                self.sensor_positions,
            )
        print(f" - for camB - {100*(i+1)/n:6.2f}%")
        return

    def add_northings_eastings(self):
        """
        Calculate northings and eastings.
        """
        print("Calculating northings and eastings...")
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
        return

    def write_outputs(self):
        """
        Create .csv file text and write output files.
        """
        # ==========================================================================================
        output_dr_centre_path = renavpath / "csv" / "dead_reckoning"
        if not output_dr_centre_path.exists():
            output_dr_centre_path.mkdir(parents=True)
        camera_name = "hybis_camera"
        output_dr_centre_path = output_dr_centre_path / ("auv_dr_" + camera_name + ".csv")
        parser.write(output_dr_centre_path)
        parser.write(output_dr_centre_path)
        Console.info("Output written to", output_dr_centre_path)
        # ===========================================================================================
        print("Generating .csv message strings...")
        self.data_for_LM165_at_DVL = [
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
        n = len(vector_LM165_at_DVL)
        for i, image_object in enumerate(vector_LM165_at_DVL):
            print(
                f" - for LM165 - {100*i/n}%",
                end='\r',
                flush=True,
            )
        print(f" - for LM165 - {100*(i+1)/n}%")
        self.data_for_Xviii = [
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
        n = len(vector_Xviii)
        for i, image_object in enumerate(vector_Xviii):
            print(
                f" - for Xviii - {100*i/n}%",
                end='\r',
                flush=True,
            )
            self.data_for_Xviii.append()
        print(f" - for Xviii - {100*(i+1)/n}%")
        return

