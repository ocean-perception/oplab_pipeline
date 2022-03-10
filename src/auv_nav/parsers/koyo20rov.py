# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""


from pathlib import Path

import numpy as np
import pandas as pd

from auv_nav.tools.body_to_inertial import body_to_inertial
from auv_nav.tools.interpolate import interpolate
from auv_nav.tools.time_conversions import date_time_to_epoch
from oplab import Console, Vehicle, get_raw_folder


class Camera:
    """Class to parse and store camera FileTime data"""
    def __init__(self,
        date_Cam,
        time_Cam,
        ms_Cam,
        index,
        sensor_positions,
        name,
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
        self.sensor_positions=sensor_positions
        self.name=name
        
        if date_Cam != 0:
            self.epoch_timestamp = self.convert_times_to_epoch(
                date_Cam,
                time_Cam,
                ms_Cam,
            )
        else:
            self.epoch_timestamp = stamp
    
    def add_lever_arms(self):
        """Add the lever arms to depth, altitude, lat, and lon
        measurements.
        """
        self.depth = self.depth + body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            (
                self.positions[self.name]["surge_m"]
                - self.positions["depth"]["surge_m"]
            ),
            (
                self.positions[self.name]["sway_m"]
                - self.positions["depth"]["sway_m"]
            ),
            (
                self.positions[self.name]["heave_m"]
                - self.positions["depth"]["heave_m"]
            ),
        )[2]
        self.altitude = self.altitude - body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            (
                self.positions[self.name]["surge_m"]
                - self.positions["dvl"]["surge_m"]
            ),
            (
                self.positions[self.name]["sway_m"]
                - self.positions["dvl"]["sway_m"]
            ),
            (
                self.positions[self.name]["heave_m"]
                - self.positions["dvl"]["heave_m"]
            ),
        )[2]
        (self.lat, self.lon) = (self.lat, self.lon) + body_to_inertial(
            self.roll,
            self.pitch,
            self.heading,
            (
                self.positions[self.name]["surge_m"]
                - self.positions["ins"]["surge_m"]
            ),
            (
                self.positions[self.name]["sway_m"]
                - self.positions["ins"]["sway_m"]
            ),
            (
                self.positions[self.name]["heave_m"]
                - self.positions["ins"]["heave_m"]
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
    """Class to parse and write koyo20rov nav data."""

    def __init__(self,
        nav_dirpath,
        image_dirpath,  # unneeded?
        reference_lat,
        reference_lon,
    ):
        # Obtain and check all input paths.
        nav_dirpath = get_raw_folder(nav_dirpath)
        nav_dirpath = Path(nav_dirpath).resolve()
        assert nav_dirpath.exists()
        assert nav_dirpath.is_dir()
        filepath_pos1 = (nav_dirpath / "ship_log/201102.csv").resolve()
        assert filepath_pos1.exists()
        filepath_pos2 = (nav_dirpath / "ship_log/201103.csv").resolve()
        assert filepath_pos2.exists()
        filepath_rot = (nav_dirpath / "TCM/TCM.csv").resolve()
        assert filepath_rot.exists()
        filepath_LM165 = (
            nav_dirpath
            / "../image/LM165/FileTime.csv"
        ).resolve()
        assert filepath_LM165.exists()
        filepath_Xviii = (
            nav_dirpath
            / "../image/Xviii/FileTime.csv"
        ).resolve()
        assert filepath_Xviii.exists()
        vehicle_yaml_filepath = (
            nav_dirpath
            / ".."
            / "vehicle.yaml"
        ).resolve()
        assert vehicle_yaml_filepath.exists()

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

        # Read in all data, concatenate position files, and assign data to
        # variable names.
        Console.info("Loading vehicle.yaml at", vehicle_yaml_filepath)
        vehicle = Vehicle(vehicle_yaml_filepath)
        sensor_positions = {}
        for key in vehicle.data:
            x_pos = vehicle.data[key]["surge_m"]
            y_pos = vehicle.data[key]["sway_m"]
            z_pos = vehicle.data[key]["heave_m"]
            sensor_positions[key] = np.array([x_pos, y_pos, z_pos])

        dataframe_pos1 = pd.read_csv(
            filepath_pos1,
            encoding="shift_jis",
            header=None,
        )
        dataframe_pos2 = pd.read_csv(
            filepath_pos2,
            encoding="shift_jis",
            header=None,
        )
        dataframe_pos = pd.concat(
            [dataframe_pos1, dataframe_pos2],
            ignore_index=True,
        )
        print(f"Found {len(dataframe_pos)} position records!")
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
        pos_vector = []
        for i, this_date in enumerate(date):
            if type(this_date) is not str:
                continue
            elif len(this_date) < 8:
                continue
            else:
                pos_vector.append(
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
        pos_vector = sorted(pos_vector, key=return_epoch_timestamp)
        # Deal with missing values
        for i in range(len(pos_vector) - 1):
            if pos_vector[i].altitude == 0:
                pos_vector[i].altitude = interpolate(
                    pos_vector[i].epoch_timestamp,
                    pos_vector[i - 1].epoch_timestamp,
                    pos_vector[i + 1].epoch_timestamp,
                    pos_vector[i - 1].altitude,
                    pos_vector[i + 1].altitude,
                )
            if pos_vector[i].depth == 0:
                pos_vector[i].depth = interpolate(
                    pos_vector[i].epoch_timestamp,
                    pos_vector[i - 1].epoch_timestamp,
                    pos_vector[i + 1].epoch_timestamp,
                    pos_vector[i - 1].depth,
                    pos_vector[i + 1].depth,
                )

        dataframe_rot = pd.read_csv(filepath_rot)
        print(f"Found {len(dataframe_rot)} rotation records!")
        rot_date = list(dataframe_rot["date(YYYYMMDD)"])
        rot_time = list(dataframe_rot["time(hhmmss)"])
        timems = list(dataframe_rot["time_ms(iiii)"])
        heading = list(dataframe_rot["heading(degrees)"])
        pitch = list(dataframe_rot["pitch(degrees)"])
        roll = list(dataframe_rot["roll(degrees)"])
        rot_vector = []
        for i, this_date in enumerate(rot_date):
            rot_vector.append(
                RovRot(
                    heading[i],
                    pitch[i],
                    roll[i],
                    this_date,
                    rot_time[i],
                    timems[i],
                )
            )

        dataframe_LM165 = pd.read_csv(filepath_LM165)
        print(f"Found {len(dataframe_LM165)} LM165 image records!")
        date_LM165 = list(dataframe_LM165["date"])
        time_LM165 = list(dataframe_LM165["time"])
        index_LM165 = list(dataframe_LM165["index"])
        ms_LM165 = list(dataframe_LM165["ms"])
        vector_LM165 = []
        for i, this_date in enumerate(date_LM165):
            vector_LM165.append(
                Camera(
                    this_date,
                    time_LM165[i],
                    ms_LM165[i],
                    index_LM165[i],
                    sensor_positions,
                    "LM165",
                )
            )

        dataframe_Xviii = pd.read_csv(filepath_Xviii)
        print(f"Found {len(dataframe_Xviii)} Xviii image records!")
        date_Xviii = list(dataframe_Xviii["date"])
        time_Xviii = list(dataframe_Xviii["time"])
        ms_Xviii = list(dataframe_Xviii["ms"])
        index_Xviii = list(dataframe_Xviii["index"])
        vector_Xviii = []
        for i, this_date in enumerate(date_Xviii):
            vector_Xviii.append(
                Camera(
                    this_date,
                    time_Xviii[i],
                    ms_Xviii[i],
                    index_Xviii[i],
                    sensor_positions,
                    "Cam303235077",
                )
            )
        
        # Now interpolate position vectors and rotation vectors to give
        # respective vectors at the timestamps of the LM165 images records
        # and then of the Xviii image records.

        # interpolate LM165 from position vectors
        n = len(vector_LM165)
        for j, image_object in enumerate(vector_LM165):
            print(
                f"Interpolating LM165 from position vectors - {100*j/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_epoch = image_object.epoch_timestamp
            # If image later than all position data, just stop.
            if image_epoch > pos_vector[-1].epoch_timestamp:
                break
            # If image earlier than all position data, just move on.
            if image_epoch < pos_vector[0].epoch_timestamp:
                continue
            # Find the position data that immediately preceeds the image.
            i = 0
            while (image_epoch > pos_vector[i].epoch_timestamp):
                i+=1
            # Interpolate.
            image_object.lat = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].lat,
                pos_vector[i].lat,
            )
            image_object.lon = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].lon,
                pos_vector[i].lon,
            )
            image_object.depth = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].depth,
                pos_vector[i].depth,
            )
            image_object.altitude = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].altitude,
                pos_vector[i].altitude,
            )
        print(f"Interpolating LM165 from position vectors - {100*(j+1)/n:6.2f}%")

        # interpolate LM165 from rotation vectors
        n = len(vector_LM165)
        for j, image_object in enumerate(vector_LM165):
            print(
                f"Interpolating LM165 from rotation vectors - {100*j/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_epoch = image_object.epoch_timestamp
            # If image later than all rotation data, just stop.
            if image_epoch > rot_vector[-1].epoch_timestamp:
                break
            # If image earlier than all position data, just move on.
            if image_epoch < rot_vector[0].epoch_timestamp:
                continue
            # Find the position data that immediately preceeds the image.
            i = 0
            while (image_epoch > rot_vector[i].epoch_timestamp):
                i+=1
            # Interpolate.
            image_object.heading = interpolate(
                image_epoch,
                rot_vector[i-1].epoch_timestamp,
                rot_vector[i].epoch_timestamp,
                rot_vector[i-1].heading,
                rot_vector[i].heading,
            )
            image_object.pitch = interpolate(
                image_epoch,
                rot_vector[i-1].epoch_timestamp,
                rot_vector[i].epoch_timestamp,
                rot_vector[i-1].pitch,
                rot_vector[i].pitch,
            )
            image_object.roll = interpolate(
                image_epoch,
                rot_vector[i-1].epoch_timestamp,
                rot_vector[i].epoch_timestamp,
                rot_vector[i-1].roll,
                rot_vector[i].roll,
            )
        print(f"Interpolating LM165 from rotation vectors - {100*(j+1)/n:6.2f}%")
        
        # interpolate Xvii from position vectors
        n = len(vector_Xviii)
        for j, image_object in enumerate(vector_Xviii):
            print(
                f"Interpolating Xviii from position vectors - {100*j/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_epoch = image_object.epoch_timestamp
            # If image later than all position data, just stop.
            if image_epoch > pos_vector[-1].epoch_timestamp:
                break
            # If image earlier than all position data, just move on.
            if image_epoch < pos_vector[0].epoch_timestamp:
                continue
            # Find the position data that immediately preceeds the image.
            i = 0
            while (image_epoch > pos_vector[i].epoch_timestamp):
                i+=1
            # Interpolate.
            image_object.lat = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].lat,
                pos_vector[i].lat,
            )
            image_object.lon = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].lon,
                pos_vector[i].lon,
            )
            image_object.depth = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].depth,
                pos_vector[i].depth,
            )
            image_object.altitude = interpolate(
                image_epoch,
                pos_vector[i-1].epoch_timestamp,
                pos_vector[i].epoch_timestamp,
                pos_vector[i-1].altitude,
                pos_vector[i].altitude,
            )
        print(f"Interpolating Xviii from position vectors - {100*(j+1)/n:6.2f}%")

        # interpolate vector_Xvii from rotation vectors
        n = len(vector_Xviii)
        for j, image_object in enumerate(vector_Xviii):
            print(
                f"Interpolating Xviii from rotation vectors - {100*j/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_epoch = image_object.epoch_timestamp
            # If image later than all rotation data, just stop.
            if image_epoch > rot_vector[-1].epoch_timestamp:
                break
            # If image earlier than all position data, just move on.
            if image_epoch < rot_vector[0].epoch_timestamp:
                continue
            # Find the position data that immediately preceeds the image.
            i = 0
            while (image_epoch > rot_vector[i].epoch_timestamp):
                i+=1
            # Interpolate.
            image_object.heading = interpolate(
                image_epoch,
                rot_vector[i-1].epoch_timestamp,
                rot_vector[i].epoch_timestamp,
                rot_vector[i-1].heading,
                rot_vector[i].heading,
            )
            image_object.pitch = interpolate(
                image_epoch,
                rot_vector[i-1].epoch_timestamp,
                rot_vector[i].epoch_timestamp,
                rot_vector[i-1].pitch,
                rot_vector[i].pitch,
            )
            image_object.roll = interpolate(
                image_epoch,
                rot_vector[i-1].epoch_timestamp,
                rot_vector[i].epoch_timestamp,
                rot_vector[i-1].roll,
                rot_vector[i].roll,
            )
        print(f"Interpolating Xviii from rotation vectors - {100*(j+1)/n:6.2f}%")
        print("Finished interpolating everything.")

        print("Adding lever arms...")
        # for LM165
        n = len(vector_LM165)
        for i, image_object in enumerate(vector_LM165):
            print(f" - for LM165 - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_object.add_lever_arms()
        print(f" - for LM165 - {100*(i+1)/n:6.2f}%")
        # for Xviii
        n = len(vector_Xviii)
        for i, image_object in enumerate(vector_Xviii):
            print(f" - for Xviii - {100*i/n:6.2f}%",
                end='\r',
                flush=True,
            )
            image_object.add_lever_arms()
        print(f" - for Xviii - {100*(i+1)/n:6.2f}%")

        print("Saving out .csv files...")
        header = [
            "relative_path",
            # "northing [m]",
            # "easting [m]",
            "depth [m]",
            "roll [deg]",
            "pitch [deg]",
            "heading [deg]",
            "altitude [m]",
            "timestamp [s]",
            "latitude [deg]",
            "longitude [deg]",
            # etc
        ]
        # Start with simpler cam csv file
        


        return vector_LM165, vector_Xviii
