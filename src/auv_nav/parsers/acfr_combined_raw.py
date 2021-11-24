# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from auv_nav.sensors import (
    Altitude,
    BodyVelocity,
    Camera,
    Depth,
    InertialVelocity,
    Orientation,
    Other,
    Usbl,
)
from oplab import Console, get_processed_folder


class AcfrCombinedRawWriter:
    """
    This output format has four main sensor types:
    * RDI: mixed body velocity, orientation and altitude measurement
    * PHINS_COMPASS: for orientation
    * PAROSCI: for pressure sensor
    * VIS: for cameras
    * SSBL_FIX: for USBL or SSBL global localization
    """

    def __init__(self, mission, vehicle, filepath):
        self.mission = mission
        self.vehicle = vehicle
        self.filepath = filepath
        self.rdi_altitude = None
        self.rdi_orientation = None
        self.data = ""

        outpath = get_processed_folder(self.filepath)
        config_filename = outpath / "mission.cfg"

        outpath = outpath / "dRAWLOGS_cv"

        if not outpath.exists():
            outpath.mkdir(parents=True)

        self.nav_file = outpath / "combined.RAW.auv"

        with config_filename.open("w") as f:
            data = (
                "MAG_VAR_LAT "
                + str(float(self.mission.origin.latitude))
                + "\nMAG_VAR_LNG "
                + str(float(self.mission.origin.longitude))
                + '\nMAG_VAR_DATE "'
                + str(self.mission.origin.date)
                + '"'
                + "\nMAGNETIC_VAR_DEG "
                + str(float(0))
            )
            f.write(data)
        # keep the file opened
        self.f = self.nav_file.open("w")

    def rdi_ready(self):
        if self.rdi_altitude is not None and self.rdi_orientation is not None:
            return True
        else:
            return False

    def add(self, measurement):
        data = None
        if type(measurement) is BodyVelocity:
            if self.rdi_ready():
                data = measurement.to_acfr(self.rdi_altitude, self.rdi_orientation)
                self.rdi_orientation = None
                self.rdi_altitude = None
        elif type(measurement) is InertialVelocity:
            pass
        elif type(measurement) is Altitude:
            self.rdi_altitude = measurement
        elif type(measurement) is Depth:
            data = measurement.to_acfr()
        elif type(measurement) is Usbl:
            data = measurement.to_acfr()
        elif type(measurement) is Orientation:
            data = measurement.to_acfr()
            self.rdi_orientation = measurement
        elif type(measurement) is Other:
            pass
        elif type(measurement) is Camera:
            # Get rid of laser images
            if "xxx" in measurement.filename:
                pass
            else:
                data = measurement.to_acfr()
        else:
            Console.error(
                "AcfrConverter type {} not supported".format(type(measurement))
            )
        if data is not None:
            self.f.write(data)
