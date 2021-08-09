# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from auv_nav.sensors import Category, Orientation
from oplab import Console


def parse_template(mission, vehicle, category, output_format, outpath):
    # Get your data from a file using mission paths, for example
    your_data = None

    # Let's say you want a new IMU, instance the measurement to work
    orientation = Orientation()

    data_list = []
    if category == Category.ORIENTATION:
        Console.info("... parsing orientation")
        for i in your_data:
            # Provide a parser in the sensors.py class
            orientation.from_your_data(i)
            data = orientation.export(output_format)
            data_list.append(data)

    return data_list
