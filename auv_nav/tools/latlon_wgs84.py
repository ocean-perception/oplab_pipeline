# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License. 
See LICENSE.md file in the project root for full license information.  
"""

import math
from geographiclib.geodesic import Geodesic


def latlon_to_metres(latitude, longitude, latitude_reference, longitude_reference):
    ret = Geodesic.WGS84.Inverse(latitude, 
                                 longitude, 
                                 latitude_reference, 
                                 longitude_reference)
    distance = ret['s12']

    y = math.sin(longitude - longitude_reference) * math.cos(latitude)
    x = (math.cos(latitude_reference) * math.sin(latitude) 
         - math.sin(latitude_reference) * math.cos(latitude) 
         * math.cos(longitude - longitude_reference))

    bearing = math.degrees(math.atan2(y, x))

    while bearing > 360:
        bearing = bearing - 360
    while bearing < 0:
        bearing = bearing + 360

    return (distance, bearing)


def metres_to_latlon(latitude, longitude, eastings, northings):
    s12 = math.sqrt(eastings**2 + northings**2)
    azi1 = math.degrees(math.atan2(eastings, northings))
    ret = Geodesic.WGS84.Direct(latitude, 
                                longitude, 
                                azi1, 
                                s12)
    latitude_offset = ret['lat2']
    longitude_offset = ret['lon2']
    return (latitude_offset, longitude_offset)
