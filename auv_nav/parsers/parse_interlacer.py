# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import json
from array import array
from operator import itemgetter

from oplab import Console


def sort_values(value, order_index=1):
    return zip(
        *sorted([(i, e) for i, e in enumerate(value)], key=itemgetter(order_index))
    )


# Scripts to interlace data
def parse_interlacer(outpath, filename):
    data = array("f")
    value = []
    data_ordered = []

    filepath = outpath / filename

    try:
        with filepath.open("r") as json_file:
            data = json.load(json_file)
            for i, data_packet in enumerate(data):
                value.append(str(float(data_packet["epoch_timestamp"])))

    except ValueError:
        Console.quit("Error: no data in JSON file")

    # sort data in order of epoch_timestamp
    sorted_index, sorted_items = sort_values(value)

    # store interlaced data in order of time
    for i in range(len(data)):
        data_ordered.append((data[sorted_index[i]]))

    # write out interlaced json file
    with filepath.open("w") as fileout:
        json.dump(data_ordered, fileout, indent=2)
        fileout.close()
