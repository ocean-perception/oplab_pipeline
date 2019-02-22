# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

import json
from array import array
from operator import itemgetter
from auv_nav.tools.console import Console


def sort_values(value, order_index=1):
    return zip(*sorted([(i, e) for i, e in enumerate(value)],
               key=itemgetter(order_index)))


# Scripts to interlace data
def parse_interlacer(ftype, outpath, filename):
    data = array('f')
    value = []
    data_original = []
    data_ordered = []

    filepath = outpath / filename

    if ftype == 'oplab':
        try:
            with filepath.open('r') as json_file:
                data = json.load(json_file)
                for i in range(len(data)):
                    data_packet = data[i]
                    value.append(str(float(data_packet['epoch_timestamp'])))

        except ValueError:
            Console.quit('Error: no data in JSON file')

        # sort data in order of epoch_timestamp
        sorted_index, sorted_items = sort_values(value)

        # store interlaced data in order of time
        for i in range(len(data)):
            data_ordered.append((data[sorted_index[i]]))

        # write out interlaced json file
        with filepath.open('w') as fileout:
            json.dump(data_ordered, fileout, indent=2)

    if ftype == 'acfr':
        try:
            with filepath.open('r') as acfr_file:
                for line in acfr_file.readlines():
                    line_split = line.strip().split(':')
                    line_split_tailed = line_split[1].strip().split(' ')
                    value.append(str(line_split_tailed[0]))
                    data_original.append(line)

        except ValueError:
            Console.quit('Error: no data in RAW.auv file')

        # sort data in order of epoch_timestamp
        sorted_index, sorted_items = sort_values(value)

        for i in range(len(data_original)):
            data_ordered.append(data_original[sorted_index[i]])

        # write out interlaced acfr file
        with filepath.open('w') as fileout:
            for i in range(len(data_original)):
                fileout.write(str(data_ordered[i]))

        fileout.close()
