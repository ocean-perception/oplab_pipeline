# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, University of Southampton
All rights reserved.
"""

from datetime import datetime, timedelta
from pytz import reference
import calendar
import time


def date_time_to_epoch(yyyy, mm, dd, hh, mm1, ss, timezone_offset_to_utc=0):
    utc_date_time = (datetime(yyyy, mm, dd, hh, mm1, ss)
                     - timedelta(hours=timezone_offset_to_utc))
    epochtime = calendar.timegm(utc_date_time.timetuple())
    return epochtime


def epoch_to_localtime(epochtime):
    localtime = time.localtime(epochtime)
    return localtime


def epoch_to_utc(epochtime):
    utc_time = time.gmtime(epochtime)
    return utc_time


def get_localtimezone():
    localtimezone = reference.LocalTimezone().tzname(datetime.now())  # string
    return localtimezone


def epoch_to_day(epoch):
    return time.strftime('%Y/%m/%d',
                         time.localtime(epoch))


def string_to_epoch(datetime):
    yyyy = int(datetime[0:4])
    mm = int(datetime[4:6])
    dd = int(datetime[6:8])

    hours = int(datetime[8:10])
    mins = int(datetime[10:12])
    secs = int(datetime[12:14])

    return date_time_to_epoch(yyyy, mm, dd, hours, mins, secs)


def epoch_from_json(json):
    epoch_timestamp = json['epoch_timestamp']
    #start_datetime = time.strftime(
    #    '%Y%m%d%H%M%S', time.localtime(epoch_timestamp))
    #return string_to_epoch(start_datetime)
    return epoch_timestamp


def epoch_to_datetime(epoch_timestamp):
    return time.strftime('%Y%m%d%H%M%S',
                         time.gmtime(epoch_timestamp))


def read_timezone(timezone):
    if isinstance(timezone, str):
        if timezone == 'utc' or timezone == 'UTC':
            timezone_offset = 0.
        elif timezone == 'jst' or timezone == 'JST':
            timezone_offset = 9.
    else:
        try:
            timezone_offset = float(timezone)
        except ValueError:
            print('Error: timezone', timezone, 'in mission.yaml not \
                  recognised, please enter value from UTC in hours')
            return
    return timezone_offset
