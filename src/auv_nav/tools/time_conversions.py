# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import time
from datetime import datetime, timedelta, timezone

import pytz

from oplab.console import Console


def date_time_to_epoch(
    yyyy,
    mm,
    dd,
    hh,
    mm1,
    ss,
    timezone_offset_to_utc=0,
    us=0,
):
    utc_date_time = datetime(yyyy, mm, dd, hh, mm1, ss, us) - timedelta(
        hours=timezone_offset_to_utc
    )
    epochtime = utc_date_time.timestamp()
    return epochtime


def epoch_to_localtime(epochtime):
    localtime = time.localtime(epochtime)
    return localtime


def epoch_to_utctime(epochtime):
    utctime = time.gmtime(epochtime)
    return utctime


# def epoch_to_datetime(epoch_timestamp):
#     return datetime.datetime.fromtimestamp(epoch_timestamp)


def get_localtimezone():
    localtimezone = datetime.now(timezone.utc).astimezone().tzinfo
    # and convert tostring
    return str(localtimezone)


def epoch_to_day(epoch):
    return time.strftime("%Y/%m/%d", time.localtime(epoch))


def string_to_epoch(datetime):
    yyyy = int(datetime[0:4])
    mm = int(datetime[4:6])
    dd = int(datetime[6:8])

    hours = int(datetime[8:10])
    mins = int(datetime[10:12])
    secs = int(datetime[12:14])

    return date_time_to_epoch(yyyy, mm, dd, hours, mins, secs)


def epoch_from_json(json):
    epoch_timestamp = json["epoch_timestamp"]
    # start_datetime = time.strftime(
    #    '%Y%m%d%H%M%S', time.localtime(epoch_timestamp))
    # return string_to_epoch(start_datetime)
    return epoch_timestamp


def epoch_to_datetime(epoch_timestamp):
    return time.strftime("%Y%m%d%H%M%S", time.gmtime(epoch_timestamp))


def read_timezone(timezone):
    if isinstance(timezone, str):
        if timezone == "utc" or timezone == "UTC":
            timezone_offset_h = 0.0
        elif timezone == "cet" or timezone == "CET":
            timezone_offset_h = 0.0
        elif timezone == "jst" or timezone == "JST":
            timezone_offset_h = 9.0
    else:
        try:
            timezone_offset_h = float(timezone)
        except ValueError:
            print(
                "Error: timezone",
                timezone,
                "in mission.yaml not \
                  recognised, please enter value from UTC in hours",
            )
            return
    return timezone_offset_h


def datetime_tz_to_epoch(naive_datetime, timezone_str):
    """Converts a timezoned datetime to UTC epoch

    Parameters
    ----------
    naive_datetime : datetime
        Datetime object with no timezone information
    timezone_str : str
        Timezone string

    Returns
    -------
    float
        Epoch time in UTC
    """
    if timezone_str not in pytz.all_timezones:
        Console.error("Valid timezones are:")
        Console.error(pytz.common_timezones)
        Console.quit("Timezone not recognised")
    local_time = pytz.timezone(timezone_str)
    local_datetime = local_time.localize(naive_datetime, is_dst=None)
    utc = pytz.timezone("UTC")
    utc_datetime = local_datetime.astimezone(utc)
    return utc_datetime.timestamp()
