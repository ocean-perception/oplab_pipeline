from datetime import datetime, timedelta
from pytz import reference
import calendar, time

def date_time_to_epoch(yyyy,mm,dd,HH,MM,SS,timezone_offset_to_utc):
	utc_date_time = datetime(yyyy,mm,dd,HH,MM,SS) - timedelta(hours=timezone_offset_to_utc)
	epochtime = calendar.timegm(utc_date_time.timetuple())
	return epochtime

def epoch_to_localtime(epochtime):
	localtime = time.localtime(epochtime)
	return localtime

def get_localtimezone():
	localtimezone = reference.LocalTimezone().tzname(datetime.now()) # string
	return localtimezone