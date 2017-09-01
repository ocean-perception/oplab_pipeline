# parse_acfr_images

# Scripts to parse acfr image acquisition data

# Author: Blair Thornton
# Date: 31/08/2017

import os
from datetime import datetime
import sys, codecs, time, json
from array import array
epoch_timestamp_camera1 = array('f')
epoch_timestamp_camera2 = array('f')
#http://www.json.org/
data_list=[]
#need to make acfr parsers
class parse_acfr_images:
	def __init__(self, filepath, sensor_string, camera1_label, camera2_label, category, timezone, timeoffset, ftype, fileout):

		# parser meta data
		class_string = 'measurement'
		
		# read in timezone
		if isinstance(timezone, str):			
			if timezone == 'utc' or  timezone == 'UTC':
				timezone_offset = 0
			elif timezone == 'jst' or  timezone == 'JST':
				timezone_offset = 9;
			else:
				try:
					timezone_offset=float(timezone)
				except ValueError:
					print('Error: timezone', timezone, 'in mission.cfg not recognised, please enter value from UTC in hours')
					return

		# convert to seconds from utc
		timeoffset = -timezone_offset*60*60 + timeoffset 

		print('parsing ', sensor_string, ' images')
		
		# determine file paths
		
		all_list = os.listdir(filepath)

		camera1_filename = [ line for line in all_list if camera1_label in line]
		camera2_filename = [ line for line in all_list if camera2_label in line]
		
		for i in range(len(camera1_filename)):
			camera1_filename_split = camera1_filename[i].strip().split('_') 
			
			date_string = camera1_filename_split[1]
			time_string = camera1_filename_split[2]
			ms_time_string = camera1_filename_split[3]

			# read in date
			yyyy = int(date_string[0:4])
			mm = int(date_string[5:6])
			dd = int(date_string[7:8])

			# read in time			
			hour=int(time_string[0:2])
			mins=int(time_string[2:4])
			secs=int(time_string[4:6])
			msec=int(ms_time_string[0:3])

			dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
			time_tuple = dt_obj.timetuple()
			epoch_time = time.mktime(time_tuple)
			
			epoch_timestamp_camera1.append(float(epoch_time+msec/1000+timeoffset))

		
		for i in range(len(camera2_filename)):
			camera2_filename_split = camera2_filename[i].strip().split('_')
			
			date_string = camera2_filename_split[1]
			time_string = camera2_filename_split[2]
			ms_time_string = camera2_filename_split[3]

			# read in date
			yyyy = int(date_string[0:4])
			mm = int(date_string[5:6])
			dd = int(date_string[7:8])

			# read in time			
			hour=int(time_string[0:2])
			mins=int(time_string[2:4])
			secs=int(time_string[4:6])
			msec=int(ms_time_string[0:3])

			dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
			time_tuple = dt_obj.timetuple()
			epoch_time = time.mktime(time_tuple)
			epoch_timestamp_camera2.append(float(epoch_time+msec/1000+timeoffset))

		for i in range(len(camera1_filename)):
			
			value= min(epoch_timestamp_camera1[i]-epoch_timestamp_camera2)
			print(value)
			


