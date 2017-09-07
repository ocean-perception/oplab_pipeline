# parse_acfr_images

# Scripts to parse acfr image acquisition data

# Author: Blair Thornton
# Date: 31/08/2017

import os, operator, sys
import codecs, time, json, glob
from datetime import datetime

epoch_timestamp_camera1 = []
epoch_timestamp_camera2 = []
values = []
data_list = []
tolerance = 0.01 # stereo pair must be within 10ms of each other

#http://www.json.org/

class parse_acfr_images:
	def __init__(self, filepath, sensor_string, camera1_label, camera2_label, category, timezone, timeoffset, ftype, outpath, fileoutname, fileout):

		# parser meta data
		class_string = 'measurement'
		frame_string = 'body'
		category = 'image'

		
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

		print('Parsing', sensor_string, 'images')
		
		# determine file paths
		
		all_list = os.listdir(filepath)

		camera1_filename = [ line for line in all_list if camera1_label in line and '.txt' not in line]
		camera2_filename = [ line for line in all_list if camera2_label in line and '.txt' not in line]
		
		data_list=[]
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
			epoch_timestamp = float(epoch_time+msec/1000+timeoffset)
			
			epoch_timestamp_camera1.append(str(epoch_timestamp))					
		
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
			epoch_timestamp = float(epoch_time+msec/1000+timeoffset)

			epoch_timestamp_camera2.append(str(epoch_timestamp))

		


		for i in range(len(camera1_filename)):
			# print(epoch_timestamp_camera1[i])
			values=[]
			for j in range(len(camera2_filename)):
				# print(epoch_timestamp_camera2[j])
					values.append(abs(float(epoch_timestamp_camera1[i])-float(epoch_timestamp_camera2[j])))
									
			(sync_difference,sync_pair) = min((v,k) for k,v in enumerate(values))		
						
			if sync_difference < tolerance:
				if ftype == 'oplab':					
					data = {'epoch_timestamp': float(epoch_timestamp_camera1[i]), 'class': class_string, 'sensor': sensor_string, 'frame': frame_string, 'category': category, 'camera1': [{'epoch_timestamp': float(epoch_timestamp_camera1[i]), 'filename': str(camera1_filename[i])}], 'camera2':  [{'epoch_timestamp': float(epoch_timestamp_camera2[sync_pair]), 'filename': str(camera2_filename[sync_pair])}]}
					data_list.append(data)
				if ftype == 'acfr':
					data = 'VIS: ' + str(float(epoch_timestamp_camera1[i])) + ' [' + str(float(epoch_timestamp_camera1[i])) + '] ' + str(camera1_filename[i]) + ' exp: 0\n'
					fileout.write(data)
					data = 'VIS: ' + str(float(epoch_timestamp_camera2[sync_pair])) + ' [' + str(float(epoch_timestamp_camera2[sync_pair])) + '] ' + str(camera2_filename[sync_pair]) + ' exp: 0\n'
					fileout.write(data)

		if ftype == 'oplab':
			fileout.close()
			for filein in glob.glob(outpath + '/' + fileoutname):
				try:
					with open(filein, 'rb') as json_file:						
						data_in=json.load(json_file)						
						for i in range(len(data_in)):
							data_list.insert(0,data_in[len(data_in)-i-1])				        
						
				except ValueError:					
					print('Initialising JSON file')

			with open(outpath + '/' + fileoutname,'w') as fileout:
				json.dump(data_list, fileout)	
				del data_list
