# parse_gaps

# Scripts to parse ixsea blue gaps data. Interpolates ship gps reading for valid underwater position measurements to determine accurate range and so measurement uncertainty

# Author: Blair Thornton
# Date: 31/08/2017

import os
from datetime import datetime
import sys, math
import time, json, glob
#http://www.json.org/
sys.path.append("..")
from lib_coordinates.latlon_wgs84 import latlon_to_metres
from lib_coordinates.latlon_wgs84 import metres_to_latlon

data_list=[]

class parse_gaps:
	def __init__(self, filepath, category, timezone, timeoffset, latitude_reference, longitude_reference, ftype, outpath, fileoutname, fileout):

		# parser meta data    
		class_string = 'measurement'
		sensor_string = 'gaps'
		frame_string = 'inertial'

		# define headers used in phins
		header_absolute = '<< $PTSAG' #georeferenced strings
		header_heading = '<< $HEHDT'

		# gaps std models
		distance_std_factor = 0.6/100 # usbl catalogue gaps spec

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

		# determine file paths
		all_list = os.listdir(filepath)
		gaps_list = [ line for line in all_list if '.dat' in line]
		print(str(len(gaps_list)) + ' GAPS files found')
		
		# extract data from files
		data_list=[]
		for i in range(len(gaps_list)):
			path_gaps = filepath + '/' + gaps_list[i]
			with open(path_gaps) as gaps:
				# initialise flag				
				flag_got_time = 0
				for line in gaps.readlines():
					line_split = line.strip().split('*') 
					line_split_no_checksum = line_split[0].strip().split(',')

					# keep on upating ship position to find the prior interpolation value of ship coordinates
					if line_split_no_checksum[0] == header_absolute:
						
						# start with a ship coordinate
						if line_split_no_checksum[6] == '0' and flag_got_time != 3:
							# read in date
							 yyyy = int(line_split_no_checksum[5])
							 mm = int(line_split_no_checksum[4])
							 dd = int(line_split_no_checksum[3])

							 # print(yyyy,mm,dd)
							 # read in time
							 time_string=str(line_split_no_checksum[2])
							 hour=int(time_string[0:2])
							 mins=int(time_string[2:4])
							 secs=int(time_string[4:6])
							 msec=int(time_string[7:10])
							 #print(hour,mins,secs)
							 dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
							 
							 time_tuple = dt_obj.timetuple()
							 epoch_time = time.mktime(time_tuple)
							 epoch_timestamp_ship_prior = epoch_time+msec/1000+timeoffset

							 # get position
							 latitude_string = line_split_no_checksum[7]
							 latitude_degrees_ship_prior = int(latitude_string[0:2])
							 latitude_minutes_ship_prior = float(latitude_string[2:10])
							 latitude_prior=latitude_degrees_ship_prior+latitude_minutes_ship_prior/60.0							 							

							 longitude_string = line_split_no_checksum[9]
							 longitude_degrees_ship_prior = int(longitude_string[0:3])
							 longitude_minutes_ship_prior = float(longitude_string[3:11])
							 longitude_prior=longitude_degrees_ship_prior+longitude_minutes_ship_prior/60.0

							 # flag raised to proceed
							 flag_got_time = 1

					if line_split_no_checksum[0] == header_heading and flag_got_time == 1:
						heading_ship_prior = float(line_split_no_checksum[1])						
						flag_got_time = 2

					# only consider data where an underwater body is measured
					if line_split_no_checksum[0] == header_absolute and flag_got_time == 2:
						if line_split_no_checksum[6] != '0':
							# check all 4 hydrophones are valid and needs to be a sensor measurement (not calculated
							if line_split_no_checksum[11] == 'F' and line_split_no_checksum[13] == '1': 
								# read in date
								yyyy = int(line_split_no_checksum[5])
								mm = int(line_split_no_checksum[4])
								dd = int(line_split_no_checksum[3])

								# read in time
								time_string=str(line_split_no_checksum[2])
								hour=int(time_string[0:2])
								mins=int(time_string[2:4])
								secs=int(time_string[4:6])
								msec=int(time_string[7:10])

								dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
								
								time_tuple = dt_obj.timetuple()
								epoch_time = time.mktime(time_tuple)
								epoch_timestamp = epoch_time+msec/1000+timeoffset
								 
								# get position
								latitude_string = line_split_no_checksum[7]
								latitude_degrees = int(latitude_string[0:2])
								latitude_minutes = float(latitude_string[2:10])

								longitude_string = line_split_no_checksum[9]
								longitude_degrees = int(longitude_string[0:3])
								longitude_minutes = float(longitude_string[3:11])

								depth = float(line_split_no_checksum[12])

								# flag raised to proceed
								flag_got_time = 3						
							else:
								flag_got_time = 0

					if line_split_no_checksum[0] == header_absolute and flag_got_time == 3:
						# find next ship coordinate
						if line_split_no_checksum[6] == '0':
							
							# read in date
							yyyy = int(line_split_no_checksum[5])
							mm = int(line_split_no_checksum[4])
							dd = int(line_split_no_checksum[3])

							# read in time
							time_string=str(line_split_no_checksum[2])
							hour=int(time_string[0:2])
							mins=int(time_string[2:4])
							secs=int(time_string[4:6])
							msec=int(time_string[7:10])

							# calculate epoch time
							dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
							time_tuple = dt_obj.timetuple()
							epoch_time = time.mktime(time_tuple)
							epoch_timestamp_ship_posterior = epoch_time+msec/1000+timeoffset

							# get position
							latitude_string = line_split_no_checksum[7]
							latitude_degrees_ship_posterior = int(latitude_string[0:2])
							latitude_minutes_ship_posterior = float(latitude_string[2:10])
							latitude_posterior=latitude_degrees_ship_posterior+latitude_minutes_ship_posterior/60.0

							longitude_string = line_split_no_checksum[9]
							longitude_degrees_ship_posterior = int(longitude_string[0:3])
							longitude_minutes_ship_posterior = float(longitude_string[3:11])
							longitude_posterior=longitude_degrees_ship_posterior+longitude_minutes_ship_posterior/60.0

							# flag raised to proceed
							flag_got_time = 4

					if line_split_no_checksum[0] == header_heading and flag_got_time == 4:

						heading_ship_posterior = float(line_split_no_checksum[1])
						
						# interpolate for the ships location and heading					
						inter_time=(epoch_timestamp-epoch_timestamp_ship_prior)/(epoch_timestamp_ship_posterior-epoch_timestamp_ship_prior)
						longitude_ship = inter_time*(longitude_posterior-longitude_prior)+longitude_prior
						latitude_ship = inter_time*(latitude_posterior-latitude_prior)+latitude_prior
						heading_ship = inter_time*(heading_ship_posterior-heading_ship_prior)+heading_ship_prior

						while heading_ship > 360:
							heading_ship = heading_ship - 360
						while heading_ship < 0:
							heading_ship = heading_ship + 360

						# determine range to input to uncertainty model
						latitude = latitude_degrees+latitude_minutes/60.0
						longitude = longitude_degrees+longitude_minutes/60.0
						
						lateral_distance,bearing = latlon_to_metres(latitude, longitude, latitude_ship, longitude_ship)

						distance = math.sqrt(lateral_distance * lateral_distance + depth * depth)
						distance_std = distance_std_factor*distance

						# determine uncertainty in terms of latitude and longitude
						latitude_offset,longitude_offset = metres_to_latlon(latitude, longitude, distance_std, distance_std)

						latitude_std = latitude - latitude_offset
						longitude_std = longitude - longitude_offset

						# calculate in metres from reference
						lateral_distance_ship,bearing_ship = latlon_to_metres(latitude_ship, longitude_ship, latitude_reference, longitude_reference)
						eastings_ship = math.sin(bearing_ship*math.pi/180.0)*lateral_distance_ship
						northings_ship = math.cos(bearing_ship*math.pi/180.0)*lateral_distance_ship

						lateral_distance_target,bearing_target = latlon_to_metres(latitude, longitude, latitude_reference, longitude_reference)
						eastings_target = math.sin(bearing_target*math.pi/180.0)*lateral_distance_target
						northings_target = math.cos(bearing_target*math.pi/180.0)*lateral_distance_target

						# reset flag
						flag_got_time = 0

						if ftype == 'oplab':
							data = {'epoch_timestamp': float(epoch_timestamp), 'class': class_string, 'sensor': sensor_string, 'frame': frame_string, 'category': category, 'data_ship': [{'latitude': float(latitude_ship), 'longitude': float(longitude_ship)}, {'northings': float(northings_ship), 'eastings': float(eastings_ship)}, {'heading': float(heading_ship)}], 'data_target': [{'latitude': float(latitude), 'latitude_std': float(latitude_std)}, {'longitude': float(longitude), 'longitude_std': float(longitude_std)}, {'northings': float(northings_target), 'northings_std': float(distance_std)}, {'eastings': float(eastings_target), 'eastings_std': float(distance_std)}, {'depth': float(depth)}]}
							data_list.append(data)

						if ftype == 'acfr':
							data = 'SSBL_FIX: ' + str(float(epoch_timestamp)) + ' ship_x: ' + str(float(northings_ship)) + ' ship_y: ' + str(float(eastings_ship)) + ' target_x: ' + str(float(northings_target)) + ' target_y: ' + str(float(eastings_target)) + ' target_z: ' + str(float(depth)) + ' target_hr: ' + str(float(lateral_distance)) + ' target_sr: ' + str(float(distance)) + ' target_bearing: ' + str(float(bearing)) + '\n'
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
