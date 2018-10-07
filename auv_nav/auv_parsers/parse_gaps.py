# parse_gaps

# Scripts to parse ixsea blue gaps data. Interpolates ship gps reading for valid underwater position measurements to determine accurate range and so measurement uncertainty

# Author: Blair Thornton
# Date: 31/08/2017

import os
# from datetime import datetime
import sys, math
import time, json, glob
#http://www.json.org/
# sys.path.append("..")
from auv_nav.auv_coordinates.latlon_wgs84 import latlon_to_metres
from auv_nav.auv_coordinates.latlon_wgs84 import metres_to_latlon
from auv_nav.auv_conversions.time_conversions import date_time_to_epoch

class parse_gaps:
	def __init__(self, filepath, category, timezone, timeoffset, latitude_reference, longitude_reference, ftype, outpath, fileoutname, usbl_id):
		return
	def __new__(self, filepath, category, timezone, timeoffset, latitude_reference, longitude_reference, ftype, outpath, fileoutname, usbl_id):

		# parser meta data    
		class_string = 'measurement'
		sensor_string = 'gaps'
		frame_string = 'inertial'

		# define headers used in phins
		header_absolute = '$PTSAG'# '<< $PTSAG' #georeferenced strings
		header_heading = '$HEHDT'# '<< $HEHDT'

		# gaps std models
		distance_std_factor = 1/100 # usbl catalogue gaps spec
		distance_std_offset = 2 # 2m lateral error on DGPS to be added
		broken_packet_flag = False

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
				print('Error: timezone', timezone, 'in mission.yaml not recognised, please enter value from UTC in hours')
				return

		# convert to seconds from utc
		# timeoffset = -timezone_offset*60*60 + timeoffset 

		# determine file paths
		all_list = os.listdir(filepath)
		gaps_list = [ line for line in all_list if '.dat' in line]
		print(str(len(gaps_list)) + ' GAPS file(s) found')
		
		# extract data from files
		data_list=[]
		if ftype == 'acfr':
			data_list = ''
		for i in range(len(gaps_list)):
			path_gaps = filepath + os.sep + gaps_list[i]
			
			with open(path_gaps, errors = 'ignore') as gaps:
				# initialise flag				
				flag_got_time = 0
				for line in gaps.readlines():
					line_split = line.strip().split('*') 
					line_split_no_checksum = line_split[0].strip().split(',')
					broken_packet_flag = False
					#print(line_split_no_checksum)
					# keep on upating ship position to find the prior interpolation value of ship coordinates
					if header_absolute in line_split_no_checksum[0]: # line_split_no_checksum[0] == header_absolute:
						# start with a ship coordinate
						if line_split_no_checksum[6] == str(usbl_id) and flag_got_time == 2:
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

								try:								    
									msec=int(time_string[7:10])
								except ValueError as verr:
									broken_packet_flag=True
									pass

								if secs >= 60:
									mins += 1
									secs = 0
									broken_packet_flag = True

								epoch_time = date_time_to_epoch(yyyy,mm,dd,hour,mins,secs,timezone_offset)

								# dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
								# time_tuple = dt_obj.timetuple()
								# epoch_time = time.mktime(time_tuple)
								epoch_timestamp = epoch_time+msec/1000+timeoffset
									 
									# get position
								latitude_string = line_split_no_checksum[7]
								latitude_degrees = int(latitude_string[0:2])
								latitude_minutes = float(latitude_string[2:10])
								if line_split_no_checksum[8] == 'S':
									latitude_negative_flag = True

								longitude_string = line_split_no_checksum[9]
								longitude_degrees = int(longitude_string[0:3])
								longitude_minutes = float(longitude_string[3:11])
								if line_split_no_checksum[10] == 'W':
									longitude_negative_flag = True

								depth = float(line_split_no_checksum[12])

								# flag raised to proceed
								flag_got_time = 3						
							else:
								flag_got_time = 0

						if line_split_no_checksum[6] == '0':
							if flag_got_time < 3:
							
							# read in date
							
								yyyy = int(line_split_no_checksum[5])
								mm = int(line_split_no_checksum[4])
								dd = int(line_split_no_checksum[3])

								 # print(yyyy,mm,dd)
								 # read in time
								time_string=str(line_split_no_checksum[2])
								# print(time_string)
								hour=int(time_string[0:2])
								mins=int(time_string[2:4])
								secs=int(time_string[4:6])
								
								try:								    
									msec=int(time_string[7:10])
								except ValueError as verr:
									broken_packet_flag=True
									pass

								if secs >= 60:
									mins += 1
									secs = 0
									broken_packet_flag = True

								epoch_time = date_time_to_epoch(yyyy,mm,dd,hour,mins,secs,timezone_offset)
								# dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
								# time_tuple = dt_obj.timetuple()
								# epoch_time = time.mktime(time_tuple)
								epoch_timestamp_ship_prior = epoch_time+msec/1000+timeoffset

								# get position
								latitude_string = line_split_no_checksum[7]
								latitude_degrees_ship_prior = int(latitude_string[0:2])
								latitude_minutes_ship_prior = float(latitude_string[2:10])
								latitude_prior=latitude_degrees_ship_prior+latitude_minutes_ship_prior/60.0		
								if line_split_no_checksum[8] == 'S':
									latitude_prior = -latitude_prior	

								longitude_string = line_split_no_checksum[9]
								longitude_degrees_ship_prior = int(longitude_string[0:3])
								longitude_minutes_ship_prior = float(longitude_string[3:11])
								longitude_prior=longitude_degrees_ship_prior+longitude_minutes_ship_prior/60.0
								if line_split_no_checksum[10] == 'W':
									longitude_prior = -longitude_prior

								# flag raised to proceed								
								if flag_got_time < 2:
								    flag_got_time = flag_got_time + 1

							elif flag_got_time >= 3:
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
									epoch_time = date_time_to_epoch(yyyy,mm,dd,hour,mins,secs,timezone_offset)
									# dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
									# time_tuple = dt_obj.timetuple()
									# epoch_time = time.mktime(time_tuple)
									epoch_timestamp_ship_posterior = epoch_time+msec/1000+timeoffset

									# get position
									latitude_string = line_split_no_checksum[7]
									latitude_degrees_ship_posterior = int(latitude_string[0:2])
									latitude_minutes_ship_posterior = float(latitude_string[2:10])
									latitude_posterior=latitude_degrees_ship_posterior+latitude_minutes_ship_posterior/60.0
									if line_split_no_checksum[8] == 'S':
										latitude_posterior = -latitude_posterior

									longitude_string = line_split_no_checksum[9]
									longitude_degrees_ship_posterior = int(longitude_string[0:3])
									longitude_minutes_ship_posterior = float(longitude_string[3:11])
									longitude_posterior=longitude_degrees_ship_posterior+longitude_minutes_ship_posterior/60.0
									if line_split_no_checksum[10] == 'W':
										longitude_posterior = -longitude_posterior

									# flag raised to proceed
									flag_got_time = flag_got_time + 1


					if header_heading in line_split_no_checksum[0]: # line_split_no_checksum[0] == header_heading: 
						if flag_got_time < 3:
							heading_ship_prior = float(line_split_no_checksum[1])
							if flag_got_time < 2:
								    flag_got_time = flag_got_time + 1												
										
						else:					

							heading_ship_posterior = float(line_split_no_checksum[1])
							flag_got_time = flag_got_time + 1
						
					if flag_got_time >= 5:
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
						try:
							if latitude_negative_flag:
								latitude = -latitude
						except NameError:
							pass
						longitude = longitude_degrees+longitude_minutes/60.0
						try:
							if longitude_negative_flag:
								longitude = -longitude
						except NameError:
							pass
						
						lateral_distance,bearing = latlon_to_metres(latitude, longitude, latitude_ship, longitude_ship)

						distance = math.sqrt(lateral_distance * lateral_distance + depth * depth)
						distance_std = distance_std_factor*distance + distance_std_offset

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

						if broken_packet_flag == False:																						

							if ftype == 'oplab':							
								data = {'epoch_timestamp': float(epoch_timestamp), 'class': class_string, 'sensor': sensor_string, 'frame': frame_string, 'category': category, 'data_ship': [{'latitude': float(latitude_ship), 'longitude': float(longitude_ship)}, {'northings': float(northings_ship), 'eastings': float(eastings_ship)}, {'heading': float(heading_ship)}], 'data_target': [{'latitude': float(latitude), 'latitude_std': float(latitude_std)}, {'longitude': float(longitude), 'longitude_std': float(longitude_std)}, {'northings': float(northings_target), 'northings_std': float(distance_std)}, {'eastings': float(eastings_target), 'eastings_std': float(distance_std)}, {'depth': float(depth), 'depth_std': float(distance_std)}, {'distance_to_ship': float(distance)}]}
								data_list.append(data)

							if ftype == 'acfr':
								data = 'SSBL_FIX: ' + str(float(epoch_timestamp)) + ' ship_x: ' + str(float(northings_ship)) + ' ship_y: ' + str(float(eastings_ship)) + ' target_x: ' + str(float(northings_target)) + ' target_y: ' + str(float(eastings_target)) + ' target_z: ' + str(float(depth)) + ' target_hr: ' + str(float(lateral_distance)) + ' target_sr: ' + str(float(distance)) + ' target_bearing: ' + str(float(bearing)) + '\n'
								data_list += data

						else:
	 					    print('Warning: Badly formatted packet (GAPS TIME)')
	 					    print(line)
	 					    print('Moving on')
								#print(hour,mins,secs)

						# reset flag						
						flag_got_time = 0


		return data_list