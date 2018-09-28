# parse_phins

# Scripts to parse ixsea blue phins data

# Author: Blair Thornton
# Date: 31/08/2017

# from datetime import datetime
import hashlib, sys, os
import codecs, time, json, glob


# sys.path.append("..")
from lib_converttime.converttime import date_time_to_epoch
from lib_coordinates.body_to_inertial import body_to_inertial
#http://www.json.org/
#need to make acfr parsers
#needs tidying up!
class parse_phins:
	def __init__(self, filepath, filename, category, timezone, timeoffset, headingoffset, ftype, outpath, fileoutname):
		return
	def __new__(self, filepath, filename, category, timezone, timeoffset, headingoffset, ftype, outpath, fileoutname):
		# parser meta data
		class_string = 'measurement'
		sensor_string = 'phins'

		#phins std models
		depth_std_factor=0.01/100 #from catalogue paroscientific		
		velocity_std_factor=0.001 #from catalogue rdi whn1200/600
		velocity_std_offset=0.2 #from catalogue rdi whn1200/600
		altitude_std_factor=1/100 #acoustic velocity assumed 1% accurate

		# define headers used in phins
		header_start = '$PIXSE'
		header_time = 'TIME__'		
		header_heading = '$HEHDT'
		header_attitude = 'ATITUD'
		header_attitude_std = 'STDHRP'

		
		header_dvl = 'LOGIN_'		
		header_vel = 'SPEED_'
		header_vel_std = 'STDSPD'

		header_depth = 'DEPIN_'
		header_altitude = 'LOGDVL'	


		# read in date from filename
		yyyy = int(filename[0:4])
		mm = int(filename[4:6])
		dd = int(filename[6:8])

		
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
		# timeoffset = -timezone_offset*60*60 + timeoffset 

		# parse phins data
		print('...... parsing phins standard data')
		data_list=[]
		if ftype == 'acfr':
			data_list = ''
		with codecs.open(filepath + filename,'r',encoding='utf-8', errors='ignore') as filein:
			flag_got_time = 0
			data=''
			for line in filein.readlines():				
				line_split = line.strip().split('*')
				line_split_no_checksum = line_split[0].strip().split(',')
				
				if len(line_split) == 2 and (line_split_no_checksum[0] == header_start or line_split_no_checksum[0] == header_heading):
					# get time stamp
					# do a check sum as a lot of broken packets are found in phins data
					check_sum= str(line_split[1])
					
					# extract from $ to * as per phins manual
					string_to_check=','.join(line_split_no_checksum)					
					string_to_check=string_to_check[1:len(string_to_check)]
					
					broken_packet_flag=0 # reset broken packet flag
					string_sum =0
					
					for i in range(len(string_to_check)):
						string_sum ^= ord(string_to_check[i])

					if str(hex(string_sum)[2:].zfill(2).upper()) == check_sum.upper():
					
						if line_split_no_checksum[1]  == header_time:						
							time_string=str(line_split_no_checksum[2])
							hour=int(time_string[0:2])
							mins=int(time_string[2:4])
							try:
								secs=int(time_string[4:6])

								# phins sometimes returns 60s...
								if secs >= 60:
									broken_packet_flag = 1

							except ValueError:
								broken_packet_flag = 1

							try:
								msec=int(time_string[7:10])						

							except ValueError:																				
								broken_packet_flag = 1

							if broken_packet_flag == 0:		
								epoch_time = date_time_to_epoch(yyyy,mm,dd,hour,mins,secs,timezone_offset)																				
								# dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
								
								# time_tuple = dt_obj.timetuple()
								# epoch_time = time.mktime(time_tuple)
								epoch_timestamp = epoch_time+msec/1000+timeoffset						
								flag_got_time = 1

							else:
								print('Warning: Badly formatted packet (PHINS TIME)')
								print(line)


						else:
							# get other data only if have a time stamp						
							if flag_got_time >= 1:
								# routine for oplab data format
								if ftype == 'oplab':
									if category == 'velocity':

										#print(flag_got_time,line_split_no_checksum[1])

										if line_split_no_checksum[1]  == header_dvl: # and flag_got_time == 3:
											
											frame_string = 'body'
												
											x_velocity=float(line_split_no_checksum[2]) # DVL convention is +ve aft to forward
											#y_velocity=float(line_split_no_checksum[3]) # DVL convention is +ve port to starboard so the minus shouldn't be necessary?
											y_velocity=float(line_split_no_checksum[3]) # DVL convention is +ve port to starboard
											z_velocity=float(line_split_no_checksum[4]) # DVL convention is bottom to top +ve

											# account for sensor rotational offset
											# print('IN :',x_velocity, y_velocity, z_velocity)
											[x_velocity,y_velocity,z_velocity] = body_to_inertial(0, 0, headingoffset, x_velocity, y_velocity, z_velocity)
											# print('OUT:',x_velocity, y_velocity, z_velocity)
											y_velocity=-1*y_velocity
											z_velocity=-1*z_velocity

											x_velocity_std=abs(x_velocity)*velocity_std_factor+velocity_std_offset
											y_velocity_std=abs(y_velocity)*velocity_std_factor+velocity_std_offset
											z_velocity_std=abs(z_velocity)*velocity_std_factor+velocity_std_offset

											velocity_time=str(line_split_no_checksum[6])

											hour_dvl=int(velocity_time[0:2])
											mins_dvl=int(velocity_time[2:4])
											try:
												secs_dvl=int(velocity_time[4:6])
												if secs_dvl >= 60:
													broken_packet_flag = 1

											except ValueError:											
												broken_packet_flag = 1

											try:
												msec_dvl=int(velocity_time[7:10])
												
											except ValueError:											
												broken_packet_flag = 1
												
											if broken_packet_flag == 0:
												epoch_time_dvl = date_time_to_epoch(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl,timezone_offset)
												# dt_obj_dvl = datetime(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl)
												# time_tuple_dvl = dt_obj_dvl.timetuple()
												# epoch_time_dvl = time.mktime(time_tuple_dvl)
												epoch_timestamp_dvl = epoch_time_dvl+msec_dvl/1000+timeoffset


												# write out in the required format interlace at end										
												data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_dvl': float(epoch_timestamp_dvl),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'x_velocity':float(x_velocity),'x_velocity_std':float(x_velocity_std)},{'y_velocity':float(y_velocity),'y_velocity_std':float(y_velocity_std)},{'z_velocity':float(z_velocity),'z_velocity_std':float(z_velocity_std)}]}												
												data_list.append(data)
												
											else:
												print('Warning: Badly formatted packet (DVL TIME)')
												print(line)

											#set flag for next data
											flag_got_time = flag_got_time + 1

										if line_split_no_checksum[1]  == header_vel:		

											frame_string = 'inertial'
																							
											#east_velocity=-1*float(line_split_no_checksum[2]) # phins convention is west +ve so a minus should be necessary
											east_velocity=float(line_split_no_checksum[2]) # phins convention is west +ve so a minus should be necessary
											north_velocity=float(line_split_no_checksum[3])									
											down_velocity=-1*float(line_split_no_checksum[4]) # phins convention is up +ve

											#set flag for next data
											flag_got_time = flag_got_time + 1

										if line_split_no_checksum[1]  == header_vel_std:

											east_velocity_std=float(line_split_no_checksum[2]) # phins convention is west +ve
											north_velocity_std=float(line_split_no_checksum[3])											
											down_velocity_std=-1*float(line_split_no_checksum[4]) # phins convention is up +ve												

											flag_got_time = flag_got_time + 1

										if flag_got_time == 4:
												# write out in the required format interlace at end										
											data = {'epoch_timestamp': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'north_velocity':float(north_velocity),'north_velocity_std':float(north_velocity_std)},{'east_velocity':float(east_velocity),'east_velocity_std':float(east_velocity_std)},{'down_velocity':float(down_velocity),'down_velocity_std':float(down_velocity_std)}]}
											data_list.append(data)											

											#reset flag for next data
											flag_got_time = 0

									if category == 'orientation':
										frame_string = 'body'
										if line_split_no_checksum[0] == header_heading:
											heading=float(line_split_no_checksum[1]) # phins +ve clockwise so no need to change
											flag_got_time = flag_got_time + 1 #2

										if line_split_no_checksum[1]  == header_attitude:
											roll=-1*float(line_split_no_checksum[2])
											pitch=-1*float(line_split_no_checksum[3]) # phins +ve nose up so no need to change								
											flag_got_time = flag_got_time + 1

										if line_split_no_checksum[1]  == header_attitude_std:
											heading_std=float(line_split_no_checksum[2])
											roll_std=float(line_split_no_checksum[3])
											pitch_std=float(line_split_no_checksum[4])
											flag_got_time = flag_got_time + 1

										if flag_got_time == 4:
											
											# account for sensor rotational offset										
											[roll, pitch, heading] = body_to_inertial(0, 0, headingoffset, roll, pitch, heading)
											[roll_std, pitch_std, heading_std] = body_to_inertial(0, 0, headingoffset, roll_std, pitch_std, heading_std)

											#heading=heading+headingoffset
											if heading >360:
												heading=heading-360
											if heading < 0:
												heading=heading+360
											
											# write out in the required format interlace at end											
											data = {'epoch_timestamp': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'heading':float(heading),'heading_std':float(heading_std)},{'roll':float(roll),'roll_std':float(roll_std)},{'pitch':float(pitch),'pitch_std':float(pitch_std)}]}											
											data_list.append(data)

											#reset flag for next data
											flag_got_time = 0

											

									if category == 'depth':
										frame_string = 'inertial'
										if line_split_no_checksum[1]  == header_depth:
											try:
												depth=float(line_split_no_checksum[2])

												depth_std=depth*depth_std_factor

												depth_time=str(line_split_no_checksum[3])	
												hour_depth=int(depth_time[0:2])
												mins_depth=int(depth_time[2:4])
												secs_depth=int(depth_time[4:6])
												msec_depth=int(depth_time[7:10])

												epoch_time_depth = date_time_to_epoch(yyyy,mm,dd,hour_depth,mins_depth,secs_depth,timezone_offset)
												# dt_obj_depth = datetime(yyyy,mm,dd,hour_depth,mins_depth,secs_depth)
												# time_tuple_depth = dt_obj_depth.timetuple()
												# epoch_time_depth = time.mktime(time_tuple_depth)
												epoch_timestamp_depth = epoch_time_depth+msec_depth/1000+timeoffset

												# write out in the required format interlace at end
												data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_depth': float(epoch_timestamp_depth),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'depth':float(depth),'depth_std':float(depth_std)}]}
												data_list.append(data)

												#reset flag for next data
												flag_got_time = 0
											except ValueError:
												pass

									if category == 'altitude':
										frame_string = 'body'										
										if line_split_no_checksum[1]  == header_dvl:									
											velocity_time=str(line_split_no_checksum[6])
											hour_dvl=int(velocity_time[0:2])
											mins_dvl=int(velocity_time[2:4])
											try:
												secs_dvl=int(velocity_time[4:6])
												if secs_dvl >= 60:
													broken_packet_flag = 1

											except ValueError:											
												broken_packet_flag = 1

											try:
												msec_dvl=int(velocity_time[7:10])
											
											except ValueError:											
												broken_packet_flag = 1
											
											if broken_packet_flag == 0:						
												
												epoch_time_dvl = date_time_to_epoch(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl,timezone_offset)
												# dt_obj_dvl = datetime(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl)
												# time_tuple_dvl = dt_obj_dvl.timetuple()
												# epoch_time_dvl = time.mktime(time_tuple_dvl)
												epoch_timestamp_dvl = epoch_time_dvl+msec_dvl/1000+timeoffset

												flag_got_time = flag_got_time + 1
											else:
												print('Warning: Badly formatted packet (DVL TIME)(DVL ALTITUDE)')
												print(line)
												flag_got_time = flag_got_time - 1 # reinitiate packet
											
										if line_split_no_checksum[1]  == header_altitude:
											sound_velocity=float(line_split_no_checksum[2])
											sound_velocity_correction=float(line_split_no_checksum[3])
											altitude=float(line_split_no_checksum[4])

											altitude_std=altitude*altitude_std_factor	
											flag_got_time = flag_got_time + 1								
											
										if flag_got_time == 3:
											#reset flag for next data
											flag_got_time = 0

											# write out in the required format interlace at end
											data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_dvl': float(epoch_timestamp_dvl),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'altitude':float(altitude),'altitude_std':float(altitude_std)},{'sound_velocity':float(sound_velocity),'sound_velocity_correction':float(sound_velocity_correction)}]}											
											data_list.append(data)																	

								if ftype == 'acfr':
									if category == 'velocity':
										# order of latitude and dvl swap for different files
										if line_split_no_checksum[1]  == header_altitude :
											
											sound_velocity=float(line_split_no_checksum[2])
											sound_velocity_correction=float(line_split_no_checksum[3])
											altitude=float(line_split_no_checksum[4])										
											flag_got_time = flag_got_time + 1
										
										if line_split_no_checksum[1]  == header_dvl: 
											
											xx_velocity=float(line_split_no_checksum[2])
											yy_velocity=float(line_split_no_checksum[3])
											#yy_velocity=-1*float(line_split_no_checksum[3]) # according to the manual, the minus shouldn't be needed
											zz_velocity=float(line_split_no_checksum[4]) # DVL convention is bottom to top +ve

											# account for sensor offset
											[xx_velocity,yy_velocity,zz_velocity] = body_to_inertial(0, 0, headingoffset, xx_velocity, yy_velocity, zz_velocity)
											yy_velocity=-1*yy_velocity
											zz_velocity=-1*zz_velocity

											velocity_time=str(line_split_no_checksum[6])
											hour_dvl=int(velocity_time[0:2])
											mins_dvl=int(velocity_time[2:4])
											
											try:
												secs_dvl=int(velocity_time[4:6])
												if secs_dvl >= 60:
													broken_packet_flag = 1

											except ValueError:											
												broken_packet_flag = 1

											
											try:
												msec_dvl=int(velocity_time[7:10])

											except ValueError:											
												broken_packet_flag = 1
											
											if broken_packet_flag == 0:
												
												epoch_time_dvl = date_time_to_epoch(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl,timezone_offset)
												# dt_obj_dvl = datetime(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl)
												# time_tuple_dvl = dt_obj_dvl.timetuple()
												# epoch_time_dvl = time.mktime(time_tuple_dvl)
												epoch_timestamp_dvl = epoch_time_dvl+msec_dvl/1000+timeoffset
												flag_got_time = flag_got_time+1																								

											else:
												print('Warning: Badly formatted packet (DVL TIME)')												
												flag_got_time = flag_got_time-1

										
										# use measurements of velocity from PHINS																			
										
										
										if line_split_no_checksum[0] == header_heading:
											heading=float(line_split_no_checksum[1])									
											flag_got_time = flag_got_time + 1

										if line_split_no_checksum[1]  == header_attitude:
											roll=-1*float(line_split_no_checksum[2])
											pitch=-1*float(line_split_no_checksum[3])	
											flag_got_time = flag_got_time + 1								

											#reset flag for next data
										if flag_got_time == 5:

											flag_got_time = 0

											#print(data)
											try:
											# write out in the required format interlace at end																				
												data = 'RDI: ' + str(float(epoch_timestamp_dvl)) + ' alt:' + str(float(altitude)) + ' r1:0 r2:0 r3:0 r4:0 h:' + str(float(heading)) + ' p:' + str(float(pitch)) + ' r:' + str(float(roll)) + ' vx:' + str(float(xx_velocity)) + ' vy:' + str(float(yy_velocity)) + ' vz:' + str(float(zz_velocity)) + ' nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:0 h_true:0 p_gimbal:0 sv: ' + str(float(sound_velocity)) + '\n'
												data_list += data
											except UnboundLocalError:
												pass

									if category == 'orientation':
										if line_split_no_checksum[0] == header_heading:
											heading=float(line_split_no_checksum[1])
											flag_got_time = flag_got_time +1

										if line_split_no_checksum[1]  == header_attitude:
											roll=-1*float(line_split_no_checksum[2])
											pitch=-1*float(line_split_no_checksum[3])							
											flag_got_time = flag_got_time +1
										

										if line_split_no_checksum[1]  == header_attitude_std:
											heading_std=float(line_split_no_checksum[2])
											roll_std=float(line_split_no_checksum[3])
											pitch_std=float(line_split_no_checksum[4])
											flag_got_time = flag_got_time +1

										if flag_got_time == 4:
											#reset flag for next data
											flag_got_time = 0

											# account for sensor rotational offset
											[roll, pitch, heading] = body_to_inertial(0, 0, headingoffset, roll, pitch, heading)
											[roll_std, pitch_std, heading_std] = body_to_inertial(0, 0, headingoffset, roll_std, pitch_std, heading_std)
											
											#heading=heading+headingoffset
											if heading >360:
												heading=heading-360
											if heading < 0:
												heading=heading+360
											
											#print(data)												
											# write out in the required format interlace at end
											data = 'PHINS_COMPASS: ' + str(float(epoch_timestamp)) + ' r: ' + str(float(roll)) + ' p: ' + str(float(pitch)) + ' h: ' + str(float(heading)) + ' std_r: ' + str(float(roll_std)) + ' std_p: ' + str(float(pitch_std)) + ' std_h: ' + str(float(heading_std)) + '\n'
											data_list += data

									if category == 'depth':		
										if line_split_no_checksum[1]  == header_depth:
											try:
												depth=float(line_split_no_checksum[2])

												depth_std=depth*depth_std_factor

												depth_time=str(line_split_no_checksum[3])
												hour_depth=int(depth_time[0:2])
												mins_depth=int(depth_time[2:4])
												secs_depth=int(depth_time[4:6])
												msec_depth=int(depth_time[7:10])
												
												epoch_time_depth = date_time_to_epoch(yyyy,mm,dd,hour_depth,mins_depth,secs_depth,timezone_offset)
												# dt_obj_depth = datetime(yyyy,mm,dd,hour_depth,mins_depth,secs_depth)
												# time_tuple_depth = dt_obj_depth.timetuple()
												# epoch_time_depth = time.mktime(time_tuple_depth)
												epoch_timestamp_depth = epoch_time_depth+msec_depth/1000+timeoffset

												# write out in the required format interlace at end
												data = 'PAROSCI: ' + str(float(epoch_timestamp_depth)) + ' ' + str(float(depth)) + '\n'
												data_list += data

												#reset flag for next data
												flag_got_time = 0
											except ValueError:
												pass
							else:
								continue
					else:
						print('Broken packet: ', line)
						print('Check sum calculated ', hex(string_sum).zfill(2).upper())
						print('Does not match that provided', check_sum.upper())
						print('Ignore and move on')

		return data_list