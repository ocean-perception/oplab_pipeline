# parse_phins

# Scripts to parse ixsea blue phins data

# Author: Blair Thornton
# Date: 31/08/2017

from datetime import datetime
import codecs, time, json, glob
#http://www.json.org/
data_list=[]
#need to make acfr parsers
class parse_phins:
	def __init__(self, filepath, filename, category, timezone, timeoffset, ftype, outpath, fileoutname, fileout):

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
		header_depth = 'DEPIN_'
		header_altitude = 'LOGDVL'	


		# read in date from filename
		yyyy = int(filename[0:4])
		mm = int(filename[5:6])
		dd = int(filename[7:8])

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

		# parse phins data
		print('Parsing phins standard data')
		with codecs.open(filepath + filename,'r',encoding='utf-8', errors='ignore') as filein:
			flag_got_time = 0
			data=''			
			for line in filein.readlines():				
				line_split = line.strip().split('*')
				line_split_no_checksum = line_split[0].strip().split(',')
				
				# print(line_split_no_checksum)
				# what are the uncertainties
				

				if line_split_no_checksum[0] == header_start or line_split_no_checksum[0] == header_heading:
					# get time stamp
					if line_split_no_checksum[1]  == header_time:						
						time_string=str(line_split_no_checksum[2])
						hour=int(time_string[0:2])
						mins=int(time_string[2:4])
						secs=int(time_string[4:6])
						msec=int(time_string[7:10])						

						dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
						time_tuple = dt_obj.timetuple()
						epoch_time = time.mktime(time_tuple)
						epoch_timestamp = epoch_time+msec/1000+timeoffset						
						flag_got_time = 1										

					else:
						# get other data only if have a time stamp						
						if flag_got_time >= 1:
							# routine for oplab data format
							if ftype == 'oplab':
								if category == 'velocity':
									frame_string = 'body'
									if line_split_no_checksum[1]  == header_dvl:
										xx_velocity=float(line_split_no_checksum[2])
										yy_velocity=float(line_split_no_checksum[3])
										zz_velocity=float(line_split_no_checksum[4])

										xx_velocity_std=abs(xx_velocity)*velocity_std_factor+velocity_std_offset
										yy_velocity_std=abs(yy_velocity)*velocity_std_factor+velocity_std_offset
										zz_velocity_std=abs(zz_velocity)*velocity_std_factor+velocity_std_offset

										velocity_time=str(line_split_no_checksum[6])
										hour_dvl=int(velocity_time[0:2])
										mins_dvl=int(velocity_time[2:4])
										secs_dvl=int(velocity_time[4:6])
										msec_dvl=int(velocity_time[7:10])						
									
										dt_obj_dvl = datetime(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl)
										time_tuple_dvl = dt_obj_dvl.timetuple()
										epoch_time_dvl = time.mktime(time_tuple_dvl)
										epoch_timestamp_dvl = epoch_time_dvl+msec_dvl/1000+timeoffset

										#reset flag for next data
										flag_got_time = 0

										# write out in the required format interlace at end										
										data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_dvl': float(epoch_timestamp_dvl),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'xx_velocity':float(xx_velocity),'xx_velocity_std':float(xx_velocity_std)},{'yy_velocity':float(yy_velocity),'yy_velocity_std':float(yy_velocity_std)},{'zz_velocity':float(zz_velocity),'zz_velocity_std':float(zz_velocity_std)}]}
										data_list.append(data)

								if category == 'orientation':
									frame_string = 'inertial'
									if line_split_no_checksum[0] == header_heading:
										heading=float(line_split_no_checksum[1])
										flag_got_time = 2

									if line_split_no_checksum[1]  == header_attitude and flag_got_time == 2:
										roll=float(line_split_no_checksum[2])
										pitch=float(line_split_no_checksum[3])									
										flag_got_time = 3

									if line_split_no_checksum[1]  == header_attitude_std and flag_got_time == 3:
										heading_std=float(line_split_no_checksum[2])
										roll_std=float(line_split_no_checksum[3])
										pitch_std=float(line_split_no_checksum[4])
										
										#reset flag for next data
										flag_got_time = 0

										# write out in the required format interlace at end
										data = {'epoch_timestamp': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'heading':float(heading),'heading_std':float(heading_std)},{'roll':float(roll),'roll_std':float(roll_std)},{'pitch':float(pitch),'pitch_std':float(pitch_std)}]}
										data_list.append(data)
										

								if category == 'depth':
									frame_string = 'inertial'
									if line_split_no_checksum[1]  == header_depth:
										depth=float(line_split_no_checksum[2])

										depth_std=depth*depth_std_factor

										depth_time=str(line_split_no_checksum[3])	
										hour_depth=int(depth_time[0:2])
										mins_depth=int(depth_time[2:4])
										secs_depth=int(depth_time[4:6])
										msec_depth=int(depth_time[7:10])						
									
										dt_obj_depth = datetime(yyyy,mm,dd,hour_depth,mins_depth,secs_depth)
										time_tuple_depth = dt_obj_depth.timetuple()
										epoch_time_depth = time.mktime(time_tuple_depth)
										epoch_timestamp_depth = epoch_time_depth+msec_depth/1000+timeoffset

										#reset flag for next data
										flag_got_time = 0

										# write out in the required format interlace at end
										data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_depth': float(epoch_timestamp_depth),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'depth':float(depth),'depth_std':float(depth_std)}]}
										data_list.append(data)

								if category == 'altitude':
									frame_string = 'body'
									if line_split_no_checksum[1]  == header_dvl:									
										velocity_time=str(line_split_no_checksum[6])
										hour_dvl=int(velocity_time[0:2])
										mins_dvl=int(velocity_time[2:4])
										secs_dvl=int(velocity_time[4:6])
										msec_dvl=int(velocity_time[7:10])						
									
										dt_obj_dvl = datetime(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl)
										time_tuple_dvl = dt_obj_dvl.timetuple()
										epoch_time_dvl = time.mktime(time_tuple_dvl)
										epoch_timestamp_dvl = epoch_time_dvl+msec_dvl/1000+timeoffset

										flag_got_time = 2
										
									if line_split_no_checksum[1]  == header_altitude and flag_got_time == 2:
										sound_velocity=float(line_split_no_checksum[2])
										sound_velocity_correction=float(line_split_no_checksum[3])
										altitude=float(line_split_no_checksum[4])

										altitude_std=altitude*altitude_std_factor									
										
										#reset flag for next data
										flag_got_time = 0

										# write out in the required format interlace at end
										data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_dvl': float(epoch_timestamp_dvl),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'altitude':float(altitude),'altitude_std':float(altitude_std)},{'sound_velocity':float(sound_velocity),'sound_velocity_correction':float(sound_velocity_correction)}]}
										data_list.append(data)																	

							if ftype == 'acfr':
								if category == 'velocity':
																		
									if line_split_no_checksum[1]  == header_dvl and flag_got_time == 1: 
										xx_velocity=float(line_split_no_checksum[2])
										yy_velocity=float(line_split_no_checksum[3])
										zz_velocity=float(line_split_no_checksum[4])

										velocity_time=str(line_split_no_checksum[6])
										hour_dvl=int(velocity_time[0:2])
										mins_dvl=int(velocity_time[2:4])
										secs_dvl=int(velocity_time[4:6])
										msec_dvl=int(velocity_time[7:10])						
									
										dt_obj_dvl = datetime(yyyy,mm,dd,hour_dvl,mins_dvl,secs_dvl)
										time_tuple_dvl = dt_obj_dvl.timetuple()
										epoch_time_dvl = time.mktime(time_tuple_dvl)
										epoch_timestamp_dvl = epoch_time_dvl+msec_dvl/1000+timeoffset
										flag_got_time = 2

									if line_split_no_checksum[1]  == header_altitude and flag_got_time == 2:
										sound_velocity=float(line_split_no_checksum[2])
										sound_velocity_correction=float(line_split_no_checksum[3])
										altitude=float(line_split_no_checksum[4])

										flag_got_time = 3
									
									if line_split_no_checksum[0] == header_heading and flag_got_time == 3:
										heading=float(line_split_no_checksum[1])									
										flag_got_time = 4

									if line_split_no_checksum[1]  == header_attitude and flag_got_time == 4:
										roll=float(line_split_no_checksum[2])
										pitch=float(line_split_no_checksum[3])									

										#reset flag for next data
										flag_got_time = 0

										# write out in the required format interlace at end										
										data = 'RDI: ' + str(float(epoch_timestamp_dvl)) + ' alt:' + str(float(altitude)) + ' r1:0 r2:0 r3:0 r4:0 h:' + str(float(heading)) + ' p:' + str(float(pitch)) + ' r:' + str(float(roll)) + ' vx:' + str(float(xx_velocity)) + ' vy:' + str(float(yy_velocity)) + ' vz:' + str(float(zz_velocity)) + ' nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:32768 h_true:0 p_gimbal:0 sv: ' + str(float(sound_velocity)) + '\n'
										fileout.write(data)

								if category == 'orientation':
									if line_split_no_checksum[0] == header_heading:
										heading=float(line_split_no_checksum[1])
										flag_got_time = 2

									if line_split_no_checksum[1]  == header_attitude and flag_got_time ==2:
										roll=float(line_split_no_checksum[2])
										pitch=float(line_split_no_checksum[3])							
										flag_got_time = 3		
									
									if line_split_no_checksum[1]  == header_attitude_std and flag_got_time == 3:
										heading_std=float(line_split_no_checksum[2])
										roll_std=float(line_split_no_checksum[3])
										pitch_std=float(line_split_no_checksum[4])

										#reset flag for next data
										flag_got_time = 0

										# write out in the required format interlace at end
										data = 'PHINS_COMPASS: ' + str(float(epoch_timestamp)) + ' r: ' + str(float(roll)) + ' p: ' + str(float(pitch)) + ' h: ' + str(float(heading)) + ' std_r: ' + str(float(roll_std)) + ' std_p: ' + str(float(pitch_std)) + ' std_h: ' + str(float(heading_std)) + '\n'
										fileout.write(data)

								if category == 'depth':		
									if line_split_no_checksum[1]  == header_depth:
										depth=float(line_split_no_checksum[2])

										depth_std=depth*depth_std_factor

										depth_time=str(line_split_no_checksum[3])
										hour_depth=int(depth_time[0:2])
										mins_depth=int(depth_time[2:4])
										secs_depth=int(depth_time[4:6])
										msec_depth=int(depth_time[7:10])
									
										dt_obj_depth = datetime(yyyy,mm,dd,hour_depth,mins_depth,secs_depth)
										time_tuple_depth = dt_obj_depth.timetuple()
										epoch_time_depth = time.mktime(time_tuple_depth)
										epoch_timestamp_depth = epoch_time_depth+msec_depth/1000+timeoffset

										#reset flag for next data
										flag_got_time = 0

										# write out in the required format interlace at end
										data = 'PAROSCI: ' + str(float(epoch_timestamp_depth)) + ' ' + str(float(depth)) + '\n'
										fileout.write(data)
						else:
							continue
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
