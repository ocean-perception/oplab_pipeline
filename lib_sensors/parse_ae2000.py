# parse_ae2000

# Scripts to parse ae2000 logs

# Author: Blair Thornton
# Date: 14/02/2018

from datetime import datetime
import hashlib, sys, os
import codecs, time, json, glob
import pandas as pd

sys.path.append("..")
from lib_coordinates.body_to_inertial import body_to_inertial
#http://www.json.org/
data_list=[]
#need to make acfr parsers
class parse_ae2000:
	def __init__(self, filepath, filename, category, timezone, timeoffset, headingoffset, ftype, outpath, fileoutname, fileout):

		# parser meta data
		class_string = 'measurement'
		sensor_string = 'ae20000'

		#phins std models
		depth_std_factor=0.01/100 #from catalogue paroscientific		
		velocity_std_factor=0.001 #from catalogue rdi whn1200/600
		velocity_std_offset=0.2 #from catalogue rdi whn1200/600
		altitude_std_factor=1/100 #acoustic velocity assumed 1% accurate
		altitude_limit=200# value given by ae2000 when there is no bottom lock
		# read in date from filename
		

		yyyy = int(filename[3:5])+2000
		mm = int(filename[5:7])
		dd = int(filename[7:9])

		
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
		print('  Parsing ae2000 logs...')
		data_list=[]
		df = pd.read_csv(filepath + filename)
		
		time_column = df.iloc[:,0] # list of time value in the first column (starting from 2nd row, not considering first row)
		for row_index in range(len(time_column)): # length of this should match every other column
			timestamp=time_column[row_index].split(':')
			
			hour=int(timestamp[0])
			mins=int(timestamp[1])
			timestamp_s_ms=timestamp[2].split('.')
			secs=int(timestamp_s_ms[0])
			msec=int(timestamp_s_ms[1])

			dt_obj = datetime(yyyy,mm,dd,hour,mins,secs)
								
			time_tuple = dt_obj.timetuple()
			epoch_time = time.mktime(time_tuple)
			epoch_timestamp = epoch_time+msec/1000+timeoffset

			if float(df['Height'][row_index])<altitude_limit:
				if ftype == 'oplab':
					if category == 'velocity':

						frame_string = 'body'
											
						x_velocity=float(df['vx'][row_index]) # DVL convention is +ve aft to forward
						y_velocity=float(df['vy'][row_index]) # DVL convention is +ve port to starboard
						z_velocity=float(df['vz'][row_index]) # DVL convention is bottom to top +ve

						# account for sensor rotational offset
						[x_velocity,y_velocity,z_velocity] = body_to_inertial(0, 0, headingoffset, x_velocity, y_velocity, z_velocity)
						# print('OUT:',x_velocity, y_velocity, z_velocity)
						# y_velocity=-1*y_velocity
						# z_velocity=-1*z_velocity

						x_velocity_std=abs(x_velocity)*velocity_std_factor+velocity_std_offset
						y_velocity_std=abs(y_velocity)*velocity_std_factor+velocity_std_offset
						z_velocity_std=abs(z_velocity)*velocity_std_factor+velocity_std_offset
						
						# write out in the required format interlace at end										
						data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_dvl': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'x_velocity':float(x_velocity),'x_velocity_std':float(x_velocity_std)},{'y_velocity':float(y_velocity),'y_velocity_std':float(y_velocity_std)},{'z_velocity':float(z_velocity),'z_velocity_std':float(z_velocity_std)}]}												
						data_list.append(data)

						frame_string = 'inertial'
						east_velocity=float(df['vye'][row_index]) # phins convention is west +ve so a minus should be necessary
						north_velocity=float(df['vxe'][row_index])
						down_velocity=-9999

						east_velocity_std=abs(east_velocity)*velocity_std_factor+velocity_std_offset
						north_velocity_std=abs(north_velocity)*velocity_std_factor+velocity_std_offset
						down_velocity_std=-9999
											# write out in the required format interlace at end										
						data = {'epoch_timestamp': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'north_velocity':float(north_velocity),'north_velocity_std':float(north_velocity_std)},{'east_velocity':float(east_velocity),'east_velocity_std':float(east_velocity_std)},{'down_velocity':float(down_velocity),'down_velocity_std':float(down_velocity_std)}]}
						data_list.append(data)											

					if category == 'orientation':
						frame_string = 'body'
							
						roll=float(df['Roll'][row_index])
						pitch=float(df['Pitch'][row_index])
						heading=float(df['Yaw'][row_index])

						heading_std=-9999
						roll_std=-9999
						pitch_std=-9999
						# account for sensor rotational offset										
						[roll, pitch, heading] = body_to_inertial(0, 0, headingoffset, roll, pitch, heading)
						
						#heading=heading+headingoffset
						if heading >360:
							heading=heading-360
						if heading < 0:
							heading=heading+360
										
							# write out in the required format interlace at end											
						data = {'epoch_timestamp': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'heading':float(heading),'heading_std':float(heading_std)},{'roll':float(roll),'roll_std':float(roll_std)},{'pitch':float(pitch),'pitch_std':float(pitch_std)}]}											
						data_list.append(data)

						frame_string = 'body'
						sub_category = 'angular_rate'

						roll_rate=float(df['rrate'][row_index])
						pitch_rate=float(df['prate'][row_index])
						heading_rate=float(df['yrate'][row_index])

						heading_rate_std=-9999
						roll_rate_std=-9999
						pitch_rate_std=-9999

						data = {'epoch_timestamp': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': sub_category,'data': [{'heading_rate':float(heading_rate),'heading_rate_std':float(heading_rate_std)},{'roll_rate':float(roll_rate),'roll_rate_std':float(roll_rate_std)},{'pitch_rate':float(pitch_rate),'pitch_rate_std':float(pitch_rate_std)}]}									
						data_list.append(data)

										
					if category == 'depth':
						frame_string = 'inertial'
							
						depth=float(df['Depth'][row_index])
						depth_std=depth*depth_std_factor

										# write out in the required format interlace at end
						data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_depth': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'depth':float(depth),'depth_std':float(depth_std)}]}
						data_list.append(data)

					if category == 'altitude':
						frame_string = 'body'																		
						altitude=float(df['Height'][row_index])
						altitude_std=altitude*altitude_std_factor	
						sound_velocity=-9999
						sound_velocity_correction=-9999

							# write out in the required format interlace at end
						data = {'epoch_timestamp': float(epoch_timestamp),'epoch_timestamp_dvl': float(epoch_timestamp),'class': class_string,'sensor':sensor_string,'frame':frame_string,'category': category,'data': [{'altitude':float(altitude),'altitude_std':float(altitude_std)},{'sound_velocity':float(sound_velocity),'sound_velocity_correction':float(sound_velocity_correction)}]}											
						data_list.append(data)																	

				if ftype == 'acfr':
					if category == 'velocity':
				
						
										
						sound_velocity=-9999
						sound_velocity_correction=-9999
						altitude=float(df['Height'][row_index])

						xx_velocity=float(df['vx'][row_index]) # DVL convention is +ve aft to forward
						yy_velocity=float(df['vy'][row_index]) # DVL convention is +ve port to starboard
						zz_velocity=float(df['vz'][row_index]) # DVL convention is bottom to top +ve

							
									# account for sensor offset
							# [xx_velocity,yy_velocity,zz_velocity] = body_to_inertial(0, 0, headingoffset, xx_velocity, yy_velocity, zz_velocity)
							# yy_velocity=-1*yy_velocity
							# zz_velocity=-1*zz_velocity

						roll=float(df['Roll'][row_index])
						pitch=float(df['Pitch'][row_index])
						heading=float(df['Yaw'][row_index])

									#print(data)
							# write out in the required format interlace at end																				
						data = 'RDI: ' + str(float(epoch_timestamp)) + ' alt:' + str(float(altitude)) + ' r1:0 r2:0 r3:0 r4:0 h:' + str(float(heading)) + ' p:' + str(float(pitch)) + ' r:' + str(float(roll)) + ' vx:' + str(float(xx_velocity)) + ' vy:' + str(float(yy_velocity)) + ' vz:' + str(float(zz_velocity)) + ' nx:0 ny:0 nz:0 COG:0 SOG:0 bt_status:0 h_true:0 p_gimbal:0 sv: ' + str(float(sound_velocity)) + '\n'
						fileout.write(data)

					if category == 'orientation':
						roll=float(df['Roll'][row_index])
						pitch=float(df['Pitch'][row_index])
						heading=float(df['Yaw'][row_index])

						roll_std=-9999
						pitch_std=-9999
						heading_std=-9999
							
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
						fileout.write(data)

					if category == 'depth':																				
							
						depth=float(df['Depth'][row_index])
						# write out in the required format interlace at end
						data = 'PAROSCI: ' + str(float(epoch_timestamp)) + ' ' + str(float(depth)) + '\n'
						fileout.write(data)
					else:
						continue
			# else:
			# 	print('no bottom lock')
		print('  ...done parsing ae2000 logs.')

		print('  Writing converted ae2000 data to file...')
		if ftype == 'oplab':
			fileout.close()
			for filein in glob.glob(outpath + os.sep + fileoutname):
				try:
					with open(filein, 'rb') as json_file:					
						data_in=json.load(json_file)						
						for i in range(len(data_in)):
							data_list.insert(0,data_in[len(data_in)-i-1])				        
						
				except ValueError:					
					print('An error occurred while initialising JSON file')

			with open(outpath + os.sep + fileoutname,'w') as fileout:
				json.dump(data_list, fileout)	
				del data_list
		print('  ...done writing converted ae2000 data to file.')