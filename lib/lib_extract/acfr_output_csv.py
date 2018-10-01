import pandas as pd
import os, sys

def acfr_output_csv(filepath):
	# takes in acfr outputs 'stereo_pose_est.data' and 'vehicle_pose_est.data' and turn their values into csv files for QGis
	# filepath = path to folder that contains 'stereo_pose_est.data' and 'vehicle_pose_est.data'

	#'D:/OneDrive - University of Southampton/PhD/2_SLAM/auv_nav/master/auv_nav_results/processed/2015/NT15-E03/20150918/acfr_position_format/gas_hydrates/stereo_pose_est.data'

	def extract_stereo_info(stereo_filepath):
		# stereo_pose_est.data

		# ['8 ', '1442534994.5840001106262207 ', '37.5383751887904040 ', '137.9430460888389405 ', '-180.2516985319970786 ', '416.3475334547667330 ', '966.2567859154922871 ', '-0.1195825035805833 ', '-0.0077240708471807 ', '-3.1310079485491022 ', 'sx_090900_image0000843_FC.png ', 'sx_090900_image0000843_AC.png ', '6.3090000000000002 ', '2.1488708681282573 ', '0']

		pose_identifier = [] # 1) integer value
		timestamp = [] # 2) in seconds
		latitude = [] # 3) in degrees
		longitude = [] # 4) in degrees
		x_position = [] # 5) in meters (North), relative to local nav frame
		y_position = [] # 6) in meters (East), relative to local nav frame
		z_position = [] # 7) in meters (Depth), relative to local nav frame
		x_axis_euler_angle = [] # 8) in radians, relative to local nav frame
		y_axis_euler_angle = [] # 9) in radians, relative to local nav frame
		z_axis_euler_angle = [] # 10) in radians, relative to local nav frame
		left_image_name = [] # 11)
		right_image_name = [] # 12)
		vehicle_altitude = [] # 13) in meters
		approx_bounding_image_radius = []# 14) in meters
		likely_trajectory_cross_over_point = []# 15) 1 for true, 0 for false

		with open (stereo_filepath ,'r') as f:
			for line in f:
				if '%' not in line and 'ORIGIN' not in line and len(line)>1:
					# print (line)
					# print (line[:-1].split(' 	'))
					# print (line[:-1].split(' 	'))
					# print (line[:-1].split('\t'))

					line_list = line[:-1].split('\t')

					pose_identifier.append(int(line_list[0]))
					timestamp.append(float(line_list[1]))
					latitude.append(float(line_list[2]))
					longitude.append(float(line_list[3]))
					x_position.append(float(line_list[4]))
					y_position.append(float(line_list[5]))
					z_position.append(float(line_list[6]))
					x_axis_euler_angle.append(float(line_list[7]))
					y_axis_euler_angle.append(float(line_list[8]))
					z_axis_euler_angle.append(float(line_list[9]))
					left_image_name.append(line_list[10][:-1])
					right_image_name.append(line_list[11][:-1])
					vehicle_altitude.append(float(line_list[12]))
					approx_bounding_image_radius.append(float(line_list[13]))
					likely_trajectory_cross_over_point.append(int(line_list[14]))

		df = pd.DataFrame()
		df['Left_image_name'] = left_image_name
		df['Right_image_name'] = right_image_name
		df['Northing [m]'] = x_position
		df['Easting [m]'] = y_position
		df['Depth [m]'] = z_position
		# df['Roll [rad]'] = ?
		# df['Pitch [rad]'] = 
		# df['Heading [rad]'] = 
		df['Altitude [m]'] = vehicle_altitude
		df['Timestamp'] = timestamp
		df['Latitude [deg]'] = latitude
		df['Longitude [deg]'] = longitude
		# This order follows adrian mapping pipeline (except that Roll Pitch Yaw is in deg not rad)
		return df

	def extract_vehicle_info(vehicle_filepath):
		pose_identifier = [] # 1) integer value
		timestamp = [] # 2) in seconds
		latitude = [] # 3) in degrees
		longitude = [] # 4) in degrees
		x_position = [] # 5) (Northing) in meters, relative to local nav frame
		y_position = [] # 6) (Easting) in meters, relative to local nav frame
		z_position = [] # 7) (Depth) in meters, relative to local nav frame
		x_axis_euler_angle = [] # 8) (Roll) in radians, relative to local nav frame
		y_axis_euler_angle = [] # 9) (Pitch) in radians, relative to local nav frame
		z_axis_euler_angle = [] # 10) (Yaw/Heading) in radians, relative to local nav frame
		altitude = [] # 11) in meters. (0 when unknown)

		with open (vehicle_filepath ,'r') as f:
			for line in f:
				if '%' not in line and 'ORIGIN' not in line and len(line)>1:

					line_list = line[:-1].split('\t')

					pose_identifier.append(int(line_list[0]))
					timestamp.append(float(line_list[1]))
					latitude.append(float(line_list[2]))
					longitude.append(float(line_list[3]))
					x_position.append(float(line_list[4]))
					y_position.append(float(line_list[5]))
					z_position.append(float(line_list[6]))
					x_axis_euler_angle.append(float(line_list[7]))
					y_axis_euler_angle.append(float(line_list[8]))
					z_axis_euler_angle.append(float(line_list[9]))
					altitude.append(float(line_list[10]))

		df = pd.DataFrame()
		df['Timestamp'] = timestamp
		df['Northing [m]'] = x_position
		df['Easting [m]'] = y_position
		df['Depth [m]'] = z_position
		# df['Roll [rad]'] = ?
		# df['Pitch [rad]'] = 
		# df['Heading [rad]'] = 
		df['Altitude [m]'] = altitude
		df['Latitude [deg]'] = latitude
		df['Longitude [deg]'] = longitude
		return df

	stereo_filepath = filepath + os.sep + 'stereo_pose_est.data'
	stereo_df = extract_stereo_info(stereo_filepath)
	stereo_df.to_csv(filepath + os.sep + 'stereo_pose_est.csv', header=True, index=False)

	# vehicle_filepath = filepath + os.sep + 'vehicle_pose_est.data'
	# vehicle_df = extract_vehicle_info(vehicle_filepath)
	# vehicle_df.to_csv(filepath + os.sep + 'vehicle_pose_est.csv', header=True, index=False)

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print('Error: not enough arguments')
		# syntax_error()
	else:
		# read in filepath, start time and finish time from function call
		flag_i = False
		# flag_q = False
		for i in range(len(sys.argv)):
			option = sys.argv[i]
			if option == "-i":
				filepath = sys.argv[i+1]
				flag_i = True
			# elif option == "-q":
			# 	yaml_filepath = sys.argv[i+1]
			# 	flag_q = True
			# elif option == "-h":
			# 	print ('usage: mosaic_unsupervised_clustering.py -i <filepath containing csv_config.yaml>')
		if flag_i:
			acfr_output_csv(filepath)
		# elif flag_q:
		# 	pass
