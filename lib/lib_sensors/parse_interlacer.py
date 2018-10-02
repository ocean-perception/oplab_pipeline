# parse_interlacer

# Scripts to interlace data

# Author: Blair Thornton
# Date: 01/09/2017

import json, glob, os
#http://www.json.org/
from array import array 
from operator import itemgetter

### temporary solution for fk180731 cruise ###
import numpy as np
# sys.path.append("..")
from lib_usbl_filter.usbl_filter import usbl_filter
from lib_extract import sensor_classes as sens_cls
import time
from datetime import datetime
### temporary solution for fk180731 cruise ###

# foo_list = []	# debug

#class parse_interlacer:
#	def __init__(self, ftype, outpath, filename):
def parse_interlacer(ftype, outpath, filename):
	data = array('f')
	value = []
	data_original = []
	data_ordered = []

	# debug
	# bar_list=[]
	# print("Length of foo_list:", len(foo_list)) # prints 0 on first run, 2 on second run
	# print("Length of bar_list:", len(bar_list)) # prints 0 on first run, always
	# foo_list.append("foo_list is part of the module.") 
	# foo_list.append("What is added now will be there when this module is called next time.")
	# bar_list.append("bar_list is a local variable.")
	# bar_list.append("It will start empty when the module is called next time, regardless of what we do with it this time.")
	# end debug

	if ftype == 'oplab':
			for filein in glob.glob(outpath + os.sep + filename):
				try:
					with open(filein, 'r') as json_file:
						data=json.load(json_file)
						
						for i in range(len(data)):
							data_packet=data[i]
							value.append(str(float(data_packet['epoch_timestamp'])))
				
				except ValueError:
					print('Error: no data in JSON file')
				
				# sort data in order of epoch_timestamp
				sorted_index, sorted_items  = zip(*sorted([(i,e) for i,e in enumerate(value)], key=itemgetter(1)))
			
			# store interlaced data in order of time
			for i in range(len(data)):
				data_ordered.append((data[sorted_index[i]]))

			# write out interlaced json file
			with open(outpath + os.sep + filename,'w') as fileout:
				json.dump(data_ordered, fileout)

	if ftype == 'acfr':
		try:
			with open(outpath + os.sep + filename, 'r') as acfr_file:
				for line in acfr_file.readlines():
					line_split = line.strip().split(':')
					line_split_tailed = line_split[1].strip().split(' ')
					value.append(str(line_split_tailed[0]))
					data_original.append(line)

		except ValueError:
				print('Error: no data in RAW.auv file')

		# sort data in order of epoch_timestamp
		sorted_index, sorted_items  = zip(*sorted([(i,e) for i,e in enumerate(value)], key=itemgetter(1)))

		for i in range(len(data_original)):
			data_ordered.append(data_original[sorted_index[i]])

		### usbl filter , temporary solution for fk180731 cruise ### !! mostly from extract_data.py. merge the functions at one place if implementing usbl filter here!!!
		# def datetime_to_epochtime(yyyy, mm, dd, hours, mins, secs):
		# 	dt_obj = datetime(yyyy,mm,dd,hours,mins,secs)
		# 	time_tuple = dt_obj.timetuple()
		# 	return time.mktime(time_tuple)

		# usbl_list = []
		# depth_list = []

		# epoch_start_time = datetime_to_epochtime(2018,8,3,15,30,0)
		# epoch_finish_time = datetime_to_epochtime(2018,8,3,20,22,0)

		# index2remove = []
		# for z in range(len(data_ordered)):
		# 	line_split = data_ordered[z].split(' ')
		# 	if str(line_split[0]) == 'PAROSCI:':
		# 		epoch_timestamp=float(line_split[1])
		# 		if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
		# 			depth = sens_cls.depth()
		# 			depth.timestamp = float(line_split[1])
		# 			depth.depth = float(line_split[2])
		# 			depth_list.append(depth)
		# 	if str(line_split[0]) == 'SSBL_FIX:':
		# 		epoch_timestamp=float(line_split[1])
		# 		if epoch_timestamp >= epoch_start_time and epoch_timestamp <= epoch_finish_time:
		# 			usbl = sens_cls.usbl()
		# 			usbl.timestamp = float(line_split[1])
		# 			for i in range(len(line_split)-1):
		# 				if line_split[i] == 'ship_x:':
		# 					usbl.northings_ship = float(line_split[i+1])
		# 				if line_split[i] == 'ship_y:':
		# 					usbl.eastings_ship = float(line_split[i+1])
		# 				if line_split[i] == 'target_x:':
		# 					usbl.northings = float(line_split[i+1])
		# 				if line_split[i] == 'target_y:':
		# 					usbl.eastings = float(line_split[i+1])
		# 				if line_split[i] == 'target_z:':
		# 					usbl.depth = float(line_split[i+1])
		# 				if line_split[i] == 'target_hr:':
		# 					usbl.lateral_distance = float(line_split[i+1])
		# 				if line_split[i] == 'target_sr:':
		# 					usbl.distance = float(line_split[i+1])
		# 				if line_split[i] == 'target_bearing:':
		# 					usbl.bearing = float(line_split[i+1])
		# 			usbl_list.append(usbl)
		# 		index2remove.append(z)

		# # remove all usbl data
		# data_ordered = np.array(data_ordered)
		# data_ordered = np.delete(data_ordered, index2remove)
		# data_ordered = data_ordered.tolist()

		# new_usbl_list = usbl_filter(usbl_list, depth_list, 2, 2, 'acfr')

		# # add all usbl data
		# for i in new_usbl_list:
		# 	data = 'SSBL_FIX: ' + str(float(i.timestamp)) + ' ship_x: ' + str(float(i.northings_ship)) + ' ship_y: ' + str(float(i.eastings_ship)) + ' target_x: ' + str(float(i.northings)) + ' target_y: ' + str(float(i.eastings)) + ' target_z: ' + str(float(i.depth)) + ' target_hr: ' + str(float(i.lateral_distance)) + ' target_sr: ' + str(float(i.distance)) + ' target_bearing: ' + str(float(i.bearing)) + '\n'
		# 	data_ordered.append(data)

		# value = []
		# # redo sorting 
		# for line in data_ordered:
		# 	line_split = line.strip().split(':')
		# 	line_split_tailed = line_split[1].strip().split(' ')
		# 	value.append(str(line_split_tailed[0]))

		# # sort data in order of epoch_timestamp
		# sorted_index, sorted_items  = zip(*sorted([(i,e) for i,e in enumerate(value)], key=itemgetter(1)))
		# new_data_ordered = []
		# for i in range(len(data_ordered)):
		# 	new_data_ordered.append(data_ordered[sorted_index[i]])
		# ##

		# with open(outpath + os.sep + filename,'w') as fileout:
		# 	for i in range(len(new_data_ordered)):
		# 		fileout.write(str(new_data_ordered[i]))
	
		### usbl filter , temporary solution for fk180731 cruise ###
		
		# write out interlaced acfr file
		with open(outpath + os.sep + filename,'w') as fileout:
			for i in range(len(data_original)):
				fileout.write(str(data_ordered[i]))
		
		fileout.close()
