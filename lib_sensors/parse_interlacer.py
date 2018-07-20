# parse_interlacer

# Scripts to interlace data

# Author: Blair Thornton
# Date: 01/09/2017

import json, glob, os
#http://www.json.org/
from array import array 
from operator import itemgetter

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

		# write out interlaced acfr file

		with open(outpath + os.sep + filename,'w') as fileout:
			for i in range(len(data_original)):
				fileout.write(str(data_ordered[i]))

		fileout.close()
