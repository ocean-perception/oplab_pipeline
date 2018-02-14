# This looks at nav_standard.json file stored in the format of [{"<DataName>":<Value>,"<DataName>":<Value>, ...}, ...] 
# and displays the information such as 'Categories' in it, start date time, finish date time, etc...
"""Display infomation of json file​
		inputs are
			data_check.py <options>
				-i <path to nav folder containing json file>
	example input: 
	python data_check.py -i \\OPLAB-SURF\data\reconstruction\processed\2017\SSK17-01\ts_un_006\
"""
# Author: Jin Wei Lim
# Date: 27/12/2017

from prettytable import PrettyTable, ALL
import sys, os
import json, time
import math
from matplotlib import collections as matcoll
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class display_info:
	# Goes through each data element in json file and first check for different types category, 
	# and then different types of frame (body or inertial). Displays a sample information for each different 
	# type of data. In the future if element in json file contains more variation other than different 
	# category/frame, need to expand code to check for variations in them too.
	def __init__(self, filepath, ftype):
		if ftype == 'oplab':
			def print_lines(object, time_difference):
				line = ['','']
				for i in object:
					if 'data' in i:
						line[0] += i + '\n'
						index = 0
						for j in object[i]:
							if index == 0:
								line[1] += str(j) + '\n'
								index += 1
							else: 
								line[1] += str(j) + '\n'
								line[0] += '\n'
					else:
						line[0] += i + '\n'
						line[1] += str(object[i]) + '\n'
				line[0] += 'Approximate update rate (Hz)'
				line[1] += str(1/time_difference)
				if object['frame'] != 'body':
					if object['frame'] != 'inertial':
						line[0] += 'Warning\n'
						line[1] += 'Multiple ''frame''in this category\n'
				return line
			
			outpath = filepath + 'nav' + os.sep

			print('Loading json file')
			category_list = [] # Contains all the 'category' in the json file.
			full_data_list = [] # Contains the same number of elements as <category_list>, each containing additional list of elements that are data from different 'frame'.
			start_time = 0
			finish_time = 0 

			# Loads and sorts data elements into their respective 'category' and 'frame'.
			with open(outpath + 'nav_standard.json', 'rb') as json_file:
				data_in = json.load(json_file)
				start_time = data_in[0]['epoch_timestamp']
				for i in data_in:
					# to capture start_time and finish_time
					if i['epoch_timestamp'] < start_time:
						start_time = i['epoch_timestamp']
					if i['epoch_timestamp'] > finish_time:
						finish_time = i['epoch_timestamp']
					# to find out how many categories are there
					if i['category'] in category_list:
						# to record all different types of frames
						if i['frame'] == 'body':
							#if not category_sample_list[category_list.index(i['category'])][0]:
							full_data_list[category_list.index(i['category'])][0].append(i)
						elif i['frame'] == 'inertial':
							full_data_list[category_list.index(i['category'])][1].append(i)
						else:
							if not full_data_list[category_list.index(i['category'])][2]:
								full_data_list[category_list.index(i['category'])][2].append(i)
								print ('Warning: %s''s frame contains something different than body or inertial --> %s' % (i['category'], i))
							else:
								flag_same_frame = 0
								for j in full_data_list[category_list.index(i['category'])][2:]:
									if j['frame'] == i['frame']:
										flag_same_frame = 1
										j.append(i)
								if flag_same_frame == 0:
									full_data_list[category_list.index(i['category'])].append([])
									full_data_list[category_list.index(i['category'])][-1].append(i)
									print ('Warning: %s''s frame contains more than 1 different obeject other than body and inertial --> %s' % (i['category'], i))
					else:
						category_list.append(i['category'])
						full_data_list.append([[],[],[]])

						# to record all different types of frames
						if i['frame'] == 'body':
							full_data_list[category_list.index(i['category'])][0].append(i)
						elif i['frame'] == 'inertial':
							full_data_list[category_list.index(i['category'])][1].append(i)
						else:
							full_data_list[category_list.index(i['category'])][2].append(i)
							print ('Warning: %s''s frame contains something different than body or inertial --> %s' % (i['category'], i))

			# Create a table of each data 'category' and 'frame' variation, with additional approximation of how 
			# frequent the sensors collects data through calculating the difference between the first two epoch_timestamp.
			print ('Creating table')
			t = PrettyTable(['Category', 'No. of data', 'Details', 'Sample Value'])
			epoch_timestamp_data_points = []
			titles = []
			for i in full_data_list:
				for j in i:
					if not j:
						pass
					else:
						#
						titles.append(j[0]['category'] + ' - ' + j[0]['frame'])
						epoch_timestamp_data_points.append(j)

						#
						time_difference = j[1]['epoch_timestamp'] - j[0]['epoch_timestamp']
						n = 0
						while time_difference == 0:
							n += 1
							time_difference = i[1 + n]['epoch_timestamp'] - i[n]['epoch_timestamp']
						line = print_lines(j[0], time_difference)
						t.add_row([j[0]['category'], len(j), line[0], line[1]])
			# Create a plot that represents the timestamp history of each data variation.
			print ('Creating plot')
			f, ax = plt.subplots(nrows = len(epoch_timestamp_data_points), sharex=True)
			n = 0
			print ('length of epoch_timestamp_data_points list: ', len(epoch_timestamp_data_points))
			for i in epoch_timestamp_data_points:
				print ('Plotting %i datapoints of %s ...' % (len(i), titles[n]))
				lines = []
				for j in i:
					pair = [(j['epoch_timestamp'], 0),(j['epoch_timestamp'],1)]
					lines.append(pair)
				linecoll = matcoll.LineCollection(lines)
				ax[len(epoch_timestamp_data_points)-n-1].add_collection(linecoll)
				ax[len(epoch_timestamp_data_points)-n-1].set_title(titles[n], y=0)
				plt.sca(ax[len(epoch_timestamp_data_points)-n-1])
				plt.yticks(())
				plt.ylim(0, 1)
				plt.xticks(rotation = 'vertical')
				n += 1
			plt.axis([start_time, finish_time, 0, 1])
			x_formatter = ticker.FuncFormatter(lambda x, pos:'{0:s}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(x))))#ScalarFormatter(useOffset=False)
			ax[0].xaxis.set_major_formatter(x_formatter)
			plt.tight_layout()
			f.subplots_adjust(hspace = 0)
			plt.show()
			f.savefig(outpath + 'timestamp_history.pdf')
			plt.close()

			start_end_text = 'Start time is: %s (UTC), %d (epoch)\nFinish time is: %s (UTC), %d (epoch)\n' % (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time)), start_time, time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(finish_time)), finish_time)
			
			t.align['Sample Value'] = 'l'
			t.hrules = ALL
			text_file = open(outpath + 'json_data_info.txt', 'w')
			text_file.write(start_end_text)
			text_file.write(t.get_string())
			text_file.close()
			print (start_end_text)
			print (t)
			print('Outputs saved to %s' %(outpath))
		else:
			print('ACFR ftype to be done')