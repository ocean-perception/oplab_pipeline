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

# https://plot.ly/python/time-series/
# https://plot.ly/python/table-subplots/
# https://plot.ly/python/reference/#layout-xaxis-rangeslider
# https://help.plot.ly/date-format-and-time-series/
# https://plot.ly/python/reference/#layout-xaxis-tickformat
# https://community.plot.ly/t/how-to-make-the-messy-date-ticks-organized/7477/3
# https://plot.ly/python/reference/#layout-xaxis-tickvals
# https://github.com/d3/d3-format/blob/master/README.md#locale_format
# https://github.com/d3/d3-time-format/blob/master/README.md#locale_format
# https://plot.ly/python/reference/#layout-updatemenus

from prettytable import PrettyTable, ALL
import sys, os
import json, time
import math
from matplotlib import collections as matcoll
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import plotly.offline as py
import plotly.graph_objs as go

# sys.path.append("..")
from lib_converttime.converttime import epoch_to_localtime, get_localtimezone

class display_info:
	# Goes through each data element in json file and first check for different types category, 
	# and then different types of frame (body or inertial). Displays a sample information for each different 
	# type of data. In the future if element in json file contains more variation other than different 
	# category/frame, need to expand code to check for variations in them too.
	def __init__(self, filepath, ftype):
		if ftype == 'oplab':

			# Helper functions
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

			def create_trace(x_list, y_list, trace_name): #,trace_color):
				trace = go.Scattergl(x=x_list,y=y_list,name=trace_name,mode='markers') # ,xaxis='x1',yaxis='y1')
					#marker=dict(colour=trace_color, ''),
					#'rgba(152, 0, 0, .8)'),#,size = 10, line = dict(width = 2,color = 'rgb(0, 0, 0)'),
	                # line=dict(color=trace_color)#rgb(205, 12, 24)'))#, width = 4, dash = 'dot')
	               	# legendgroup='group11'
				return trace

			outpath = filepath + os.sep # + 'nav' + os.sep

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
			trace_list = []
			plotlytable_info_list = [[],[],[],[]]
			for i in full_data_list:
				for j in i:
					if not j:
						pass
					else:
						#
						title = j[0]['category'] + ' - ' + j[0]['frame']
						titles.append(title)
						epoch_timestamp_data_points.append(j)

						# print the info
						time_difference = j[1]['epoch_timestamp'] - j[0]['epoch_timestamp']
						n = 0
						while time_difference == 0 or time_difference < 0.002:
							n += 1
							time_difference = j[1 + n]['epoch_timestamp'] - j[n]['epoch_timestamp']
						line = print_lines(j[0], time_difference)
						t.add_row([j[0]['category'], len(j), line[0], line[1]])

						# plotly table
						plotlytable_info_list[0].append(j[0]['category'])
						plotlytable_info_list[1].append(len(j))
						plotlytable_info_list[2].append(line[0].replace('\n','<br>'))
						plotlytable_info_list[3].append(line[1].replace('\n','<br>'))

						# plotly plot
						x_values = [time.strftime('%Y-%m-%d %H:%M:%S', epoch_to_localtime(k['epoch_timestamp']))+'.{}'.format(('{:.6f}'.format(k['epoch_timestamp']-int(k['epoch_timestamp'])))[2:9]) for k in j] # format is 'yyyy-mm-dd HH:MM:SS.ssssss'
						# x_values = [k['epoch_timestamp'] for k in j]
						y_values = [title] * len(x_values)
						trace_list.append(create_trace(x_values, y_values, title))
			
			table_trace = go.Table(
				# domain=dict(x=[0,1],
							# y=[0.4,1]),
				columnorder = [1,2,3,4],
				columnwidth = [1,1,2,5], #[80,400]
				header = dict(
					values = [['<b>Category</b>'],['<b>No. of data</b>'],['<b>Details</b>'],['<b>Sample Value</b>']],
					line = dict(color = '#506784'),
					fill = dict(color = '#119DFF'),
					align = ['center','center', 'center','left'],
					font = dict(color = 'white', size = 12),
					height = 40
				),
				cells = dict(
					values = plotlytable_info_list,
					line = dict(color = '#506784'),
					fill = dict(color = ['#25FEFD', 'white']),
					align = ['center', 'center', 'center', 'left'],
					font = dict(color = '#506784', size = 12),
					height = 30
				)
			)

			layout_table = go.Layout(title='Json Data Info Table<br>Start time is: %s (%s), %d (epoch)<br>Finish time is: %s (%s), %d (epoch)' % (time.strftime('%Y-%m-%d %H:%M:%S', epoch_to_localtime(start_time)), get_localtimezone(), start_time, time.strftime('%Y-%m-%d %H:%M:%S', epoch_to_localtime(finish_time)), get_localtimezone(), finish_time))
			table_fig = go.Figure(data=[table_trace], layout=layout_table)
			py.plot(table_fig, filename=outpath + 'json_data_info.html', auto_open=True)

			layout = go.Layout(
				# width=950,
				# height=800,
				title='Timestamp History Plot<br>Start time is: %s (%s), %d (epoch)<br>Finish time is: %s (%s), %d (epoch)' % (time.strftime('%Y-%m-%d %H:%M:%S', epoch_to_localtime(start_time)), get_localtimezone(), start_time, time.strftime('%Y-%m-%d %H:%M:%S', epoch_to_localtime(finish_time)), get_localtimezone(), finish_time),
				hovermode='closest',
				xaxis=dict(
					# domain=[0,1],
					# anchor='y1',
					title='Date time (%s)' % (get_localtimezone()),
					# tickmode='array',
					# tickvals=list(range(int(start_time), int(finish_time), 300)),
					# ticktext=[time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(k)) for k in list(range(int(start_time), int(finish_time), 300))],
					# hoverformat='%x %X', #'%s'
					# tickformat='%H %M %S',#'%X', 
					# tickformat='.3f',
					# nticks= 10,
					rangeselector = dict(
						buttons=list([
							dict(count=5,
								 label='5 secs',
								 step='second',
								 stepmode='backward'),
							dict(count=60,
								 label='1 min',
								 step='second',
								 stepmode='backward'),
							dict(count=300,
								 label='5 mins',
								 step='second',
								 stepmode='backward'),
							dict(count=1200,
								 label='20 mins',
								 step='second',
								 stepmode='backward'),
							dict(count=3600,
								 label='1 hour',
								 step='second',
								 stepmode='backward'),
							dict(count=7200,
								 label='2 hours',
								 step='second',
								 stepmode='backward'),
							dict(step='all')
						])
					),
					rangeslider=dict(thickness=0.05),
					type='date'
				),
				yaxis=dict(
					# domain=[0, 0.32],
					title='Category-Frame',
					# anchor='x1'
				),
				dragmode='pan',
				margin = go.Margin(
					l=150
				)
			)
			config={'scrollZoom':True}
			fig = go.Figure(data=list(reversed(trace_list)), layout=layout)
			py.plot(fig, config=config, filename=outpath + 'timestamp_history.html', auto_open=True)

		# Create a plot that represents the timestamp history of each data variation.
			# print ('Creating plot')
			# f, ax = plt.subplots(nrows = len(epoch_timestamp_data_points), sharex=True)
			# n = 0
			# print ('length of epoch_timestamp_data_points list: ', len(epoch_timestamp_data_points))
			# for i in epoch_timestamp_data_points:
			# 	print ('Plotting %i datapoints of %s ...' % (len(i), titles[n]))
			# 	lines = []
			# 	for j in i:
			# 		pair = [(j['epoch_timestamp'], 0),(j['epoch_timestamp'],1)]
			# 		lines.append(pair)
			# 	linecoll = matcoll.LineCollection(lines)
			# 	ax[len(epoch_timestamp_data_points)-n-1].add_collection(linecoll)
			# 	ax[len(epoch_timestamp_data_points)-n-1].set_title(titles[n], y=0)
			# 	plt.sca(ax[len(epoch_timestamp_data_points)-n-1])
			# 	plt.yticks(())
			# 	plt.ylim(0, 1)
			# 	plt.xticks(rotation = 'vertical')
			# 	n += 1
			# plt.axis([start_time, finish_time, 0, 1])
			# x_formatter = ticker.FuncFormatter(lambda x, pos:'{0:s}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x))))#gmtime(x))))
			# # x_formatter = ticker.ScalarFormatter(useOffset=False)
			# # x_formatter = ticker.FormatStrFormatter('%0.0f')
			# ax[0].xaxis.set_major_formatter(x_formatter)
			# plt.tight_layout()
			# f.subplots_adjust(hspace = 0)
			# # plt.show()
			# f.savefig(outpath + 'timestamp_history.pdf')
			# plt.close()

			start_end_text = 'Start time is: %s (%s), %d (epoch)\nFinish time is: %s (%s), %d (epoch)\n' % (time.strftime('%Y-%m-%d %H:%M:%S', epoch_to_localtime(start_time)), get_localtimezone(), start_time, time.strftime('%Y-%m-%d %H:%M:%S', epoch_to_localtime(finish_time)), get_localtimezone(), finish_time) #changed from gmtime to localtime
			
			t.align['Sample Value'] = 'l'
			t.hrules = ALL
			# text_file = open(outpath + 'json_data_info.txt', 'w')
			# text_file.write(start_end_text)
			# text_file.write(t.get_string())
			# text_file.close()
			print (start_end_text)
			print (t)
			print('Outputs saved to %s' %(outpath))
		else:
			print('ACFR ftype to be done')