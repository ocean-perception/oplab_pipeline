# parse_gaps

# Scripts to parse ixsea blue gaps data

# Author: Blair Thornton
# Date: 25/08/2017

import csv
from datetime import datetime
import codecs, time, json
#http://www.json.org/
data_list=[]
#need to make acfr parsers
class parse_gaps:
	def __init__(self, filepath, filename, category, timezone, timeoffset, ftype, fileout):

		# parser meta data
