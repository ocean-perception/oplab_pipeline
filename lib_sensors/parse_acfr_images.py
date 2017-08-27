# parse_acfr_images

# Scripts to parse acfr image acquisition data

# Author: Blair Thornton
# Date: 25/08/2017

import csv
from datetime import datetime
import codecs, time, json
#http://www.json.org/
data_list=[]
#need to make acfr parsers
class parse_acfr_images:
	def __init__(self, filepath, filename, category, timezone, timeoffset, ftype, fileout):

		# parser meta data
