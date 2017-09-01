# parse_acfr_images

# Scripts to parse acfr image acquisition data

# Author: Blair Thornton
# Date: 25/08/2017

from datetime import datetime
import time, json
#http://www.json.org/
data_list=[]
#need to make acfr parsers
class parse_interlacer:
	def __init__(self, filepath, filename, category, timezone, timeoffset, ftype, fileout):

		# parser meta data
