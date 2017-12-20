# interpolate

# Scripts to interpolate values

# Author: Blair Thornton
# Date: 14/12/2017

import math

class interpolate:
	def __init__(self, x_query, x_lower, x_upper, y_lower, y_upper):
		return
				

	def __new__(cls, x_query, x_lower, x_upper, y_lower, y_upper):

		if x_upper == x_lower:
			y_query=y_lower

		else:
			y_query=(y_upper-y_lower)/(x_upper-x_lower)*(x_query-x_lower)+y_lower
				

		return y_query
