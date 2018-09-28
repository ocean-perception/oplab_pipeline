# dead_reckoning

# Scripts to integrate velocities and return positions

# Author: Blair Thornton
# Date: 14/12/2017

import math

class dead_reckoning:
	def __init__(self, time_now, time_previous, north_velocity_previous, east_velocity_previous, northings_previous, eastings_previous):
		return
				

	#def __new__(cls, time_now, time_previous, north_velocity_previous, east_velocity_previous, northings_previous, eastings_previous):

		#northings_now = (time_now - time_previous)*north_velocity_previous + northings_previous
		#eastings_now = (time_now - time_previous)*east_velocity_previous + eastings_previous				
	
	def __new__(cls, time_now, time_previous, north_velocity_now, north_velocity_previous, east_velocity_now, east_velocity_previous, northings_previous, eastings_previous):

		northings_now = ((time_now - time_previous)*(north_velocity_previous + north_velocity_now))/2 + northings_previous
		eastings_now = ((time_now - time_previous)*(east_velocity_previous + east_velocity_now))/2 + eastings_previous

		return northings_now, eastings_now
