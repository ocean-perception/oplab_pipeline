# body to inertial

# Scripts to rotate a cordinate frame (e.g. body to inertial taking angles in degrees as input)

# Author: Blair Thornton
# Date: 01/09/2017

import math
#http://www.json.org/
deg_to_rad = 3.141592654/180 

class body_to_inertial:
	def __init__(self, roll, pitch, yaw, value_x, value_y, value_z):
		return
		
		
	

	def __new__(cls,roll, pitch, yaw, value_x, value_y, value_z):

		roll=roll*deg_to_rad
		pitch=pitch*deg_to_rad
		yaw=yaw*deg_to_rad

		north=((math.cos(yaw)*math.cos(pitch))*value_x
			+(-math.sin(yaw)*math.cos(roll)+math.cos(yaw)*math.sin(pitch)*math.sin(roll))*value_y
    		+(math.sin(yaw)*math.sin(roll)+(math.cos(yaw)*math.cos(roll)*math.sin(pitch)))*value_z)
		east=((math.sin(yaw)*math.cos(pitch))*value_x
    		+(math.cos(yaw)*math.cos(roll)+math.sin(roll)*math.sin(pitch)*math.sin(yaw))*value_y
    		+(-math.cos(yaw)*math.sin(roll)+math.sin(yaw)*math.cos(roll)*math.sin(pitch))*value_z)
		down=((-math.sin(pitch)*value_x)
    		+(math.cos(pitch)*math.sin(roll)) * value_y
    		+(math.cos(pitch)*math.cos(roll))*value_z)
		
		return north,east,down
