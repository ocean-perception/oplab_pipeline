# body to inertial

# Scripts to rotate a cordinate frame (e.g. body to inertial taking angles in degrees as input)

# Author: Blair Thornton
# Date: 01/09/2017

import math
#http://www.json.org/
deg_to_rad = 3.141592654/180 

class body_to_inertial:
	def __init__(self, roll, pitch, yaw, old_x, old_y, old_z):
		return
		
		
	

	def __new__(cls,roll, pitch, yaw, old_x, old_y, old_z):

		roll=roll*deg_to_rad
		pitch=pitch*deg_to_rad
		yaw=yaw*deg_to_rad

		new_x=((math.cos(yaw)*math.cos(pitch))*old_x
			+(-math.sin(yaw)*math.cos(roll)+math.cos(yaw)*math.sin(pitch)*math.sin(roll))*old_y
    		+(math.sin(yaw)*math.sin(roll)+(math.cos(yaw)*math.cos(roll)*math.sin(pitch)))*old_z)
		new_y=((math.sin(yaw)*math.cos(pitch))*old_x
    		+(math.cos(yaw)*math.cos(roll)+math.sin(roll)*math.sin(pitch)*math.sin(yaw))*old_y
    		+(-math.cos(yaw)*math.sin(roll)+math.sin(yaw)*math.cos(roll)*math.sin(pitch))*old_z)
		new_z=((-math.sin(pitch)*old_x)
    		+(math.cos(pitch)*math.sin(roll)) * old_y
    		+(math.cos(pitch)*math.cos(roll))*old_z)
				

		return new_x,new_y,new_z
