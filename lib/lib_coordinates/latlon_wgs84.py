# latlon_to_metres

# Scripts to convert latlon (decimal degrees) to range and bearing using the Haversine formula
# and convert distance from latitude and longitude references to latitude longitude uncertainty
# http://www.movable-type.co.uk/scripts/latlong.html

# Author: Blair Thornton
# Date: 31/08/2017

import math

class latlon_to_metres:
	def __init__(self, latitutde, longitude, latitutde_reference, longitude_reference):
		self.distance=distance
		self.bearing=bearing

		self.latitude_offset=latitude_offset
		self.longitude_offset=longitude_offset


def latlon_to_metres(latitude, longitude, latitude_reference, longitude_reference):
	
	# average radius of earth, m
	R=6378137.0

	latitude=latitude*math.pi/180.0
	latitude_reference=latitude_reference*math.pi/180.0

	longitude=longitude*math.pi/180.0
	longitude_reference=longitude_reference*math.pi/180.0

	a = math.sin((latitude-latitude_reference)/2)*math.sin((latitude-latitude_reference)/2) + math.cos(latitude_reference) * math.cos(latitude) * math.sin((longitude-longitude_reference)/2) * math.sin((longitude-longitude_reference)/2)
	c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

	distance=R*c

	y = math.sin(longitude-longitude_reference) * math.cos(latitude)
	x = math.cos(latitude_reference) * math.sin(latitude) - math.sin(latitude_reference) * math.cos(latitude) * math.cos(longitude-longitude_reference)

	bearing = math.atan2(y,x)*180/math.pi	

	while bearing > 360:
		bearing = bearing - 360
	while bearing < 0:
		bearing = bearing + 360
    
	return (distance,bearing)

def metres_to_latlon(latitude, longitude, eastings, northings):

	# average radius of earth, m
	R=6378137.0
	
	latitude=latitude*math.pi/180.0
	longitude=longitude*math.pi/180.0

	latitude_offset=math.asin( math.sin(latitude) *  math.cos(northings/R) + math.cos(latitude) * math.sin(northings/R) * math.cos(0.0*math.pi/180.0)   )
	longitude_offset=longitude + math.atan2( math.sin(90.0*math.pi/180.0) *  math.sin(eastings/R) * math.cos(latitude) , math.cos(eastings/R) - math.sin(latitude) * math.sin(latitude_offset))

	latitude_offset=latitude_offset*180.0/math.pi
	longitude_offset=longitude_offset*180.0/math.pi
	
	return (latitude_offset,longitude_offset)
