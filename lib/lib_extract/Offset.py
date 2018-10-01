class Offset(object):
	def __init__(self):
		self.x
		self.y
		self.z
		self.roll
		self.pitch
		self.yaw

	def GetX(self):
		return self.x

	def GetRoll(self):
		return self.roll

	def read(self,name, vehicle_data):
		self.x = vehicle_data[name]['x_offset']
		self.y = vehicle_data[name]['y_offset']
		self.z = vehicle_data[name]['z_offset']
		self.roll = vehicle_data[name]['roll_offset']
		self.pitch = vehicle_data[name]['pitch_offset']
		self.yaw = vehicle_data[name]['yaw_offset']