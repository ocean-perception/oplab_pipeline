class velocity_body:
	def __init__(self):
		self.timestamp = 0
		self.x_velocity = 0
		self.y_velocity = 0
		self.z_velocity = 0
		self.x_velocity_std = 0
		self.y_velocity_std = 0
		self.z_velocity_std = 0

class velocity_inertial:
	def __init__(self):
		self.timestamp = 0
		self.north_velocity = 0
		self.east_velocity = 0
		self.down_velocity = 0
		self.north_velocity_std = 0
		self.east_velocity_std = 0
		self.down_velocity_std = 0
		# mabye separate below to synced_velocity_inertial_orientation_... or call it roll yaw pitch instead of roll_interpolated, pitch...?
		self.roll_interpolated = 0
		self.pitch_interpolated = 0
		self.yaw_interpolated = 0
		self.northings_dr = 0
		self.eastings_dr = 0
		self.depth_dr = 0

class orientation:
	def __init__(self):
		self.timestamp = 0
		self.roll = 0
		self.pitch = 0
		self.yaw = 0
		self.roll_std = 0
		self.pitch_std = 0
		self.yaw_std = 0

class depth:
	def __init__(self):
		self.timestamp = 0
		self.depth = 0
		self.depth_std = 0

class altitude:
	def __init__(self):
		self.timestamp = 0
		self.altitude = 0
		self.altitude_std = 0
		# interpolate depth and add altitude for every altitude measurement
		self.seafloor_depth = 0

class usbl:
	def __init__(self):
		self.timestamp = 0
		self.latitude = 0
		self.longitude = 0
		self.northings = 0
		self.eastings = 0
		self.latitude_std = 0
		self.longitude_std = 0
		self.northings_std = 0
		self.eastings_std = 0
		self.depth = 0

class camera:
	def __init__(self):
		self.timestamp = 0
		self.filename = ''
		#
		self.northings_dr = 0
		self.eastings_dr = 0
		self.depth_dr = 0
		self.roll_interpolated = 0
		self.pitch_interpolated = 0
		self.yaw_interpolated = 0
		self.altitude_interpolated = 0

class synced_orientation_velocity_body:
	def __init__(self):
		self.timestamp = 0
		# from orientation
		self.roll = 0
		self.pitch = 0
		self.yaw = 0
		self.roll_std = 0
		self.pitch_std = 0
		self.yaw_std = 0
		# interpolated
		self.x_velocity = 0
		self.y_velocity = 0
		self.z_velocity = 0
		self.x_velocity_std = 0
		self.y_velocity_std = 0
		self.z_velocity_std = 0
		# transformed
		self.north_velocity = 0 # these will be the values used for PF etc!
		self.east_velocity = 0 # these will be the values used for PF etc!
		self.down_velocity = 0 # these will be the values used for PF etc!
		self.north_velocity_std = 0 # these will be the values used for PF etc!
		self.east_velocity_std = 0 # these will be the values used for PF etc!
		self.down_velocity_std = 0 # these will be the values used for PF etc!
		# interpolated
		self.altitude = 0
		# calculated
		self.northings_dr = 0
		self.eastings_dr = 0
		self.depth = 0 # from interpolation of depth, not dr