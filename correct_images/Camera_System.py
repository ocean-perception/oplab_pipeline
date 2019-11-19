## This is the Class file for camera systems
## Camera_System is the base class
## Biocam, Seaxerocks, Tunasand are the derived classes
## More classess can be derived and defined for future camera systems

class Camera_System:
	def __init__(self):
		# class attributes
		camera_format = None
		camera_name = None
		camera_type = None # bayer pattern
		camera_bit_depth = None
		camera_images_path = None
		camera_calibration_path = None
		camera_number_of_images = None
		camera_image_format = None
		camera_distortion = None
		camera_intrinsics = None
		camera_image_size = None
		camera_debayer_method = None
		
		# class methods
		def read_config_files(self, mission, correct_images):
			# mission --> params for mission.yaml
			# correct_images --> params for correct_images_config.yaml
			# camera_format = from mission.yaml
			# camera_name = from mission.yaml
			# camera_type = from mission.yaml
			# camera_bit_depth = from mission.yaml
			# camera_calibration_path = from correct_images.yaml
			# camera_images_path = from mission.yaml
			# camera_image_format = from correct_images.yaml
			# camera_debayer_method = from correct_images.yaml

		def find_number_of_images(self):
			# count the number of images in the path provided
			# camera_number_of_images = count

		def read_image(self, image_number):
			# function reads a particular image of bit depth and type
			# to be implemented differently for different camera formats
			# return image matrix

		def read_image_size(self, image_number):
			# read size of camera image
			# call read_image first
			# camera_image_size = size

		def read_camera_calibration_parameters(self):
			# read camera calibration parameters from the path provided
			# camera_distortion = matrix
			# camera_intrinsics = matrix

class Seaxerocks(Camera_System):
	def read_image(self, image_number):
		# define the function to read images of type raw

class Tunasand(Camera_System):
	def read_image(self, image_number):
		# define the function to read images of type tif

class Biocam(Camera_System):
	def read_image(self):
		# define the function to read images of type tif




		

