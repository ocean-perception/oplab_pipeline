# this is the class file for implementing the various correction algorithms
# IMPORT --------------------------------
# all imports go here 
# -----------------------------------------


class Corrector:
	def __init__(self, camera_system=None, correct_params=None):
		if correct_params is not None:
			self.attenuation_correct = correct_params['apply_attenuation_correction']
			self.distortion_correct = correct_params['apply_distortion_correction']
			self.gamma_correct = correct_params['apply_gamma_correction']
			self.camera_parameters_path = correct_params['camera_parameter_file_path']
			self.target_mean = correct_params['target_mean']
			self.target_std = correct_params['target_std']
		if camera_system is not None:
			self.camera_system = camera_system


	def Execute(self):
		
		# resolve paths for attenuation paremeters and output directories
		cameras = camera_system.cameras
		for camera in cameras:
			camera_name = camera.name
			bayer_pattern = camera.pattern
			image_type = camera.extension

			# Resolve path to camera images
			# TODO ----------------------
			# --------------------------

			# Compute attenuation correction coefficients

			self.calculate_attenuation_correction() # entire parse code (restructured) goes in here
			# TODO details on the data structure to read attenuation params
			# ....................

			# Apply corrections as opted by user through correct_config.yaml file
			for bayer_image in bayer_image_list:
				if self.attenuation_correct == True:
					corrected_bayer_image = self.attenuation_correction_bayer(bayer_image)
				else:
					self.pixel_stat(bayer_image)
				corrected_rgb_img = self.debayer(bayer_image, bayer_pattern, image_type) # Debayering
				
				if self.distortion_correct == True:
					map_x, map_y = self.calc_distortion_mapping(camera_parameters_path, bayer_img_size)
					corrected_rgb_img = self.correct_distortion(corrected_rgb_img, map_x, map_y)

				if self.gamma_correct == True:
					corrected_rgb_img = self.gamma_correct(corrected_rgb_img)
				self.write_image(dest_path)

		self.write_config()

	def calculate_attenuation_correction(self):
		pass

	def attenuation_correction_bayer(self, bayer_image):
		pass

	def pixel_stat(self, bayer_image):
		pass

	def debayer(bayer_image):
		pass

	def calc_distortion_mapping(self, camera_parameters_path, bayer_img_size):
		pass

	def correct_distortion(self, corrected_rgb_img, map_x, map_y):
		pass

	def gamma_correct(self, corrected_rgb_img):
		pass

	def write_image(self, dest_path):
		pass

	def write_config():
		pass






