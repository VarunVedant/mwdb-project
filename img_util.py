# Utility class containing utilities for img processing

import config

import cv2
import numpy as np
from skimage import feature
from scipy import misc



class img_util:

	@staticmethod
	def open_image_grayscale(img_file):
		"""
		Opens image in grayscale
		:param path(string):The image file name
		:return(numpy.ndarray): Image as 2D matrix
		"""
		path = config.IMG_LIST_PATH + img_file + '.jpg'
		return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


	@staticmethod
	def display_image(image, label):
		"""
		Displays the image in a window with specified label
		:param image: Image to be displayed
		:param label: Window Label
		:return: None
		"""
		cv2.imshow(label, image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	@staticmethod
	def hist(img, bins):
		"""
		Creates a Histogram for the Image passed
		:param img: Image as represented as a 2D array.
		:param bins: Number of bins in the histogram.
		:return: Returns the histogram vector computed for the Image.
		"""
		flattened_arr = np.ravel(img)
		hist_arr = np.histogram(flattened_arr, bins, range=(0, bins))       # The range mentions the min and max value of the hist
		return hist_arr


	@staticmethod
	def lbp(image, method="default", radius = 2, pixel_neighbour = 8):
		"""
		Computes the LBP for the image passed
		:param image: Grayscale Image for which LBP is to be computed.
		:param method: The LBP method type as listed in feature.local_binary_pattern function. uniform by default.
		:param radius: the radius of neighbourhood for which the lbp is computed. 2 by default.
		:param pixel_neighbour: number of neighbouring pixels. 8 by default.
		:return: LBP Image.
		"""
		lbp_image = feature.local_binary_pattern(image, pixel_neighbour, radius, method=method)
		return lbp_image
