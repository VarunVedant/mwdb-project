# Utility class containing utilities for img processing

import cv2
import numpy as np
from scipy import misc



class img_util:

	@staticmethod
	def open_image_grayscale(path):
		"""
		Opens image in grayscale
		:param path(string): The complete path where the image can be found
		:return(numpy.ndarray): Image as 2D matrix
		"""
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
