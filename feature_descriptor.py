# Feature Descriptors
import numpy as np
from skimage import feature
import img_util as iu
import config


class feature_descriptor:

	@staticmethod
	def lbp(image, method="default"):
		"""
		Returns the LBP Image for the grayscale image provided, with pixel radius 2 and 8 neighbours.
		:param image: Grayscale Image for which LBP is to be computed
		:param method: The LBP method type as listed in feature.local_binary_pattern function()
		:return: LBP Image
		"""
		radius = 2
		pixel_neighbour = 8
		lbp_image = feature.local_binary_pattern(image, pixel_neighbour, radius, method=method)
		# print(lbp_image.shape)
		return lbp_image



	@staticmethod
	def lbp_feat_desc(img_id):
		"""
		Returns the LBP histogram vector for a given Image ID.
		:param img_id: The image ID of the image to be processed.
		:return: List of histogram vectors for every 100x100 blocks in the image.
		"""

		print("\nIn LBP, ", img_id)
		image = iu.img_util.open_image_grayscale(config.IMG_LIST_PATH + img_id + '.jpg')
		iu.img_util.display_image(image, 'Image: ' + img_id)

		block_wt = block_ht = 100       # Size of a window = 100x100
		unified_feat_desc = []          # List to store the unified feature descriptors from each block
		bins = 10                       # Since in uniform LBP method, there are only 9 uniform binary values and 1 more for all non-uniform bit patterns
		i = 0

		# Prints the Feat Descriptors computed for each block.
		print('\n\n\nHistogram for Image: ', img_id)
		for x in range(0, image.shape[0], block_wt):
			for y in range(0, image.shape[1], block_ht):
				i += 1
				blocks = image[x: (x + block_wt), y: (y + block_ht)]        # Extracts 100x100 block from image

				# Calculates LBP for each pixel, since it is uniform the bit pattern is right shifted and we will have only 9 possible values
				lbp_img = feature_descriptor.lbp(blocks, "uniform")
				hist_vect = iu.img_util.hist(lbp_img, bins)                 # Calculates histogram for each window
				unified_feat_desc.append(hist_vect)                         # Stores the histogram vector for each window

				print('\n\nBlock ', i, ':')
				print(hist_vect[0])
				# histogram = np.unique(lbp_img, return_counts=True)



	@staticmethod
	def hog_feat_desc(img_id):
		print('\nIn HOG, ', img_id)
