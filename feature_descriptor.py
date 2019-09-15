"""
Class to compute Feature Descriptors for each model.
"""

import img_util as iu
import config

import os
import json
import numpy as np
import operator
from skimage import feature
from skimage import io
from skimage.feature import hog
from skimage.transform import rescale



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
		Prints the LBP histogram vector for a given Image ID.
		:param img_id: The image ID of the image to be processed.
		:return: None
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
		"""
		Prints the HOG feature descriptor vector.
		:param img_id: Image ID of image to be processed.
		:return: None
		"""
		image = io.imread(config.IMG_LIST_PATH + img_id + '.jpg')
		iu.img_util.display_image(image, 'Image: ' + img_id)
		rescaled_image = rescale(image, 0.1, anti_aliasing=True)
		hog_vec, hog_img = hog(rescaled_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
		print('\nHOG vector calculated: ', hog_vec, '\nHOG Image: \n', hog_img)
		iu.img_util.display_image(hog_img, 'HOG Image')



	@staticmethod
	def compute_lbp_vec(path):
		"""
		Computes the LBP feature descriptor for all the images in the provided path.
		:param path: Path of directory containing images.
		:return: None
		"""
		LBP_map = {}    # Will store list of images paired with their corresponding descriptors
		for img_file in os.listdir(path):
			image = iu.img_util.open_image_grayscale(path + '/' + img_file)
			# print('Image File: ', path + '/' + img_file)
			# print('Image: ', image)

			block_wt = block_ht = 100
			unified_feat_desc = []
			bins = 10

			# Generate histogram for each block
			for x in range(0, image.shape[0], block_wt):
				for y in range(0, image.shape[1], block_ht):
					blocks = image[x: (x + block_wt), y: (y + block_ht)]
					lbp_img = feature_descriptor.lbp(blocks, "uniform")
					hist_vect = iu.img_util.hist(lbp_img, bins)
					unified_feat_desc.append(hist_vect[0])

			LBP_map[img_file] = (np.array(unified_feat_desc)).tolist()

		# Store the unified descriptor in JSON file
		feature_descriptor_file = config.FEAT_DESC_DUMP + 'lbp.json'
		with open(feature_descriptor_file, 'w', encoding='utf-8') as outfile:
			json.dump(LBP_map, outfile, ensure_ascii=True, indent=2)



	@staticmethod
	def compute_hog_vec(path):
		"""
		Computes HOG feature descriptor for all the images in the provided path.
		:param path: Path of the directory containing images.
		:return: None
		"""
		print('\nIn compute_hog_vec')


	@staticmethod
	def fetch_img_desc(img_id, model_ch):
		if model_ch == '1':
			with open(config.FEAT_DESC_DUMP + "lbp.json", "r") as outfile:
				unified_hist = json.load(outfile)
			return unified_hist[img_id + '.jpg']
		else:
			with open(config.FEAT_DESC_DUMP + "hog.json", "r") as outfile:
				unified_hist = json.load(outfile)
			return unified_hist[img_id + '.jpg']



	@staticmethod
	def k_similar_imgs(query_img_id, model_ch, k):
		"""
		Returns the K most similar images to a chosen image
		:param img_id: Image ID of the image
		:return: None
		"""
		query_img_vec = feature_descriptor.fetch_img_desc(query_img_id, model_ch)
		query_img_vec_flat = np.ravel(query_img_vec)

		match_scores = {}
		for img_file in os.listdir(config.TEST_IMGS_PATH):
			img_file_id = img_file.replace('.jpg', '')
			if img_file_id != query_img_id:
				img_vec = feature_descriptor.fetch_img_desc(img_file_id, model_ch)
				img_vec_flat = np.ravel(img_vec)
				# arr_diff = img_vec_flat - query_img_vec_flat
				# match_scores[img_file_id] = np.sqrt(np.dot(arr_diff, arr_diff))
				# print('\ncosine similarity: ', np.dot(img_vec_flat, query_img_vec_flat) / (np.sqrt(img_vec_flat.dot(img_vec_flat)) * np.sqrt(query_img_vec_flat.dot(query_img_vec_flat))))
				match_scores[img_file_id] = np.dot(img_vec_flat, query_img_vec_flat) / (np.sqrt(img_vec_flat.dot(img_vec_flat)) * np.sqrt(query_img_vec_flat.dot(query_img_vec_flat)))
		sorted_scores = sorted(match_scores.items(), key=operator.itemgetter(1))

		query_image = iu.img_util.open_image_grayscale(config.TEST_IMGS_PATH + '/' + query_img_id + '.jpg')
		iu.img_util.display_image(query_image, 'Query Image: ' + query_img_id)
		i = k
		for score in sorted_scores:
			if i == 0:
				break
			print('\nImage ', score[0], ' --> ', score[1])
			i -= 1
			tmp_img = iu.img_util.open_image_grayscale(config.TEST_IMGS_PATH + '/' + score[0] + '.jpg')
			iu.img_util.display_image(tmp_img, str(k-i) + 'th similar img: ' + score[0])
