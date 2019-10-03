from feature_descriptors import feature_descriptor

import img_util as iu



class lbp(feature_descriptor):
	radius = 2
	pixel_neighbour = 8
	method = 'default'

	# Size of a window = 100x100
	block_wt = 100
	block_ht = 100



	def __init__(self, img_id):
		super().__init__(img_id)


	def return_descriptor(self):
		# Compute the LBP image
		print("\nIn LBP, ", self.img_id)
		image = iu.img_util.open_image_grayscale(self.img_id)
		iu.img_util.display_image(image, 'Image: ' + self.img_id)

		unified_feat_desc = []  # List to store the unified feature descriptors from each block
		bins = 10  # Since in uniform LBP method, there are only 9 uniform binary values and 1 more for all non-uniform bit patterns
		i = 0

		# Prints the Feat Descriptors computed for each block.
		print('\n\n\nHistogram for Image: ', self.img_id)
		for x in range(0, image.shape[0], self.block_wt):
			for y in range(0, image.shape[1], self.block_ht):
				i += 1
				blocks = image[x: (x + self.block_wt), y: (y + self.block_ht)]  # Extracts 100x100 block from image

				# Calculates LBP for each pixel, since it is uniform the bit pattern is right shifted and we will have only 9 possible values
				lbp_img = iu.img_util.lbp(blocks, "uniform")
				hist_vect = iu.img_util.hist(lbp_img, bins)  # Calculates histogram for each window
				unified_feat_desc.append(hist_vect)  # Stores the histogram vector for each window

				print('\n\nBlock ', i, ':')
				print(hist_vect[0])
		# histogram = np.unique(lbp_img, return_counts=True)
		return unified_feat_desc
