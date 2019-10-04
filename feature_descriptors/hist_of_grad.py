import config
import img_util as iu
from feature_descriptors import feature_descriptor

from skimage import io
from skimage.feature import hog
from skimage.transform import rescale



class hist_of_grad(feature_descriptor):

	orientations = 9
	pixels_per_cell = (8, 8)
	cells_per_block = (2, 2)
	visualize = True
	multichannel = True


	def __init__(self, img_id):
		super().__init__(img_id)


	def return_descriptor(self):
		image = io.imread(config.IMG_LIST_PATH + self.img_id + '.jpg')
		iu.img_util.display_image(image, 'Image: ' + self.img_id)
		rescaled_image = rescale(image, 0.1, anti_aliasing=True)
		hog_vec, hog_img = hog(rescaled_image, self.orientations, self.pixels_per_cell, self.cells_per_block, self.visualize, self.multichannel)
		print('\nHOG vector calculated: ', hog_vec, '\nHOG Image: \n', hog_img)
		# iu.img_util.display_image(hog_img, 'HOG Image')
		return hog_vec
