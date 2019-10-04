from feature_descriptors import feature_descriptor
from skimage import io
import cv2

class color_moments(feature_descriptor):

	def return_descriptor(self):
		# fileName = os.path.join(os.getcwd(), folderName, imageID)
		# image = io.imread(fileName)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		r = 0
		i = 0
		yChannel = np.empty((3, 192), float)
		uChannel = np.empty((3, 192), float)
		vChannel = np.empty((3, 192), float)
		while (r < image.shape[0]):
			c = 0
			while (c < image.shape[1]):
				block = image[r:r + 100, c:c + 100]
				yChannel[0][i] = np.mean(block[:, :, 0])
				uChannel[0][i] = np.mean(block[:, :, 1])
				vChannel[0][i] = np.mean(block[:, :, 2])
				yChannel[1][i] = np.std(block[:, :, 0])
				uChannel[1][i] = np.std(block[:, :, 1])
				vChannel[1][i] = np.std(block[:, :, 2])
				yChannel[2][i] = skew(block[:, :, 0], axis=None)
				uChannel[2][i] = skew(block[:, :, 1], axis=None)
				vChannel[2][i] = skew(block[:, :, 2], axis=None)
				c = c + 100
				i = i + 1
			r = r + 100
		return yChannel, uChannel, vChannel