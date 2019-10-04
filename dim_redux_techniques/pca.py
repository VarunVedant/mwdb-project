from sklearn.decomposition import PCA

class pca:

	def __init__(self, feat_desc, k):
		self.matrix = feat_desc
		self.k = k

	def