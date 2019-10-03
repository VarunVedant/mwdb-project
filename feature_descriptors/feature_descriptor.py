from abc import ABC, abstractmethod


class feat_desc(ABC):

	def __init__(self, img_id):
		self.img_id = img_id

	def return_descriptor(self):
		pass
