from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
from collections import OrderedDict

class TextGenerator(Generator):
	""" Online text image generator for text detection.
	"""
	def __init__(
		self,
		monogram_file, 
		image_shape=(1600, 800),
		dataset_size=10000,
		**kwargs
	):
		self.monogram_file 	= monogram_file
		
		self.image_shape 	= image_shape
		self.dataset_size 	= dataset_size
		
		super(TextGenerator, self).__init__(**kwargs)

	def size(self):
		""" Size of the dataset.
		"""
		return self.dataset_size

	def image_aspect_ratio(self, image_index):
		""" Compute the aspect ratio for an image with image_index.
		"""
		return float(self.image_shape[1]) / float(self.image_shape[0])

	def load_image(self, image_index):
		""" Load an image at the image_index.
		"""
		pass

	def load_annotations(self, image_index):
		""" Load annotations for an image_index.
		"""
		pass