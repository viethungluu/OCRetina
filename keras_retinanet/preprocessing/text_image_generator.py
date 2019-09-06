from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
from collections import OrderedDict

class TextImageGenerator(Generator):
	""" Online text image generator for text detection.
	"""

	def __init__(
		self,
		monogram_file, 
		bigram_file
		image_shape,
		dataset_size,
		**kwargs
	):
		self.monogram_file 	= monogram_file
		self.bigram_file 	= bigram_file
		
		self.image_shape 	= image_shape
		self.dataset_size 	= dataset_size

		self.classes 	= OrderedDict("word": 0)
		self.labels 	= {}
		for key, value in self.classes.items():
			self.labels[value] = key
		
		super(TextImageGenerator, self).__init__(**kwargs)

	def size(self):
		""" Size of the dataset.
		"""
		return self.dataset_size

	def num_classes(self):
		""" Number of classes in the dataset.
		"""
		return max(self.classes.values()) + 1

	def has_label(self, label):
		""" Return True if label is a known label.
		"""
		return label in self.labels

	def has_name(self, name):
		""" Returns True if name is a known class.
		"""
		return name in self.classes

	def name_to_label(self, name):
		""" Map name to label.
		"""
		return self.classes[name]

	def label_to_name(self, label):
		""" Map label to name.
		"""
		return self.labels[label]

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