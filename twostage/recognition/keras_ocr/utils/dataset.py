import os
import sys

import numpy as np
from scipy import ndimage

import cv2
import itertools

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import keras_ocr.bin  # noqa: F401
	__package__ = "keras_ocr.utils"

from .. import params

def speckle(img, scale=255):
	severity 	= np.random.uniform(0, 0.6)
	blur 		= ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
	blur 		*= scale
	
	img_speck 					= img + blur
	img_speck[img_speck > 255] 	= 255
	img_speck[img_speck <= 0] 	= 0
	
	return img_speck.astype(np.uint8)

# Translation of characters to unique integer values
def text_to_labels(text, max_word_len):
	ret = np.full(max_word_len, -1, dtype=np.int)

	if text == "":
		ret[0] = len(params.ALPHABET)
		return ret, 1

	for i, char in enumerate(text):
		ret[i] = params.ALPHABET.find(char)

	return ret, i + 1

# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
	ret = []
	for c in labels:
		if c == len(params.ALPHABET):  # CTC Blank
			ret.append("")
		else:
			ret.append(params.ALPHABET[c])
	return "".join(ret)

def decode_batch(test_func, word_batch):
	out = test_func([word_batch])[0]
	ret = []
	for j in range(out.shape[0]):
		out_best = list(np.argmax(out[j, 2:], 1))
		out_best = [k for k, g in itertools.groupby(out_best)]
		outstr = labels_to_text(out_best)
		ret.append(outstr)
	return ret

def paint_text(word, image_width, image_height, max_word_len=16, font_scale=1, thickness=2, multi_fonts=False):
	# if the word is too long, it appear to small in the image
	if len(word) > max_word_len:
		word = word[:max_word_len]

	# set font face and font size
	if multi_fonts:
		font_face 	= np.random.choice(params.FONT_LIST)
	else:
		font_face 	= cv2.FONT_HERSHEY_TRIPLEX

	# get estimated word size
	(w, h), baseline 	= cv2.getTextSize(word, font_face, font_scale, thickness)

	# define a blank image
	offset 	= 5
	image   = np.full((h + baseline + offset * 2, w + offset * 2, 3), 255, dtype=np.uint8)
	
	# draw word into image
	# put the text at the center of image
	cv2.putText(image, word, (offset, h + offset), font_face, font_scale, (0, 0, 0), thickness)

	# add speckle noise to generated image
	image 	= speckle(image)

	# rescale image to fit image_width
	image = cv2.resize(image, dsize=(image_width, image_height))

	label, length = text_to_labels(word, max_word_len)
	annotations = {'labels': label, 'length': length}

	return image, annotations

if __name__ == '__main__':
	test_phrases 	= ["loveisblind"]

	image, annotations 	= paint_text(np.random.choice(test_phrases), 128, 64)
	print(labels_to_text(annotations["labels"]))

	image 			= image.transpose((1, 0, 2))
	# image 			= image[..., 0]
	# image 			= image.T

	import matplotlib.pyplot as plt
	plt.imshow(image)
	plt.show()

	# cv2.imshow('image', image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# cv2.imwrite("image.png", image)