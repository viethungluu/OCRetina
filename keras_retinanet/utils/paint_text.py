import os
import sys

import numpy as np
from scipy import ndimage

import cv2

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.utils"

from .. import params

# Translation of characters to unique integer values
def text_to_labels(text, max_word_len):
	ret = np.full(max_word_len, -1)
	for i, char in enumerate(text):
		ret[i] = params.ALPHABET.find(char)
	return ret

# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
	ret = []
	for c in labels:
		if c == len(params.ALPHABET):  # CTC Blank
			ret.append("")
		else:
			ret.append(params.ALPHABET[c])
	return "".join(ret)

"""
Source: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
"""
def speckle(img, scale=255):
	severity 	= np.random.uniform(0, 0.6)
	blur 		= ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
	blur 		*= scale
	
	img_speck 					= img + blur
	img_speck[img_speck > 255] 	= 255
	img_speck[img_speck <= 0] 	= 0
	
	return img_speck.astype(np.uint8)

def add_punctuation_and_space(paragraph):
	# put space character between words
	sep 		= [" "] * len(paragraph)
	paragraph   = list(sum(zip(paragraph, sep), ())[:-1])

	# randomly put punctuation in paragraph
	sep_indices = np.arange(1, len(paragraph), 2)
	# number of punctuation to put
	k 			= np.random.randint(0, len(paragraph) // 10)
	# punctuation to be added to paragraph
	puncs 		= np.random.choice(params.PUNCTUATION_LIST, size=k, replace=True)
	paragraph 	= np.insert(paragraph, np.random.choice(sep_indices, size=k), puncs)
	
	return paragraph

def paint_text(paragraph, image_width, image_height, max_word_len, font_scale=1, thickness=2, line_spacing=40, multi_fonts=False):
	annotations = {'labels': np.full((0, max_word_len), -1), 'bboxes': np.empty((0, 4))}

	# define a blank image
	image 		= np.full((image_height, image_width, 3), 255, dtype=np.uint8)

	# set font face and font size
	if multi_fonts:
		font_face 	= np.random.choice(params.FONT_LIST)
	else:
		font_face 	= cv2.FONT_HERSHEY_TRIPLEX

	offset_x, offset_y 	= np.random.randint(50, 100), np.random.randint(50, 100)
	
	x, y 				= offset_x, offset_y

	# add space & punctuation character between word
	paragraph = add_punctuation_and_space(paragraph)
	for word in paragraph:
		# get estimated word size
		(w, h), baseline 	= cv2.getTextSize(word, font_face, font_scale, thickness)

		# if reach end of page width, move to begining of next line
		if x + w >= image_width - offset_x:
			y 	+= line_spacing
			x 	= offset_x

		# if reach end of page height, stop drawing
		if y + h >= image_height - offset_y:
			break

		# draw word into image
		cv2.putText(image, word, (x, y), font_face, font_scale, (0, 0, 0), thickness)
		
		# calculate the bounding box corner of drawn word
		annotations['labels'] = np.concatenate((annotations['labels'], [text_to_labels(word, max_word_len)]))
		annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
			float(x),
			float(y - h),
			float(x + w),
			float(y + baseline)
		]]))

		# move to next word position
		x 	= x + w

	# add speckle noise to generated image
	image 	= speckle(image)

	return image, annotations

if __name__ == '__main__':
	test_phrase 	= ["to", "draw", "the", "ellipse", "we", "need", "to", "pass", "several", "arguments", "one", "argument", "is", "the", "center"]
	test_phrase 	*= np.random.randint(5, 10)
	
	text_phrase 	= np.array(test_phrase)

	image, annotations 	= paint_text(test_phrase, 800, 1333, 16)

	for label, (x1, y1, x2, y2) in zip(annotations['labels'], annotations['bboxes']):
		word = labels_to_text(label)

		cv2.putText(image, word, (int(x1), int(y1)), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)
		cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

	cv2.imshow('image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()