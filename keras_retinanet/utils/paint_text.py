import numpy as np
from scipy import ndimage

import cv2

FONT_LIST = [
	cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, 
	cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, 
	cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX
	]

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

"""
Source: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
"""
def paint_text(paragraph, image_width, image_height, font_scale=1, thickness=2, line_spacing=40, multi_fonts=False):
	annotations = []

	# define a blank image
	image 		= np.full((image_height, image_width, 3), 255, dtype=np.uint8)

	# set font face and font size
	if multi_fonts:
		font_face 	= np.random.choice(FONT_LIST)
	else:
		font_face 	= cv2.FONT_HERSHEY_TRIPLEX

	offset_x, offset_y 	= np.random.randint(20, 50), np.random.randint(20, 50)
	
	x, y 				= offset_x, offset_y
	for word in paragraph:
		# get estimated word size
		(w, h), baseline 	= cv2.getTextSize(word, font_face, font_scale, thickness)
		# h 					+= baseline

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
		annotations.append((x, y - h, x + w, y + baseline))

		# move to next word position
		x 	= x + w

	# add speckle noise to generated image
	image 	= speckle(image)

	return image, annotations

if __name__ == '__main__':
	test_phrase 	= ["To", " ", "draw", " ", "the", " ", "ellipse", ",", " ", "we", " ", "need", " ", "to", " ", "pass", " ", "several", " ", "arguments", ".", " ", "One", " ", "argument", " ", "is", " ", "the", " ", "center", " ", "location", ".", " "]
	test_phrase 	*= 10

	image, annotations 	= paint_text(test_phrase, 1600, 800)
	for (x1, y1, x2, y2) in annotations:
		cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

	cv2.imshow('image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


