import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage

"""
Source: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
"""
def speckle(img):
	severity 	= np.random.uniform(0, 0.6)
	blur 		= ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
	img_speck 	= (img + blur)
	img_speck[img_speck > 1] 	= 1
	img_speck[img_speck <= 0] 	= 0
	return img_speck

"""
Source: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
"""
def paint_text(paragraph, w, h, ud=False, multi_fonts=False):
	surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
	with cairo.Context(surface) as context:
		context.set_source_rgb(1, 1, 1)  # White
		context.paint()

		# set font face and font size
		if multi_fonts:
			fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono']
			context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
							np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
		else:
			context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
		context.set_font_size(25)

		# check if paragraph fit to the image
		box 		= context.text_extents(paragraph)
		border_w_h 	= (4, 4)
		if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
			raise IOError('Could not fit string into image. Max char count is too large for given image size.')

		# teach the RNN translational invariance by
		# fitting paragraph box randomly on canvas, with some room to rotate
		# move the context origin to random offset
		max_shift_x 	= w - box[2] - border_w_h[0]
		max_shift_y 	= h - box[3] - border_w_h[1]
		top_left_x 		= np.random.randint(0, int(max_shift_x))
		if ud:
			top_left_y = np.random.randint(0, int(max_shift_y))
		else:
			top_left_y = h // 2
		
		context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
		
		# start drawing the text on the image
		context.set_source_rgb(0, 0, 0)
		for text in paragraph:
			x1, y1 = context.get_current_point()
			context.show_text(text)
			x2, y2 = context.get_current_point()

	buf 	= surface.get_data()
	a 		= np.frombuffer(buf, np.uint8)
	a.shape = (h, w, 4)
	a 		= a[:, :, 0]  # grab single channel
	a 		= a.astype(np.float32) / 255
	a 		= np.expand_dims(a, 0)
	# add speckle noise to generated image
	a 		= speckle(a)

	return a

if __name__ == '__main__':
	main()