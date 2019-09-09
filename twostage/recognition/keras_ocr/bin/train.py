import os
import sys
import argparse

import keras
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import keras_ocr.bin  # noqa: F401
	__package__ = "keras_ocr.bin"

from ..preprocessing.generator import TextGenerator
from ..losses import ctc_lambda_func

def makedirs(path):
	# Intended behavior: try to create the directory,
	# pass if the directory exists already, fails otherwise.
	# Meant for Python 2.7/3.n compatibility.
	try:
		os.makedirs(path)
	except OSError:
		if not os.path.isdir(path):
			raise

def parse_args(args):
	""" Parse the arguments.
	"""
	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	group = parser.add_mutually_exclusive_group()
	group.add_argument('--snapshot',          help='Resume training from a snapshot.')

	parser.add_argument('--monogram-path',    help='Path to file containing monogram words for training')
	parser.add_argument('--max-word-len',     help='Maximum number of words in each image.', type=int, default=16)

	parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
	parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
	parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
	parser.add_argument('--downsample-factor',help='Size of the batches.', default=2, type=int)

	parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
	parser.add_argument('--image-width',   	  help='Rescale the image so the width is image-width.', type=int, default=128)
	parser.add_argument('--image-height',  	  help='Rescale the image so the width is image-width.', type=int, default=64)

	return parser.parse_args(args)

def build_model(args):
	conv_filters = 16
	kernel_size = (3, 3)
	pool_size = 2
	time_dense_size = 32
	rnn_size = 512
	minibatch_size = 32

	image_width  = args.image_width
	image_height = args.image_height

	# input is in W x H x C format
	if keras.backend.image_data_format() == 'channels_first':
		inputs  = Input(shape=(3, image_width, image_height))
	else:
		inputs  = Input(shape=(image_width, image_height, 3))

	outputs = inputs

	outputs = Conv2D(conv_filters, kernel_size, padding='same',
				activation='relu', kernel_initializer='he_normal',
				name='conv1')(outputs)
	outputs = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(outputs)
	outputs = Conv2D(conv_filters, kernel_size, padding='same',
				activation='relu', kernel_initializer='he_normal',
				name='conv2')(outputs)
	outputs = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(outputs)

	outputs = Reshape(target_shape=(image_width // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters),
					 name='reshape')(outputs)

	# cuts down input size going into RNN:
	outputs = Dense(time_dense_size, activation='relu', name='dense1')(outputs)

	# Two layers of bidirectional GRUs
	# GRU seems to work as well, if not better than LSTM:
	gru_1 	= GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(outputs)
	gru_1b 	= GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(outputs)
	gru1_merged = add([gru_1, gru_1b])
	gru_2 	= GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
	gru_2b 	= GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

	# transforms RNN output to character activations:
	outputs = Dense(26, kernel_initializer='he_normal', 
				name='dense2')(concatenate([gru_2, gru_2b]))
	
	y_pred 	= Activation('softmax', name='softmax')(outputs)

	labels 		 = Input(name='the_labels', shape=[args.max_word_len], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')
	
	# Keras doesn't currently support loss funcs with extra parameters
	# so CTC loss is implemented in a lambda layer
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

	# clipnorm seems to speeds up convergence
	sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

	model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

	# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

	return model

def main(args=None):
	# parse arguments
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	train_generator = TextGenerator(
		args.monogram_path,
		batch_size=args.batch_size,
		max_word_len=args.max_word_len,
		image_width=args.image_width,
		image_height=args.image_height,
		downsample_factor=args.downsample_factor,
	)

	model = build_model(args)

	checkpoint = keras.callbacks.ModelCheckpoint(
		os.path.join(
			args.snapshot_path,
			'recognition_{{epoch:02d}}.h5'
		),
		verbose=1,
	)

	return training_model.fit_generator(
		generator=train_generator,
		steps_per_epoch=len(train_generator) // args.batch_size,
		epochs=args.epochs,
		verbose=1,
		callbacks=[checkpoint]
	)
	

if __name__ == '__main__':
	main()