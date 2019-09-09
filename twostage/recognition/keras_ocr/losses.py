from keras import backend as K

def ctc():
	""" Create a functor for computing the CTC loss.

	Args
		y_true: batch x timestep x category
		y_pred: same shape as y_true

	Returns
	    A functor that computes the CTC loss.
	"""
	def _ctc(y_true, y_pred):
		labels 			= y_true[:, :-2, :] # (batch x max_word_length)
		input_length 	= y_true[:, -2, :]  # (batch x 1)
		label_length 	= y_true[:, -1, :]  # (batch x 1)
		
		y_pred 			= y_pred[:, 2:, :] # batch x time_step x num_categories

		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return _ctc

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
