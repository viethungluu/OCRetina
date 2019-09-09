from keras import backend as K

def ctc():
	""" Create a functor for computing the CTC loss.

	Args
		None

	Returns
	    A functor that computes the CTC loss.
	"""
	def _ctc(y_true, y_pred):
		labels 			= y_true[:, :-2] # (batch x max_word_length)
		input_length 	= y_true[:, -2] # (batch x 1)
		label_length 	= y_true[:, -1] # (batch x 1)
		
		y_pred 			= y_pred[:, 2:, :] # batch x time_step x num_categories

		print(K.int_shape(y_pred), K.int_shape(y_true))

		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	return _ctc