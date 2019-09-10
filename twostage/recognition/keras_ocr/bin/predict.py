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
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_ocr.bin"

from ..preprocessing.generator import TextGenerator
from .. import losses
from .. import params

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('--image-path',       help='Path for image need detections.')
    parser.add_argument('--detectionn-path',  help='Path for detected bounding box.')
    parser.add_argument('--weights-file',     help='Path to saved weights file.')
    parser.add_argument('--max-word-len',     help='Maximum number of words in each image.', type=int, default=16)
    parser.add_argument('--image-width',      help='Rescale the image so the width is image-width.', type=int, default=128)
    parser.add_argument('--image-height',     help='Rescale the image so the width is image-width.', type=int, default=64)
    
    return parser.parse_args(args)

def load_model(args):
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

    outputs = Reshape(target_shape=(image_width // (pool_size ** 2), (image_height // (pool_size ** 2)) * conv_filters),
                     name='reshape')(outputs)

    # cuts down input size going into RNN:
    outputs = Dense(time_dense_size, activation='relu', name='dense1')(outputs)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1   = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(outputs)
    gru_1b  = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(outputs)
    gru1_merged = add([gru_1, gru_1b])
    gru_2   = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b  = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    outputs = Dense(27, kernel_initializer='he_normal', 
                name='dense2')(concatenate([gru_2, gru_2b]))
    
    y_pred  = Activation('softmax', name='softmax')(outputs)

    Model(inputs=inputs, outputs=y_pred).summary()

    labels          = Input(name='the_labels', shape=[args.max_word_len], dtype='float32')
    input_length    = Input(name='input_length', shape=[1], dtype='int64')
    label_length    = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(losses.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.load_weights(args.weights_file)

    model_p = Model(inputs=inputs, outputs=y_pred)

    return model_p

def decode_predict_ctc(out, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                    greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        text = labels_to_text(lables)
        results.append(text)

    return results

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    model = load_model(args)

    image = cv2.imread(args.image_path)

    all_detections = []

    import csv
    with open(args.detectionn_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            all_detections.append(row[:4])

    all_detections = np.array(all_detections)
    # sort detection be y, x location
    indices = np.lexsort((all_detections[:, 1], all_detections[:, 0]))
    for index in indices:
        x1, y1, x2, y2 = all_detections[index]
        # preprocessing
        sub_image      = image[y1: y2, x1: x2, 3]
        sub_image      = cv2.resize(sub_image, dsize=(args.image_width, args.image_height))
        sub_image      = np.expand_dims(sub_image.T, axis=0)
    
        pred           = model_p.predict(sub_image)
        pred_texts     = decode_predict_ctc(pred, top_paths=1)
        print(pred_texts)

if __name__ == '__main__':
    main()