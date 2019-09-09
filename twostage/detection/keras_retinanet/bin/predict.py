import numpy as np
import argparse

import keras

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from .. import models
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.image import resize_image, preprocess_image
from .. import params

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('--model',              help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


class RetinaNetWrapper(object):
    """docstring for RetinaNetWrapper"""
    def __init__(self, 
                model, 
                convert_model, 
                backbone,
                anchor_params  = None, 
                score_threshold=0.05,
                max_detections =2000,
                image_min_side =800,
                image_max_side =1333):
        super(RetinaNetWrapper, self).__init__()
        
        # load the model
        print('Loading model, this may take a second...')
        model = models.load_model(model, backbone_name=backbone)
        # optionally convert the model
        if convert_model:
            model = models.convert_model(model, anchor_params=anchor_params)

        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.image_min_side  = image_min_side
        self.image_max_side  = image_max_side
        self.num_classes     = max(params.CLASSES.values()) + 1

    def predict(self, raw_image):
        all_detections = [None for i in range(self.num_classes)]

        image        = preprocess_image(raw_image.copy())
        image, scale = resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
        for label in range(self.num_classes):
            all_detections[label] = image_detections[image_detections[:, -1] == label, :-1]

        return all_detections

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    model = RetinaNetWrapper(args.model, args.convert_model, args.backbone,
                             anchor_params   = anchor_params, 
                             score_threshold = args.score_threshold,
                             max_detections  = args.max_detections,
                             image_min_side  = args.image_min_side,
                             image_max_side  = args.image_max_side)

if __name__ == '__main__':
    main()