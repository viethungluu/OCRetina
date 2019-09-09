import warnings
import argparse
import sys
import os

import numpy as np
import scipy.optimize
import codecs

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from ..utils.compute_overlap import compute_overlap
from ..utils.anchors import generate_anchors, AnchorParameters, anchors_for_shape
from ..utils.image import compute_resize_scale
from ..utils.paint_text import paint_text

warnings.simplefilter("ignore")

SIZES = [32, 64, 128, 256, 512]
STRIDES = [8, 16, 32, 64, 128]
state = {'best_result': sys.maxsize}

def read_word_list(text_file):
    # monogram file is sorted by frequency in english speech
    word_list = []
    with codecs.open(text_file, mode='r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            word_list.append(word)
    return np.array(word_list)

def seed_numer(image_index):
    return image_index

def random_image_shape(image_index):
    np.random.seed(seed_numer(image_index))

    image_width, image_height = \
        np.random.randint(800 // 2, 1333 * 1.5),\
        np.random.randint(800 // 2, 1333 * 1.5)

    return image_width, image_height

def random_image_content(word_list, image_index):
    np.random.seed(seed_numer(image_index))

    return np.random.choice(word_list, size=np.random.randint(150 // 2, 150))

def load_data(word_list, image_index):
    """ Load image and annotations for an image_index.
    """
    # set seed number to fix the image shape and its text for image_index
    
    image_width, image_height   = random_image_shape(image_index)
    words_to_image              = random_image_content(image_index)

    image, annotation           = paint_text(words_to_image, image_width=image_width, image_height=image_height)

    return image, annotation

def calculate_config(values, ratio_count):
    split_point = int((ratio_count - 1) / 2)

    ratios = [1]
    for i in range(split_point):
        ratios.append(values[i])
        ratios.append(1 / values[i])

    scales = values[split_point:]

    return AnchorParameters(SIZES, STRIDES, ratios, scales)

def base_anchors_for_shape(pyramid_levels=None, anchor_params=None):
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        all_anchors = np.append(all_anchors, anchors, axis=0)

    return all_anchors

def average_overlap(values, entries, state, image_shape, mode='focal', ratio_count=3, include_stride=False):
    anchor_params = calculate_config(values, ratio_count)

    if include_stride:
        anchors = anchors_for_shape(image_shape, anchor_params=anchor_params)
    else:
        anchors = base_anchors_for_shape(anchor_params=anchor_params)

    overlap = compute_overlap(entries, anchors)
    max_overlap = np.amax(overlap, axis=1)
    not_matched = len(np.where(max_overlap < 0.5)[0])

    if mode == 'avg':
        result = 1 - np.average(max_overlap)
    elif mode == 'ce':
        result = np.average(-np.log(max_overlap))
    elif mode == 'focal':
        result = np.average(-(1 - max_overlap) ** 2 * np.log(max_overlap))
    else:
        raise Exception('Invalid mode.')

    if result < state['best_result']:
        state['best_result'] = result

        print('Current best anchor configuration')
        print(f'Ratios: {sorted(np.round(anchor_params.ratios, 3))}')
        print(f'Scales: {sorted(np.round(anchor_params.scales, 3))}')

        if include_stride:
            print(f'Average overlap: {np.round(np.average(max_overlap), 3)}')

        print(f'Number of labels that don\'t have any matching anchor: {not_matched}')
        print()

    return result, not_matched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize RetinaNet anchor configuration')
    parser.add_argument('--monogram-path',    help='Path to file containing monogram words for training')
    parser.add_argument('--max-string-len',   help='Maximum number of words in each image.', type=int, default=150)
    parser.add_argument('--scales', type=int, help='Number of scales.', default=3)
    parser.add_argument('--ratios', type=int, help='Number of ratios, has to be an odd number.', default=3)
    parser.add_argument('--include-stride', action='store_true',
                        help='Should stride of the anchors be taken into account. Setting this to false will give '
                             'more accurate results however it is much slower.')
    parser.add_argument('--objective', type=str, default='focal',
                        help='Function used to weight the difference between the target and proposed anchors. '
                             'Options: focal, avg, ce.')
    parser.add_argument('--popsize', type=int, default=15,
                        help='The total population size multiplier used by differential evolution.')
    parser.add_argument('--image-width',      help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-height',     help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--seed', type=int, help='Seed value to use for differential evolution.')
    args = parser.parse_args()

    if args.ratios % 2 != 1:
        raise Exception('The number of ratios has to be odd.')

    entries = np.zeros((0, 4))
    max_x = 0
    max_y = 0

    if args.seed:
        seed = np.random.RandomState(args.seed)
    else:
        seed = np.random.RandomState()
        
    word_list  = read_word_list(args.monogram_path)
    for i in range(4096):
        img, annotations = load_data(word_list, i)
        for anno in annotations['bboxes']:
            x1, y1, x2, y2 = list(map(lambda x: int(x), anno))

            if not x1 or not y1 or not x2 or not y2:
                continue

            scale = compute_resize_scale(img.shape, min_side=args.image_min_side, max_side=args.image_max_side)
            x1, y1, x2, y2 = list(map(lambda x: int(x) * scale, row[1:5]))

            max_x = max(x2, max_x)
            max_y = max(y2, max_y)

            if args.include_stride:
                entry = np.expand_dims(np.array([x1, y1, x2, y2]), axis=0)
                entries = np.append(entries, entry, axis=0)
            else:
                width = x2 - x1
                height = y2 - y1
                entry = np.expand_dims(np.array([-width / 2, -height / 2, width / 2, height / 2]), axis=0)
                entries = np.append(entries, entry, axis=0)

    image_shape = [max_y, max_x]

    print('Optimising anchors.')

    bounds = []
    best_result = sys.maxsize

    for i in range(int((args.ratios - 1) / 2)):
        bounds.append((1, 4))

    for i in range(args.scales):
        bounds.append((0.4, 2))

    result = scipy.optimize.differential_evolution(
        lambda x: average_overlap(x, entries, state, image_shape, args.objective, args.ratios, args.include_stride)[0],
        bounds=bounds, popsize=args.popsize, seed=seed)

    if hasattr(result, 'success') and result.success:
        print('Optimization ended successfully!')
    elif not hasattr(result, 'success'):
        print('Optimization ended!')
    else:
        print('Optimization ended unsuccessfully!')
        print(f'Reason: {result.message}')

    values = result.x
    anchor_params = calculate_config(values, args.ratios)
    (avg, not_matched) = average_overlap(values, entries, {'best_result': 0}, image_shape,
                                         'avg', args.ratios, args.include_stride)

    print()
    print('Final best anchor configuration')
    print(f'Ratios: {sorted(np.round(anchor_params.ratios, 3))}')
    print(f'Scales: {sorted(np.round(anchor_params.scales, 3))}')
    print(f'Average overlap: {1 - avg}')
    print(f'Number of labels that don\'t have any matching anchor: {not_matched}')
