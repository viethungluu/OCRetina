"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import warnings
import codecs

import keras

from ..utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)
from ..utils.config import parse_anchor_parameters
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)

from ..utils.paint_text import paint_text
from ..utils.transform import transform_aabb
from .. import params

SEED_NUMBER = 28

class TextGenerator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        monogram_file,
        num_images=1024,
        paint_func=paint_text,
        max_string_len=150,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        transform_generator = None,
        visual_effect_generator=None,
        batch_size=4,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        config=None,
        classes=params.CLASSES,
        seed_offset=0
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.paint_func     = paint_func
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.num_images     = num_images
        self.max_string_len = max_string_len

        self.transform_generator     = transform_generator
        self.visual_effect_generator = visual_effect_generator
        self.batch_size              = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.transform_parameters    = transform_parameters or TransformParameters()
        self.compute_anchor_targets  = compute_anchor_targets
        self.compute_shapes          = compute_shapes
        self.preprocess_image        = preprocess_image
        self.config                  = config
        
        self.classes                 = classes
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.seed_offset = seed_offset

        self.word_list               = self.read_word_list(monogram_file)
        print("Number of words", len(self.word_list))

        # Define groups
        self.group_images()

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def read_word_list(self, text_file):
        # monogram file is sorted by frequency in english speech
        word_list = []
        with codecs.open(text_file, mode='r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                word_list.append(word)
        return np.array(word_list)

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def on_epoch_end(self):
        """ Shuffle the dataset
        """
        if self.shuffle_groups:
            np.random.shuffle(self.groups)

    # -----------------------------------------
    def size(self):
        """ Size of the dataset.
        """
        return self.num_images

    def seed_numer(self, image_index):
        return image_index + self.seed_offset

    def _random_image_shape(self, image_index):
        np.random.seed(self.seed_numer(image_index))

        image_width, image_height = \
            np.random.randint(self.image_min_side // 2, self.image_max_side * 1.5),\
            np.random.randint(self.image_min_side // 2, self.image_max_side * 1.5)

        return image_width, image_height

    def _random_image_content(self, image_index):
        np.random.seed(self.seed_numer(image_index))
        return np.random.choice(self.word_list, size=np.random.randint(self.max_string_len // 2, self.max_string_len))

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image_width, image_height = self._random_image_shape(image_index)
        return float(image_width) / image_height

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        image, _ = self.load_data(image_index)
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        _, annotation = self.load_data(image_index)
        return annotation

    def load_data(self, image_index):
        """ Load image and annotations for an image_index.
        """
        # set seed number to fix the image shape and its text for image_index
        
        image_width, image_height   = self._random_image_shape(image_index)
        words_to_image              = self._random_image_content(image_index)

        image, annotation           = self.paint_func(words_to_image, 
                                        image_width=image_width, 
                                        image_height=image_height,
                                        multi_fonts=True)
        
        return image, annotation

    # --------------------------------------------
    def load_data_group(self, group):
        """ Load image and annotations for all data in group.
        """
        result = [self.load_data(image_index) for image_index in group]
        image_group, annotations_group = zip(*result)
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
        return image_group, annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def random_visual_effect_group_entry(self, image, annotations):
        """ Randomly transforms image and annotation.
        """
        visual_effect = next(self.visual_effect_generator)
        # apply visual effect
        image = visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, image_group, annotations_group):
        """ Randomly apply visual effect on each image.
        """
        assert(len(image_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def compute_inputs(self, image_group, num_anchors):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group, annotations_group  = self.load_data_group(group)
        image_group, annotations_group  = list(image_group), list(annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        # compute network inputs
        inputs  = self.compute_inputs(image_group, targets[0].shape[1])

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """
        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        
        return self.compute_input_output(group)