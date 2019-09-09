import numpy as np
import warnings
import codecs

import keras

from ..utils.image import (
    preprocess_image,
    resize_image,
)
from ..utils.paint_text import paint_text

SEED_NUMBER = 28

class TextGenerator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        word_file,
        paint_func=paint_text,
        batch_size=128,
        max_word_len=32,
        image_width=128,
        preprocess_image=preprocess_image,
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
        self.paint_func         = paint_func
        self.preprocess_image   = preprocess_image

        self.image_width    = image_width

        self.max_word_len   = max_word_len

        self.batch_size     = int(batch_size)
        
        self.word_list      = self.read_word_list(word_file)
        print("Number of words", len(self.word_list))

        # Define groups
        self.group_images()

    def read_word_list(self, word_file):
        # monogram file is sorted by frequency in english speech
        word_list = []
        with codecs.open(word_file, mode='r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                word_list.append(word)
        return np.array(word_list)

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        order = list(range(self.size()))
        order.sort(key=lambda x: self.image_aspect_ratio(x))
        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def on_epoch_end(self):
        """ Shuffle the dataset
        """
        self.group_images()

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        return len(self.word_list[image_index])

    def size(self):
        """ Size of the dataset.
        """
        return len(self.word_list)

    def load_data(self, image_index):
        """ Load image and annotations for an image_index.
        """
        image, annotation = self.paint_func(self.word_list[image_index], 
                                            image_width=self.image_width, 
                                            max_word_len=self.max_word_len)
        
        return image, annotation

    # --------------------------------------------
    def load_data_group(self, group):
        """ Load image and annotations for all data in group.
        """
        result = [self.load_data(image_index) for image_index in group]
        image_group, annotations_group = zip(*result)
        return image_group, annotations_group

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, image_width=self.image_width)

    def preprocess_group_entry(self, image):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index] = self.preprocess_group_entry(image_group[index])

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
            self.max_word_len
        )

        return list(batches)

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group, annotations_group  = self.load_data_group(group)
        image_group, annotations_group  = list(image_group), list(annotations_group)

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