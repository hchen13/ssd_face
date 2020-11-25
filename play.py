import os
import random
from datetime import datetime
from glob import glob
from random import shuffle

import cv2
import tensorflow as tf
import numpy as np
from ethan_toolbox import draw_boxes, show_image
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from dataset_management.tfannotation import read_tfrecord
from dataset_management.utils import augment
from prototype import pvanet, original
from training.tools import preprocess_input, restore_boxes, load_dataset_from_tfrecords, PROJECT_ROOT

if __name__ == '__main__':
    dataset_root = '/Users/ethan/datasets/WIDER_SSD/'
    batch_size = 4

    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    ssd = pvanet.SSD(aspect_ratios=aspect_ratios)
    # ssd.freeze_layers('conv2_3')
    ssd.model.load_weights('test.h5', by_name=True)
    # for layer in ssd.model.layers:
    #     layer.trainable = True
    # ssd.model.save_weights('test.h5')
