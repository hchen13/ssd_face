import os
import random
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from dataset_management.tfannotation import read_tfrecord
from dataset_management.utils import augment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preprocess_input(x):
    return tf.cast(x, tf.float32) / 127.5 - 1.


def restore_boxes(default_boxes, offsets):
    box = np.empty_like(default_boxes)
    box[:, 0] = offsets[:, 0] * default_boxes[:, 2] + default_boxes[:, 0]
    box[:, 1] = offsets[:, 1] * default_boxes[:, 3] + default_boxes[:, 1]
    box[:, 2] = np.exp(offsets[:, 2]) * default_boxes[:, 2]
    box[:, 3] = np.exp(offsets[:, 3]) * default_boxes[:, 3]
    return box


def load_dataset_from_tfrecords(dataset_root, batch_size, shuffle=True):
    print("[info] preparing training datasets...", flush=True)
    train_dir = os.path.join(dataset_root, 'train')
    valid_dir = os.path.join(dataset_root, 'valid')
    train_images, valid_images = [glob(os.path.join(p, '*.tfrecords')) for p in [train_dir, valid_dir]]
    if shuffle:
        random.shuffle(train_images)
    raw = tf.data.TFRecordDataset(train_images)
    trainset = raw.map(read_tfrecord)
    raw = tf.data.TFRecordDataset(valid_images)
    validset = raw.map(read_tfrecord)
    trainset = augment(trainset).batch(batch_size).prefetch(AUTOTUNE)
    validset = validset.batch(batch_size).prefetch(AUTOTUNE)
    print(f'[info] datasets preparation complete, '
          f'training size: {len(train_images)}, validation size: {len(valid_images)}.\n'
          f'batch size: {batch_size}.\n', flush=True)
    return trainset, validset