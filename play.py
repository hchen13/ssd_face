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
    m1 = pvanet.SSD(aspect_ratios=aspect_ratios)
    m2 = original.SSD(aspect_ratios=aspect_ratios)

    image = cv2.imread('/Users/ethan/Pictures/things/face_test2.jpg')
    m1.detect(image)
    m2.detect(image)

    def get_size(f):
        f.seek(0, 2)
        return f.tell()
    f1 = open('weights/temp/tmp_e1.h5')
    f2 = open('weights/ssd_vgg16.h5')

    print(f'image size: {image.shape}')
    n = 10
    tick = datetime.now()
    for _ in range(n):
        d = m1.detect(image)
    tock = datetime.now()
    print(f"PVANet weight file size: {get_size(f1) / 1024 / 1024:.2f}MB")
    print(f"PVANet SSD inference time: {(tock - tick).total_seconds() / n:.2f} seconds\n")

    tick = datetime.now()
    for _ in range(n):
        d = m2.detect(image)
    tock = datetime.now()
    print(f"original weight file size: {get_size(f2) / 1024 / 1024:.2f}MB")
    print(f"original VGG-based SSD inference time: {(tock - tick).total_seconds() / n:.2f} seconds\n")
    f1.close()
    f2.close()
    # ssd.init_pvanet(os.path.join(PROJECT_ROOT, 'weights', 'pvanet_init.h5'))
    # ssd.model.save_weights('weights/temp/test.h5')
    # ssd.model.load_weights('weights/temp/test.h5', by_name=True)