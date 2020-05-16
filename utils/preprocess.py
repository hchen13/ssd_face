import numpy as np


def preprocess_image(x):
    return x / 127.5 - 1


def deprocess_image(x):
    return np.floor((x + 1) * 127.5).astype('uint8')