import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape


def compute_centroids(map_shape, cell_size):
    h, w = map_shape
    cx = np.array([ (i + .5) * cell_size for i in range(w) ])
    cy = np.array([ (i + .5) * cell_size for i in range(h) ])
    centroids = np.empty(shape=(h, w, 2))
    for i in range(h):
        for j in range(w):
            centroids[i, j, 0] = cx[j]
            centroids[i, j, 1] = cy[i]
    return centroids.astype('float32')

def compute_shapes(s0, s1, aspect_ratios):
    shape_list = []
    for ratio in aspect_ratios:
        if ratio == 1.:
            shape_list.append([s0, s0])
            shape_list.append([np.sqrt(s0 * s1), np.sqrt(s0 * s1)])
        else:
            width = s0 * np.sqrt(ratio)
            height = s0 / np.sqrt(ratio)
            shape_list.append([width, height])
    return np.array(shape_list)

def create_default_boxes(map_shapes, cell_sizes, aspect_ratios):
    n_boxes = len(aspect_ratios) + 1
    default_boxes = []
    for l in range(len(map_shapes)):
        map_shape = map_shapes[l]
        cell_size = cell_sizes[l]
        image_size = cell_size * map_shape[0]
        centroids = compute_centroids(map_shape, cell_size)
        s0 = 0.2 + (0.9 - 0.2) * l / (len(map_shapes) - 1)
        s1 = 0.2 + (0.9 - 0.2) * (l + 1) / (len(map_shapes) - 1)
        shapes = compute_shapes(s0, s1, aspect_ratios)
        db = np.empty(shape=(map_shape[0], map_shape[1], n_boxes, 4))
        for i in range(n_boxes):
            db[:, :, i, :2] = centroids
            db[:, :, i, 2:] = shapes[i] * image_size
        default_boxes.append(db)
    return default_boxes


class DefaultBox(tf.keras.layers.Layer):
    def __init__(self, s0, s1, aspect_ratios=None, **kwargs):
        super(DefaultBox, self).__init__(**kwargs)
        if aspect_ratios is None:
            self.aspect_ratios = [1., ]
        else:
            self.aspect_ratios = aspect_ratios
        self.n_boxes = len(self.aspect_ratios) + 1
        self.compute_box_shape(s0, s1)

    def compute_box_shape(self, s0, s1):
        shape_list = []
        for ratio in self.aspect_ratios:
            if ratio == 1.:
                shape_list.append([s0, s0])
                s = np.sqrt(s0 * s1)
                shape_list.append([s, s])
            else:
                width = s0 * np.sqrt(ratio)
                height = s0 / np.sqrt(ratio)
                shape_list.append([width, height])
        self.shape_tensor = tf.convert_to_tensor(np.array(shape_list), dtype=tf.float32)

    def get_centroid(self, tensor_shape):
        h = tf.cast(tensor_shape[1], tf.float32)
        w = tf.cast(tensor_shape[2], tf.float32)
        cell_height = 1 / h
        cell_width = 1 / w
        cx = tf.linspace(cell_width / 2, (w - .5) * cell_width, tf.cast(w, tf.int32))
        cy = tf.linspace(cell_height / 2, (h - .5) * cell_height, tf.cast(h, tf.int32))
        cx_grid, cy_grid = tf.meshgrid(cx, cy)
        cx_grid = tf.expand_dims(cx_grid, axis=-1)
        cy_grid = tf.expand_dims(cy_grid, axis=-1)
        box_center_x = tf.tile(cx_grid, (1, 1, self.n_boxes))
        box_center_y = tf.tile(cy_grid, (1, 1, self.n_boxes))
        return box_center_x, box_center_y

    def call(self, input_tensor, **kwargs):
        tensor_shape = tf.shape(input_tensor)
        h, w = tensor_shape[1], tensor_shape[2]
        box_center_x, box_center_y = self.get_centroid(tensor_shape)
        box_center_x = tf.expand_dims(box_center_x, axis=-1)
        box_center_y = tf.expand_dims(box_center_y, axis=-1)
        s = tf.expand_dims(self.shape_tensor, axis=0)
        s = tf.expand_dims(s, axis=0)
        shape_tensor = tf.tile(s, (h, w, 1, 1))
        box_tensor = tf.concat([box_center_x, box_center_y, shape_tensor], axis=-1)
        batch_num = tf.shape(input_tensor)[0]
        box_tensor = tf.expand_dims(box_tensor, axis=0)
        box_tensor = tf.tile(box_tensor, (batch_num, 1, 1, 1, 1))
        box_tensor.set_shape(self.compute_output_shape(input_tensor.shape))
        return box_tensor

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_size, height, width, channels = [input_shape[i] for i in range(4)]
        output_shape = (batch_size, input_shape[1], input_shape[2], self.n_boxes, 4)
        return output_shape


if __name__ == '__main__':
    aspect_ratios = [1., ]

    """ USAGE #1: USE IN MODEL CREATION AS A LAYER """
    input_tensor = tf.keras.Input(shape=(None, None, 3))
    boxes = DefaultBox(s0=.2, s1=.34, aspect_ratios=aspect_ratios)(input_tensor)
    model = tf.keras.Model(input_tensor, boxes)

    """ USAGE #2: USE AS A PYTHON DEFAULT BOX FUNCTION """
    input_tensor = np.random.uniform(size=(1, 4, 4, 3)).astype('float32')
    boxes = DefaultBox(s0=.2, s1=.34, aspect_ratios=aspect_ratios)(input_tensor)
    boxes.shape  # 1, 4, 4, 2, 4
