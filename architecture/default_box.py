import numpy as np
import tensorflow as tf

from utils.helper_funcs import draw_boxes, show_image


class DefaultBox(tf.keras.layers.Layer):
    def __init__(self, scale, scale_next, cell_size, aspect_ratios=None, **kwargs):
        super(DefaultBox, self).__init__(**kwargs)
        self.aspect_ratios = [1., .5] if aspect_ratios is None else aspect_ratios
        self.n_boxes = len(self.aspect_ratios) + 1
        self.cell_size = cell_size
        self.compute_box_shape(scale, scale_next)

    def get_centroid(self, tensor_shape, image_shape):
        image_height = tf.cast(image_shape[1], tf.float64)
        image_width = tf.cast(image_shape[2], tf.float64)
        h = tf.cast(tensor_shape[1], tf.float64)
        w = tf.cast(tensor_shape[2], tf.float64)
        # cell_height = tf.cast(self.cell_size, tf.float64)
        # cell_width = tf.cast(self.cell_size, tf.float64)
        cell_height = image_height / h
        cell_width = image_width / w
        center_x = tf.linspace(cell_width / 2, (w - .5) * cell_width, tf.cast(w, tf.int32))
        center_y = tf.linspace(cell_height / 2, (h - .5) * cell_height, tf.cast(h, tf.int32))
        cx_grid, cy_grid = tf.meshgrid(center_x, center_y)
        cx_grid = tf.expand_dims(cx_grid, axis=-1)
        cy_grid = tf.expand_dims(cy_grid, axis=-1)

        box_center_x = tf.tile(cx_grid, (1, 1, self.n_boxes))
        box_center_y = tf.tile(cy_grid, (1, 1, self.n_boxes))
        return box_center_x, box_center_y

    def compute_box_shape(self, s0, s1):
        """default box width and height for all aspect ratios"""
        shape_list = []
        for ratio in self.aspect_ratios:
            if ratio == 1.:
                shape_list.append([s0, s0])
                shape_list.append([np.sqrt(s0 * s1), np.sqrt(s0 * s1)])
            else:
                width = s0 * np.sqrt(ratio)
                height = s0 / np.sqrt(ratio)
                shape_list.append([width, height])
        self.shape_tensor = tf.convert_to_tensor(np.array(shape_list))

    def call(self, inputs, **kwargs):
        input_tensor = inputs[0]
        image_shape = inputs[1]
        tensor_shape = tf.shape(input_tensor)
        h = tensor_shape[1]
        w = tensor_shape[2]
        # image_height = tf.cast(image_shape[1], tf.float64)
        # image_width = tf.cast(image_shape[2], tf.float64)
        box_center_x, box_center_y = self.get_centroid(tensor_shape, image_shape)
        box_center_x = tf.expand_dims(box_center_x, axis=-1)
        box_center_y = tf.expand_dims(box_center_y, axis=-1)

        s = self.shape_tensor * 300
        s = tf.expand_dims(s, axis=0)
        s = tf.expand_dims(s, axis=0)
        shape_tensor = tf.tile(s, (h, w, 1, 1))
        box_tensor = tf.concat([box_center_x, box_center_y, shape_tensor], axis=-1)
        batch_num = tf.shape(input_tensor)[0]
        box_tensor = tf.expand_dims(box_tensor, axis=0)
        box_tensor = tf.tile(box_tensor, (batch_num, 1, 1, 1, 1))
        return tf.reshape(box_tensor, self.compute_output_shape(tf.shape(input_tensor)))

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = [input_shape[i] for i in range(4)]
        return batch_size, height, width, self.n_boxes, 4


if __name__ == '__main__':
    s = 5
    image_tensor = tf.keras.layers.Input(shape=(None, None, 3))
    image_shape = tf.shape(image_tensor)

    input_tensor = tf.keras.layers.Input(shape=(4, 5, 4))
    db_layer = DefaultBox(0.2, 0.34)
    db = db_layer([input_tensor, image_shape])
    model = tf.keras.Model([image_tensor, input_tensor], db)

    x = np.random.uniform(0, 1., size=(1, 4, 5, 4))
    img = np.zeros(shape=(1, 400, 500, 3))
    y = model.predict([img, x])
    print("\n\ns={}, layer def. shape: {}".format(s, db.shape))
    print("output shape:", y.shape)

    for row in range(s):
        for col in range(s):
            boxes = y[0, row, col, :]
            d = draw_boxes(img[0], boxes, mode='centroid', color=(255, 255, 255))
            show_image(d, destroy=True)
