import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Reshape, Concatenate, Dropout
from tensorflow.python.keras.regularizers import l2

import architecture.backbones as bkb
from architecture.building_blocks import ConvBlock
from architecture.default_box import DefaultBox
from utils.helper_funcs import convert_box_coordinates, non_max_suppression, draw_boxes, show_image
from utils.preprocess import preprocess_image

supported_backbones = [name[4:] for name in dir(bkb) if name.startswith('bkb_')]


class SSD:
    aspect_ratios = [1., .5]
    n_boxes = len(aspect_ratios) + 1

    def __init__(self, backbone='vgg', input_shape=None):
        if backbone not in supported_backbones:
            raise TypeError("[SSD] unknown backbone architecture: \"{}\", \n\tsupported options: {}".format(
                backbone, ','.join(supported_backbones)))
        self.backbone = getattr(bkb, 'bkb_' + backbone)
        self._backbone_name = backbone
        self.feature_map_layers = 6
        self.scales = [.1 + (.9 - .1) / (self.feature_map_layers - 1) * i for i in range(self.feature_map_layers + 1)]
        self.build_model(input_shape)

    def save_model(self, file_name):
        folder = os.path.dirname(os.path.abspath(__file__))
        export_folder = os.path.join(os.path.dirname(folder), 'export')
        os.makedirs(export_folder, exist_ok=True)
        path = os.path.join(export_folder, file_name)
        self.model.save_weights(path)

    def load_model(self, file_name=None):
        if file_name is None:
            file_name = '{}_ssd.h5'.format(self._backbone_name)
        folder = os.path.dirname(os.path.abspath(__file__))
        export_folder = os.path.join(os.path.dirname(folder), 'export')
        path = os.path.join(export_folder, file_name)
        self.model.load_weights(path)

    def build_model(self, input_shape):
        image_tensor = Input(shape=[input_shape, input_shape, 3])
        image_shape = tf.shape(image_tensor)
        map0 = self.backbone(initial_filters=32, bn=False, name='feature_map0')(image_tensor)
        fc = ConvBlock(256, 3, reg=True, dilation_rate=6)(map0)
        # fc = ConvBlock(256, 3, reg=True)(map0)
        map1 = ConvBlock(256, 1, strides=2, name='feature_map1')(fc)

        _shrink, _expand = 32, 128
        map2 = ConvBlock(_shrink, 1, reg=True)(map1)
        map2 = ConvBlock(_expand, 3, strides=2, name='feature_map2')(map2)

        map3 = ConvBlock(_shrink, 1, reg=True)(map2)
        map3 = ConvBlock(_expand, 3, strides=2, name='feature_map3')(map3)

        map4 = ConvBlock(_shrink, 1, reg=True)(map3)
        map4 = ConvBlock(_expand, 3, strides=2, name='feature_map4')(map4)

        map5 = ConvBlock(_shrink, 1, reg=True)(map4)
        map5 = ConvBlock(_expand, 3, padding='valid', name='feature_map5')(map5)

        maps = [map0, map1, map2, map3, map4, map5]

        face_scores = []
        box_offsets = []
        anchors = []

        map_sizes = [38, 19, 10, 5, 3, 1]
        for i, feature_map in enumerate(maps):
            drop = Dropout(rate=.5)(feature_map)
            kernel_size = 3 if i != len(maps) - 1 else 1
            score = Conv2D(
                SSD.n_boxes, kernel_size,
                padding='same', activation='sigmoid',
                kernel_regularizer=l2(5e-4),
                name='MultiBox_face_score{}'.format(i))(drop)
            face_scores.append(score)

            offset = Conv2D(
                SSD.n_boxes * 4, kernel_size,
                kernel_regularizer=l2(5e-4),
                padding='same', name='MultiBox_offsets{}'.format(i))(drop)
            box_offsets.append(offset)

            anchor_box = DefaultBox(
                scale=self.scales[i],
                scale_next=self.scales[i + 1],
                cell_size=300 / map_sizes[i],
                name='MultiBox_anchors{}'.format(i))([feature_map, image_shape])

            anchors.append(anchor_box)

        scores_reshaped = [Reshape((-1, 1))(score) for score in face_scores]
        offsets_reshaped = [Reshape((-1, 4))(offset) for offset in box_offsets]
        anchors_reshaped = [Reshape((-1, 4))(anchor) for anchor in anchors]

        scores_concat = Concatenate(axis=1, name='face_scores')(scores_reshaped)
        offsets_concat = Concatenate(axis=1, name='face_offsets')(offsets_reshaped)
        anchors_concat = Concatenate(axis=1, name='face_anchors')(anchors_reshaped)

        self.model = Model(image_tensor, [scores_concat, offsets_concat, anchors_concat] + anchors)

    def detect_faces(self, image, threshold=.5):
        input_image = preprocess_image(image)
        if input_image.ndim != 4:
            input_image = np.expand_dims(input_image, axis=0)
        conf_pred, offsets_pred, anchors = self.model.predict(input_image)
        conf_pred = conf_pred.squeeze(axis=-1)
        conf_pred = conf_pred.squeeze(axis=0)
        offsets_pred = offsets_pred.squeeze(axis=0)
        anchors = anchors.squeeze(axis=0)
        box = np.empty_like(anchors)
        box[:, 0] = offsets_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
        box[:, 1] = offsets_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
        box[:, 2] = np.exp(offsets_pred[:, 2]) * anchors[:, 2]
        box[:, 3] = np.exp(offsets_pred[:, 3]) * anchors[:, 3]
        positive_indices = np.where(conf_pred > threshold)
        positive_boxes = box[positive_indices]
        scores = conf_pred[positive_indices]
        positive_boxes_minmax = convert_box_coordinates(positive_boxes, mode_from='centroid', mode_to='minmax')
        boxes, scores = non_max_suppression(scores, positive_boxes_minmax, method='min', threshold=.3)
        return boxes, scores


if __name__ == '__main__':
    input_image = np.zeros(shape=(1, 300, 300, 3))
    print("------- VGG backbone ---------")
    vgg = SSD(backbone='vgg', input_shape=300)
    scores, offsets, default_boxes = vgg.model.predict(input_image)
    print("input shape:", input_image.shape)
    print("score shape={}, offsets shape={}, anchors shape={}".format(scores.shape, offsets.shape, default_boxes.shape))

    # print("\n\n------- Squeeze backbone ---------")
    # squeeze = SSD(backbone='squeeze', input_shape=300)
    # scores, offsets, default_boxes = squeeze.model.predict(input_image)
    # print("input shape:", input_image.shape)
    # print("score shape={}, offsets shape={}, anchors shape={}".format(scores.shape, offsets.shape, default_boxes.shape))
    #
    # print(*vgg.model.trainable_weights, sep='\n')
