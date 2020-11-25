import cv2
from ethan_toolbox import convert_box_coordinates, non_max_suppression
from tensorflow.python.keras.layers import Conv2D

from prototype.default_box import DefaultBox
import numpy as np


class BaseSSD:
    def create_head_layers(self):
        self.conf_layers = [
            Conv2D(self.num_boxes, 3, activation='sigmoid', padding='same', name='scores1'),
            Conv2D(self.num_boxes, 3, activation='sigmoid', padding='same', name='scores2'),
            Conv2D(self.num_boxes, 3, activation='sigmoid', padding='same', name='scores3'),
            Conv2D(self.num_boxes, 3, activation='sigmoid', padding='same', name='scores4'),
            Conv2D(self.num_boxes, 3, activation='sigmoid', padding='same', name='scores5'),
            Conv2D(self.num_boxes, 3, activation='sigmoid', padding='same', name='scores6'),
            Conv2D(self.num_boxes, 1, activation='sigmoid', padding='same', name='scores7'),
        ]
        n = self.num_boxes * 4
        self.loc_layers = [
            Conv2D(n, 3, padding='same', name='offsets1'),
            Conv2D(n, 3, padding='same', name='offsets2'),
            Conv2D(n, 3, padding='same', name='offsets3'),
            Conv2D(n, 3, padding='same', name='offsets4'),
            Conv2D(n, 3, padding='same', name='offsets5'),
            Conv2D(n, 3, padding='same', name='offsets6'),
            Conv2D(n, 1, padding='same', name='offsets7'),
        ]
        scales = [.07, .15, .3, .45, .6, .75, .9, 1.05]
        self.anchor_layers = []
        for i, s0 in enumerate(scales[:-1]):
            s1 = scales[i + 1]
            db = DefaultBox(s0=s0, s1=s1, aspect_ratios=self.aspect_ratios, name=f'default_box{i + 1}')
            self.anchor_layers.append(db)

    def __init__(self, aspect_ratios=None, image_size=None):
        self.model = None
        if aspect_ratios is None:
            aspect_ratios = [1.]
        self.aspect_ratios = aspect_ratios
        self.num_boxes = len(aspect_ratios) + 1 if 1. in aspect_ratios else 0
        self.create_head_layers()
        self.build(input_shape=(image_size, image_size, 3))

    def build(self, intput_shape):
        raise NotImplementedError

    def detect(self, image, threshold=.5):
        image = cv2.resize(image, (512, 512))
        input_image = image / 127.5 - 1.
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
        boxes, scores = non_max_suppression(scores, positive_boxes_minmax, method='min', threshold=.5)
        return boxes, scores

    def freeze_layers(self, stop_name):
        for layer in self.model.layers:
            layer.trainable = False
            if layer.name == stop_name:
                break

    def unfreeze_layers(self):
        for layer in self.model.layers:
            layer.trainable = True