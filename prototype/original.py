import cv2
from ethan_toolbox import convert_box_coordinates, non_max_suppression
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Reshape, Concatenate

from prototype.base_layers import ConvBn
from prototype.default_box import DefaultBox
import numpy as np

from prototype.ssd_base import BaseSSD


class SSD(BaseSSD):

    def __init__(self, aspect_ratios=None, image_size=None):
        self.model = None
        if aspect_ratios is None:
            aspect_ratios = [1.]
        self.aspect_ratios = aspect_ratios
        self.num_boxes = len(aspect_ratios) + 1 if 1. in aspect_ratios else 0
        self.create_head_layers()
        self.build(input_shape=(image_size, image_size, 3))

    def build(self, input_shape):
        input_tensor = Input(shape=input_shape)
        # vgg block 1
        x = Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')(input_tensor)
        x = Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')(x)
        x = MaxPooling2D(2, 2, padding='same', name='block1_pool')(x)

        # vgg block 2
        x = Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')(x)
        x = Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')(x)
        x = MaxPooling2D(2, 2, padding='same', name='block2_pool')(x)

        # vgg block 3
        x = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')(x)
        x = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')(x)
        x = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv3')(x)
        x = MaxPooling2D(2, 2, padding='same', name='block3_pool')(x)

        # vgg block 4
        x = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')(x)
        conv43 = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv3')(x)
        x = MaxPooling2D(2, 2, padding='same', name='block4_pool')(conv43)

        # vgg block 5
        x = Conv2D(512, 3, padding='same', activation='relu', name='block5_conv1')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', name='block5_conv2')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', name='block5_conv3')(x)
        x = MaxPooling2D(3, 1, padding='same', name='block5_pool')(x)

        # vgg fc modified as conv
        conv6 = Conv2D(1024, 3, dilation_rate=6, activation='relu', padding='same', name='conv6')(x)
        conv7 = Conv2D(1024, 1, activation='relu', padding='same', name='conv7')(conv6)

        # extra feature map layers
        extra1 = ConvBn(256, 1, name='extra1_shrink')(conv7)
        extra1 = ConvBn(512, 3, strides=2, padding='same', name='extra1')(extra1)

        extra2 = ConvBn(128, 1, name='extra2_shrink')(extra1)
        extra2 = ConvBn(256, 3, strides=2, padding='same', name='extra2')(extra2)

        extra3 = ConvBn(128, 1, name='extra3_shrink')(extra2)
        extra3 = ConvBn(256, 3, name='extra3')(extra3)

        extra4 = ConvBn(128, 1, name='extra4_shrink')(extra3)
        extra4 = ConvBn(256, 3, name='extra4')(extra4)

        extra5 = ConvBn(128, 1, name='extra5_shrink')(extra4)
        extra5 = ConvBn(256, 4, name='extra5')(extra5)

        # heads
        feature_maps = [conv43, conv7, extra1, extra2, extra3, extra4, extra5]
        confs, locs, anchors = [], [], []
        for i in range(len(feature_maps)):
            map = feature_maps[i]
            conf = self.conf_layers[i](map)
            loc = self.loc_layers[i](map)
            anchor = self.anchor_layers[i](map)
            confs.append(conf)
            locs.append(loc)
            anchors.append(anchor)
        confs_reshaped = [Reshape((-1, 1))(conf) for conf in confs]
        locs_reshaped = [Reshape((-1, 4))(loc) for loc in locs]
        anchors_reshaped = [Reshape((-1, 4))(db) for db in anchors]

        conf_concat = Concatenate(axis=1, name='scores')(confs_reshaped)
        loc_concat = Concatenate(axis=1, name='offsets')(locs_reshaped)
        anchor_concat = Concatenate(axis=1, name='default_boxes')(anchors_reshaped)

        self.model = Model(input_tensor, [conf_concat, loc_concat, anchor_concat], name='ssd_vgg16')

    def init_vgg16(self):
        vgg16 = VGG16(weights='imagenet', include_top=True)
        for i, layer in enumerate(self.model.layers):
            layer_name = layer.name
            if layer_name.startswith('block'):
                layer_weights = vgg16.get_layer(name=layer_name).get_weights()
                layer.set_weights(layer_weights)
        fc1_w, fc1_b = vgg16.get_layer(name='fc1').get_weights()
        fc2_w, fc2_b = vgg16.get_layer(name='fc2').get_weights()
        conv6_w = np.random.choice(
            np.reshape(fc1_w, (-1, )),
            (3, 3, 512, 1024)
        )
        conv6_b = np.random.choice(fc1_b, (1024,))
        conv7_w = np.random.choice(
            np.reshape(fc2_w, (-1,)),
            (1, 1, 1024, 1024)
        )
        conv7_b = np.random.choice(fc2_b, (1024,))
        self.model.get_layer(name='conv6').set_weights([conv6_w, conv6_b])
        self.model.get_layer(name='conv7').set_weights([conv7_w, conv7_b])

    # def detect(self, image, threshold=.5):
    #     input_image = image / 127.5 - 1.
    #     if input_image.ndim != 4:
    #         input_image = np.expand_dims(input_image, axis=0)
    #     conf_pred, offsets_pred, anchors = self.model.predict(input_image)
    #     conf_pred = conf_pred.squeeze(axis=-1)
    #     conf_pred = conf_pred.squeeze(axis=0)
    #     offsets_pred = offsets_pred.squeeze(axis=0)
    #     anchors = anchors.squeeze(axis=0)
    #     box = np.empty_like(anchors)
    #     box[:, 0] = offsets_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
    #     box[:, 1] = offsets_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
    #     box[:, 2] = np.exp(offsets_pred[:, 2]) * anchors[:, 2]
    #     box[:, 3] = np.exp(offsets_pred[:, 3]) * anchors[:, 3]
    #     positive_indices = np.where(conf_pred > threshold)
    #     positive_boxes = box[positive_indices]
    #     scores = conf_pred[positive_indices]
    #     positive_boxes_minmax = convert_box_coordinates(positive_boxes, mode_from='centroid', mode_to='minmax')
    #     boxes, scores = non_max_suppression(scores, positive_boxes_minmax, method='min', threshold=.5)
    #     return boxes, scores


if __name__ == '__main__':
    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    ssd = SSD(image_size=512, aspect_ratios=aspect_ratios)
    # ssd.init_vgg16()
    # ssd.freeze_layers('block4_pool')
    ssd.model.summary()