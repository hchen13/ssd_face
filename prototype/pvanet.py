import os

import tensorflow as tf

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import MaxPooling2D, BatchNormalization, ReLU, Concatenate, Reshape

from prototype.base_layers import conv, ConvBn
from prototype.crelu import CRelu, ResidualCRelu
from prototype.inception import Inception
from prototype.ssd_base import BaseSSD
from training.tools import PROJECT_ROOT


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
        conv11 = CRelu(kernel_size=7, filters=16, strides=2, name='conv1_1')(input_tensor)
        pool11 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool1_1')(conv11)

        conv21 = ResidualCRelu(params="3 1 PJ 32-24-128 NO", name='conv2_1')(pool11)
        conv22 = ResidualCRelu(params="3 1 NO 32-24-128 BN", name='conv2_2')(conv21)
        conv23 = ResidualCRelu(params="3 1 NO 32-24-128 BN", name='conv2_3')(conv22)

        conv31 = ResidualCRelu(params="3 2 PJ 64-48-128 BN", name='conv3_1')(conv23)
        conv32 = ResidualCRelu(params="3 1 NO 64-48-128 BN", name='conv3_2')(conv31)
        conv33 = ResidualCRelu(params="3 1 PJ 64-48-192 BN", name='conv3_3')(conv32)
        conv34 = ResidualCRelu(params="3 1 NO 64-48-192 BN", name='conv3_4')(conv33)

        conv41 = Inception(params="2 PJ 64 64-128 32-48-48 256", name='conv4_1')(conv34)
        conv42 = Inception(params="1 NO 64 64-128 32-48-48 256", name='conv4_2')(conv41)
        conv43 = Inception(params="1 NO 64 64-128 32-48-48 256", name='conv4_3')(conv42)
        conv44 = Inception(params="1 NO 64 64-128 32-48-48 256", name='conv4_4')(conv43)

        conv51 = Inception(params="2 PJ 64 96-192 32-64-64 384", name='conv5_1')(conv44)
        conv52 = Inception(params="1 NO 64 96-192 32-64-64 384", name='conv5_2')(conv51)
        conv53 = Inception(params="1 NO 64 96-192 32-64-64 384", name='conv5_3')(conv52)
        conv54 = Inception(params="1 NO 64 96-192 32-64-64 384", name='conv5_4_pre')(conv53)
        conv54 = BatchNormalization(scale=False, name='conv5_4_bn')(conv54)
        conv54 = ReLU(name='conv5_4')(conv54)

        downscale = MaxPooling2D(pool_size=3, strides=2, padding='same', name='downscale')(conv34)
        upscale = tf.keras.layers.UpSampling2D(interpolation='bilinear', name='upscale')(conv54)
        concat = Concatenate(name='concat')([downscale, conv44, upscale])
        final = conv(filters=768, strides=1, kernel_size=1, activation='relu', name='pva_final')(concat)

        # extra feature map layers
        extra1 = ConvBn(256, 1, name='extra1_shrink')(final)
        extra1 = ConvBn(512, 3, strides=2, padding='same', name='extra1')(extra1)

        extra2 = ConvBn(128, 1, name='extra2_shrink')(extra1)
        extra2 = ConvBn(256, 3, strides=2, padding='same', name='extra2')(extra2)

        extra3 = ConvBn(128, 1, name='extra3_shrink')(extra2)
        extra3 = ConvBn(256, 3, name='extra3')(extra3)

        extra4 = ConvBn(128, 1, name='extra4_shrink')(extra3)
        extra4 = ConvBn(256, 3, name='extra4')(extra4)

        extra5 = ConvBn(128, 1, name='extra5_shrink')(extra4)
        extra5 = ConvBn(256, 4, name='extra5')(extra5)

        feature_maps = [conv34, final, extra1, extra2, extra3, extra4, extra5]
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

        self.model = Model(input_tensor, [conf_concat, loc_concat, anchor_concat], name='ssd_pvanet')

    def init_pvanet(self, path):
        self.model.load_weights(path, by_name=True)

if __name__ == '__main__':
    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    ssd = SSD(image_size=512, aspect_ratios=aspect_ratios)
    ssd.init_pvanet(os.path.join(PROJECT_ROOT, 'weights', 'pvanet_init.h5'))
    ssd.model.summary()