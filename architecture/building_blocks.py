import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, BatchNormalization


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', bn=False, reg=True, **kwargs):
        conv_params = {
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides,
            'padding': padding,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': tf.keras.regularizers.l2(5e-4) if reg else None,
            **kwargs
        }
        kwargs.pop('dilation_rate', None)
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = Conv2D(**conv_params)
        self.bn = BatchNormalization() if bn else None

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(x)
        x = tf.keras.layers.LeakyReLU(alpha=.1)(x)
        return x


class SqueezeBlock(tf.keras.layers.Layer):
    def __init__(self, filters, bn=False, **kwargs):
        super(SqueezeBlock, self).__init__(**kwargs)
        self.filters = filters
        self.bn = bn

    def build(self, _):
        self.squeeze_conv = ConvBlock(self.filters, 1, bn=self.bn)
        self.expand_conv1 = ConvBlock(4 * self.filters, 1, bn=self.bn)
        self.expand_conv3 = ConvBlock(4 * self.filters, 3, bn=self.bn)

    def call(self, input_tensor, **kwargs):
        s = self.squeeze_conv(input_tensor)
        e1 = self.expand_conv1(s)
        e2 = self.expand_conv3(s)
        out_tensor = tf.keras.layers.Concatenate(axis=-1)([e1, e2])
        return out_tensor


if __name__ == '__main__':
    pass
