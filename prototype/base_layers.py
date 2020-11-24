import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU


class ConvBn(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int=1, padding: str='valid', act: bool=True, **kwargs):
        super(ConvBn, self).__init__(**kwargs)
        self.conv_params = dict(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )
        self.act = act

    def build(self, input_shape):
        self.conv_layer = Conv2D(**self.conv_params)
        self.bn_layer = BatchNormalization(
            scale=False,
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        if self.act:
            self.relu_layer = ReLU()

    def call(self, input_tensor, **kwargs):
        x = self.conv_layer(input_tensor)
        x = self.bn_layer(x)
        if self.act:
            x = self.relu_layer(x)
        return x


class Linear(tf.keras.layers.Layer):
    """
    The layer implements a linear transformation that is applied upon the output of
    conv. layers, as specified in the PVANet paper.
    For each channel `c` in the conv output [height, width, channel] we apply a pair
    of different values (w, b) so that each channel will have a linear transformation
    of x' = wx + b
    """
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.w = self.add_weight(
            name='scale',
            shape=(channels, ),
            initializer=tf.initializers.Ones
        )
        self.b = self.add_weight(
            name='shift',
            shape=(channels, ),
            initializer=tf.initializers.Zeros
        )

    def call(self, x):
        return tf.add(tf.multiply(x, self.w), self.b)


def conv(**params):
    """
    The convolution layer used in this architecture, which always has same padding
    and does not use biases.
    """
    return Conv2D(padding='same', **params)