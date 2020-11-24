import tensorflow as tf
from tensorflow.python.keras.layers import MaxPooling2D, BatchNormalization, ReLU, Concatenate

from prototype.base_layers import ConvBn, conv

class Inception(tf.keras.layers.Layer):
    PARAM_KEYS = ['f1', 'f3', 'f5', 'f_out', 'projection', 'strides']

    def _register_params(self, params):
        assert isinstance(params, dict)
        if 'strides' not in params.keys():
            params['strides'] = 1
        for key in self.PARAM_KEYS:
            if key not in params.keys():
                raise KeyError(f'[Inception] key `{key}` not found in params.')
            if key == 'f3' and len(params[key]) != 2:
                raise KeyError(f'[Inception] invalid length for key `{key}`.')
            if key == 'f5' and len(params[key]) != 3:
                raise KeyError(f'[Inception] invalid length for key `{key}`.')
            setattr(self, key, params[key])

    def _parse_param_str(self, param_str):
        params = param_str.split(" ")
        f3 = list(map(int, params[3].split('-')))
        f5 = list(map(int, params[4].split("-")))
        return dict(
            strides=int(params[0]),
            projection=True if params[1] == 'PJ' else False,
            f1=int(params[2]),
            f3=f3,
            f5=f5,
            f_out=int(params[5])
        )

    def __init__(self, params, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.f_pool = 128
        if isinstance(params, str):
            params = self._parse_param_str(params)
        self._register_params(params)

    def build(self, input_shape):
        self.bn_layer = BatchNormalization(scale=False)
        self.relu_layer = ReLU()
        self.concat_layer = Concatenate()
        self.conv1 = ConvBn(filters=self.f1, kernel_size=1, strides=self.strides, padding='same', name='incep1')
        self.conv3 = [
            ConvBn(filters=self.f3[0], kernel_size=1, strides=self.strides, padding='same', name='incep2_1'),
            ConvBn(filters=self.f3[1], kernel_size=3, padding='same', name='incep2_2')
        ]
        self.conv5 = [
            ConvBn(filters=self.f5[0], kernel_size=1, padding='same', strides=self.strides, name='incep3_1'),
            ConvBn(filters=self.f5[1], kernel_size=3, padding='same', name='incep3_2'),
            ConvBn(filters=self.f5[2], kernel_size=3, padding='same', name='incep3_3')
        ]
        if self.strides == 2:
            self.pool_layers = [
                MaxPooling2D(pool_size=3, strides=2, padding='same', name='incep4_pool'),
                ConvBn(filters=self.f_pool, kernel_size=1, padding='same', name='incep4_conv')
            ]
        self.conv_out = conv(filters=self.f_out, kernel_size=1, use_bias=False, name='out')
        if self.projection:
            self.proj_conv = conv(
                filters=self.f_out, kernel_size=1,
                strides=self.strides, use_bias=False,
                name='projection')

    def call(self, input_tensor):
        # the bn and relu layers are used because all input tensors are coming out
        # of conv layers instead of ConvBn.
        layer_in = self.bn_layer(input_tensor)
        layer_in = self.relu_layer(layer_in)

        branch_outputs = []
        branch_outputs.append(self.conv1(layer_in))

        x = layer_in
        for layer in self.conv3:
            x = layer(x)
        branch_outputs.append(x)

        x = layer_in
        for layer in self.conv5:
            x = layer(x)
        branch_outputs.append(x)

        if self.strides == 2:
            x = layer_in
            for layer in self.pool_layers:
                x = layer(x)
            branch_outputs.append(x)

        concat = self.concat_layer(branch_outputs)
        x = self.conv_out(concat)
        if self.projection:
            input_tensor = self.proj_conv(input_tensor)
        return x + input_tensor


if __name__ == '__main__':
    import numpy as np
    Inception(params=dict(
        f1=64, f3=[48, 128], f5=[24, 46, 48], f_out=256,
        projection=True, strides=2,
    ))(np.random.normal(size=(1, 132, 80, 128)).astype('float32'))