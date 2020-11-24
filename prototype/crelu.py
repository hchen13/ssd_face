import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization, ReLU, Concatenate

from prototype.base_layers import Linear, ConvBn, conv


class CRelu(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int, filters: int, strides: int=1, *args, **kwargs):
        super(CRelu, self).__init__(*args, **kwargs)
        self._params = dict(
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            use_bias=False,
        )

    def build(self, input_shape):
        self.conv_layer = conv(**self._params)
        self.bn_layer = BatchNormalization(scale=False)
        self.linear_layer = Linear()
        self.relu_layer = ReLU()
        self.concat_layer = Concatenate()

    def call(self, input_tensor):
        x = self.conv_layer(input_tensor)
        x = self.bn_layer(x)
        x = self.concat_layer([x, -x])
        x = self.linear_layer(x)
        x = self.relu_layer(x)
        return x


class ResidualCRelu(tf.keras.layers.Layer):
    PARAM_KEYS = ['kernel_size', 'f_in', 'f_out', 'f', 'strides', 'projection', 'bn_input']

    def _register_params(self, params):
        assert isinstance(params, dict)
        if 'bn_input' not in params.keys():
            params['bn_input'] = False
        for key in self.PARAM_KEYS:
            if key not in params.keys():
                raise KeyError(f"[ResidualCRelu] key '{key}' not found in params")
            setattr(self, key, params[key])

    def _parse_param_str(self, param_str):
        params = param_str.split(" ")
        filters = params[3].split('-')
        return dict(
            kernel_size=int(params[0]),
            strides=int(params[1]),
            projection=True if params[2] == 'PJ' else False,
            f_in=int(filters[0]),
            f=int(filters[1]),
            f_out=int(filters[2]),
            bn_input=True if params[4] == 'BN' else False,
        )

    def __init__(self, params, *args, **kwargs):
        """
        `param_str` follows the pattern of
        kernel_size, strides, projection, 1x1-kxk-1x1, bn_input
        sample:
        3 1 PJ 24-24-64 BN: kernel 3, strides 1, with projection, filters=24-24-64, with bn input
        3 2 NO 24-24-64 NO: kernel 3, strides 2, without projection, filters, no bn input
        """
        super(ResidualCRelu, self).__init__(*args, **kwargs)
        if isinstance(params, str):
            params = self._parse_param_str(params)
        self._register_params(params)


    def build(self, input_shape):
        self.conv_in = ConvBn(filters=self.f_in, kernel_size=1, padding='same', name='1x1conv_in')
        self.crelu_layer = CRelu(
            kernel_size=self.kernel_size,
            filters=self.f,
            strides=self.strides,
            name=f'{self.kernel_size}x{self.kernel_size}C.Relu'
        )
        self.conv_out = conv(filters=self.f_out, kernel_size=1, use_bias=False, name='1x1conv_out')
        if self.bn_input:
            self.bn_layer = BatchNormalization(scale=False)
            self.relu_layer = ReLU()
        if self.projection:
            self.proj_conv_layer = conv(
                filters=self.f_out, kernel_size=1, use_bias=False,
                strides=self.strides, name='1x1residual_projection')

    def call(self, input_tensor):
        x = input_tensor

        if self.bn_input:
            x = self.bn_layer(input_tensor)
            x = self.relu_layer(x)

        x = self.conv_in(x)
        x = self.crelu_layer(x)
        x = self.conv_out(x)
        if self.projection:
            input_tensor = self.proj_conv_layer(input_tensor)
        return x + input_tensor


