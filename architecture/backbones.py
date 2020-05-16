from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, Concatenate, PReLU

from architecture.building_blocks import ConvBlock, SqueezeBlock


def bkb_vgg(initial_filters=16, bn=False, name='bottleneck'):

    def inner(input_tensor):
        # block 1
        factor = 1
        x = ConvBlock(initial_filters, 3, bn=bn)(input_tensor)
        x = ConvBlock(initial_filters, 3, bn=bn)(x)
        x = MaxPooling2D([2, 2], strides=[2, 2], name='block1_pool')(x)

        # block 2
        factor *= 2
        x = ConvBlock(initial_filters * factor, 3, bn=bn)(x)
        x = ConvBlock(initial_filters * factor, 3, bn=bn)(x)
        x = MaxPooling2D([2, 2], strides=[2, 2], name='block2_pool')(x)

        # block 3
        factor *= 2
        x = ConvBlock(initial_filters * factor, 3, bn=bn)(x)
        x = ConvBlock(initial_filters * factor, 3, bn=bn)(x)
        x = ConvBlock(initial_filters * factor, 3, bn=bn)(x)
        x = MaxPooling2D([2, 2], strides=[2, 2], padding='same', name='block3_pool')(x)

        # block 4
        factor *= 2
        x = ConvBlock(initial_filters * factor, 3, bn=bn)(x)
        x = ConvBlock(initial_filters * factor, 3, bn=bn)(x)
        x = ConvBlock(initial_filters * factor, 3, bn=bn, name=name)(x)
        # x = MaxPooling2D([2, 2], strides=[2, 2], name='block4_pool')(x)

        return x

    return inner


def bkb_squeeze(bn=False, **kwargs):
    def inner(input_tensor):
        x = ConvBlock(60, 5)(input_tensor)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = SqueezeBlock(16, bn=bn)(x)
        x = SqueezeBlock(16, bn=bn)(x)
        x = SqueezeBlock(16, bn=bn)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = SqueezeBlock(24, bn=bn)(x)
        x = SqueezeBlock(24, bn=bn)(x)
        x = SqueezeBlock(32, bn=bn)(x)
        x = SqueezeBlock(32, bn=bn)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = SqueezeBlock(32, bn=bn)(x)
        return x
    return inner
