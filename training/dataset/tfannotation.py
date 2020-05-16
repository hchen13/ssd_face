import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(image, label, delta):
    feature = {
        'image': _bytes_feature(tf.io.serialize_tensor(image)),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
        'delta': _bytes_feature(tf.io.serialize_tensor(delta))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string),
        'delta': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
    label = tf.io.parse_tensor(example['label'], out_type=tf.uint8)
    delta = tf.io.parse_tensor(example['delta'], out_type=tf.float64)
    return image, label, delta