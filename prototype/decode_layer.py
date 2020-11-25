import tensorflow as tf

class Decode(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decode, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        assert len(inputs) == 4
        confs, offsets, anchors, threshold = inputs
        confs = confs[:, 0]
        cx = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
        cy = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
        w = tf.exp(offsets[:, 2]) * anchors[:, 2]
        h = tf.exp(offsets[:, 3]) * anchors[:, 3]
        xmin = cx - .5 * w
        ymin = cy - .5 * h
        xmax = cx + .5 * w
        ymax = cy + .5 * h
        box = tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))
        positive_indices = confs > threshold
        positive_boxes = tf.boolean_mask(box, positive_indices)
        scores = tf.boolean_mask(confs, positive_indices)
        ids = tf.image.non_max_suppression(positive_boxes, scores, 400)
        boxes = tf.gather(positive_boxes, ids)
        scores = tf.gather(scores, ids)
        return boxes, scores