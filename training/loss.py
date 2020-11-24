import tensorflow as tf


@tf.function
def hard_negative_mining(loss, gt_confs, neg_ratio=3):
    pos_idx = tf.squeeze(gt_confs > 0, axis=-1)                     # shape (batch, N)
    num_pos = tf.reduce_sum(tf.cast(pos_idx, tf.int32), axis=1)     # shape (batch,)
    num_neg = num_pos * neg_ratio                                   # shape (batch,)

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)                                 # shape (batch, N)
    neg_idx = rank < tf.expand_dims(num_neg, axis=-1)               # shape (batch, N)

    return pos_idx, neg_idx


def create_losses(neg_ratio):
    @tf.function
    def inner(confs, locs, gt_confs, gt_locs):
        cross_entropy = tf.losses.BinaryCrossentropy(reduction='none')
        temp_loss = cross_entropy(gt_confs, confs)                  # shape (batch, N)
        pos_idx, neg_idx = hard_negative_mining(temp_loss, gt_confs, neg_ratio)

        cross_entropy = tf.losses.BinaryCrossentropy(reduction='sum')
        smooth_l1_loss = tf.losses.Huber(reduction='sum')

        conf_loss = cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
            confs[tf.math.logical_or(pos_idx, neg_idx)]
        )
        loc_loss = smooth_l1_loss(gt_locs[pos_idx], locs[pos_idx])

        num_pos = tf.reduce_sum(tf.cast(pos_idx, tf.float32))
        conf_loss = conf_loss / num_pos
        loc_loss = loc_loss / num_pos
        return conf_loss, loc_loss
    return inner