import tensorflow as tf


@tf.function
def smooth_l1_loss(y_true, y_pred):
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = .5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss - .5)
    return tf.reduce_sum(l1_loss, axis=-1)


@tf.function
def compute_loss(y_true, y_pred):
    labels_true, offsets_true = y_true
    labels_pred, offsets_pred = y_pred
    positives = labels_true[:, :, 0]  # shape (batch, 5820)
    negatives = 1 - positives

    # the number of positive boxes across all of the batch
    num_positives = tf.reduce_sum(positives, axis=-1)  # shape (batch, )

    # compute classification and localization losses for each box in the batch
    class_loss = tf.losses.binary_crossentropy(labels_true, labels_pred)  # shape (batch, 5820)
    loc_loss = smooth_l1_loss(offsets_true, offsets_pred)  # shape (batch, 5820)
    loc_loss = tf.reduce_sum(loc_loss * positives, axis=-1)  # shape (batch, )

    # filter class_loss for positives and negatives using the masks above
    pos_class_loss = tf.reduce_sum(class_loss * positives, axis=-1)  # shape (batch, )
    neg_class_loss_all = class_loss * negatives  # shape (batch, 5820)

    # compute the number of non-zero loss values for each sample in the batch, then determine the number of
    # negative samples we want to account for in the loss, given that it's at most
    # 3 * # positive samples, but no more than the number of non-zero loss values.
    num_nonzero_neg_class_loss = tf.math.count_nonzero(neg_class_loss_all, axis=-1, dtype=tf.int32)  # shape (batch, )
    k = tf.minimum(tf.maximum(3 * tf.cast(num_positives, tf.int32), 1), num_nonzero_neg_class_loss)  # shape (batch, )

    neg_loss_rank = tf.argsort(neg_class_loss_all, axis=-1, direction='DESCENDING')
    neg_loss_rank = tf.argsort(neg_loss_rank, axis=-1)
    k_msk = tf.cast(neg_loss_rank < tf.expand_dims(k, axis=1), tf.float32)

    # create a matrix of shape (batch, 5820) as a mask to select the first 'k' largest values in
    # each row of `neg_class_loss_all_sorted` using `k`. the mask then should have 'k' 1s from the beginning,
    # followed by (5820 - k) 0s in each row.
    # For example:
    # k = [3, 5, 1], k_mask = [
    #  [1, 1, 1, 0, 0, 0, 0, ..., 0],
    #  [1, 1, 1, 1, 1, 0, 0, ..., 0],
    #  [1, 0, 0, 0, 0, 0, 0, ..., 0]
    # ]
    # neg_class_loss_all_sorted = tf.sort(neg_class_loss_all, axis=-1, direction='DESCENDING')  # shape (batch, 5820)
    # k_mask = tf.one_hot(k, 5820)
    # k_mask = tf.cumsum(k_mask, axis=-1)
    # k_mask = 1 - k_mask
    # neg_class_loss_top_k = tf.reduce_sum(neg_class_loss_all_sorted * k_mask, axis=-1)   # shape (batch, )

    top_k_neg_class_loss = tf.reduce_sum(neg_class_loss_all * k_msk, axis=-1)

    # now that we have classification loss values for both positive and negative boxes for each sample in the batch
    # we sum them up to get the overall classification loss for each sample in the batch
    class_loss = pos_class_loss + top_k_neg_class_loss
    classification_loss = tf.reduce_mean(class_loss / tf.maximum(1., num_positives))
    localization_loss = tf.reduce_mean(loc_loss / tf.maximum(1., num_positives))
    total_loss = classification_loss + localization_loss
    return total_loss, classification_loss, localization_loss
