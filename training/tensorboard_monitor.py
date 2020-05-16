import os
import shutil
import tensorflow as tf


class Monitor:
    def __init__(self, caption):
        log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        fullpath = os.path.join(log_root, caption)
        try:
            shutil.rmtree(fullpath)
        except FileNotFoundError:
            pass
        os.makedirs(fullpath, exist_ok=True)
        self.logdir = fullpath
        self.caption = caption
        train_path = os.path.join(fullpath, 'train')
        valid_path = os.path.join(fullpath, 'valid')
        self.train_writer = tf.summary.create_file_writer(train_path)
        self.valid_writer = tf.summary.create_file_writer(valid_path)

    def scalar(self, tag, value, step):
        if tag.startswith('train_'):
            writer = self.train_writer
            tag = tag[len('train_'):]
        else:
            writer = self.valid_writer
            if tag.startswith('valid_'):
                tag = tag[len('valid_'):]
        with writer.as_default():
            tf.summary.scalar(tag, data=value, step=step)

    def image(self, tag, images, step):
        if tag.startswith('train_'):
            writer = self.train_writer
            tag = tag[len('train_'):]
        else:
            writer = self.valid_writer
            if tag.startswith('valid_'):
                tag = tag[len('valid_'):]
        with writer.as_default():
            tf.summary.image(tag, images, max_outputs=16, step=step)


def record(monitor, results, step, prefix=None):
    tags = results
    if prefix is not None:
        tags = {"{}{}".format(prefix, k): v for k, v in results.items()}

    for key, val in tags.items():
        monitor.scalar(key, val, step)