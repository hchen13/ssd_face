import os
import sys
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from imutils import resize

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)

from architecture.prototypes import SSD
from training.losses import compute_loss
from training.tensorboard_monitor import Monitor, record
from training.dataset.WIDER_data_generator import create_dataset_generator
from utils.helper_funcs import draw_boxes


@tf.function
def train_on_batch(ssd, batch_data, optimizer):
    images, labels_true, offsets_true = batch_data
    with tf.GradientTape() as tape:
        labels_pred, offsets_pred, _ = ssd.model(images)
        total_loss, c_loss, l_loss = compute_loss((labels_true, offsets_true), (labels_pred, offsets_pred))
    gradients = tape.gradient(total_loss, ssd.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.model.trainable_variables))
    return dict(
        total_loss=total_loss,
        classification_loss=c_loss,
        localization_loss=l_loss
    )


def evaluate(ssd, dataset):
    total_loss = tf.metrics.Mean(name='total_loss')
    class_loss = tf.metrics.Mean(name='class_loss')
    loc_loss = tf.metrics.Mean(name='loc_loss')
    precision = tf.metrics.Precision(name='precision')
    recall = tf.metrics.Recall(name='recall')
    for images, labels_true, offsets_true in dataset:
        labels_pred, offsets_pred, anchors = ssd.model.predict(images)
        t_loss, c_loss, l_loss = compute_loss((labels_true, offsets_true), (labels_pred, offsets_pred))
        total_loss.update_state(t_loss)
        class_loss.update_state(c_loss)
        loc_loss.update_state(l_loss)
        precision.update_state(labels_true, labels_pred)
        recall.update_state(labels_true, labels_pred)
    return dict(
        total_loss=total_loss.result().numpy(),
        classification_loss=class_loss.result().numpy(),
        localization_loss=loc_loss.result().numpy(),
        precision=precision.result().numpy(),
        recall=recall.result().numpy()
    )


def train(epochs, lr, eval_steps=500):
    print("[info] initializing model for training...")
    backbone = 'vgg'
    ssd = SSD(backbone=backbone, input_shape=None)
    ssd.load_model("initial.h5")

    train_length = 12859 / 64
    lr = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[
            int(epochs * train_length / 2),
            int(epochs * train_length * 5 / 6)],
        values=[lr, lr / .3, lr * .1]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    print("[info] initialization complete.\n")

    if not __debug:
        execution_time = datetime.now().strftime("%-y%m%d-%H:%M:%S")
        experiment_name = "SSDFace@" + execution_time
        monitor = Monitor(experiment_name)

    print("[info] start training...\n")
    global_steps = 0
    for e in range(epochs):
        print("\n[info] epoch #{}/{} @ {}:".format(e + 1, epochs, datetime.now()))
        for local_steps, batch_data in enumerate(trainset):
            global_steps += 1
            train_report = train_on_batch(ssd, batch_data, opt)

            # heart beat
            if global_steps % 7 == 0:
                print("\r[info] batch #{}.".format(local_steps), end='', flush=True)

            if global_steps % eval_steps == 0:
                print("\r[info] evaluating..", end='', flush=True)
                eval_report = evaluate(ssd, validset)
                if not __debug:
                    step = int(global_steps // eval_steps)
                    record(monitor, train_report, step, prefix='train_')
                    record(monitor, eval_report, step, prefix='valid_')
                print("\r[info] evaluation done.", end='', flush=True)

        if (e + 1) % 10 == 0 and not __debug:
            print("\n[info] saving intermediate weights at epoch #{}...".format(e + 1))
            ssd.save_model("{}_ssd_{}.h5".format(backbone, e + 1))

        if not __debug:
            test_detect(ssd, monitor, e)

    print("\n[info] training complete.")
    if not __debug:
        print("[info] saving weights...")
        ssd.save_model("{}_ssd.h5".format(backbone))
        print("[info] weights saved.")


def test_detect(ssd, monitor, step):
    for i in range(3):
        path = os.path.join(os.path.expanduser('~'), 'Pictures/things/face_test{}.jpg'.format(i))
        image = cv2.imread(path)
        image = resize(image, height=500)
        boxes, scores = ssd.detect_faces(image, threshold=.7)
        disp = draw_boxes(image, boxes, mode='minmax', color=(0, 200, 20), thickness=2)
        show = np.expand_dims(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), axis=0)
        monitor.image("detection #{}".format(i), show, step)


if __name__ == '__main__':
    __debug = True
    epochs = 120

    if os.environ.get('AIBOX') is None:  # 本地开发
        dataset_root = '/Users/ethan/datasets/WIDER/'
        record_dir = '/Users/ethan/datasets/SSD_face_dataset/'
        batch_size = 4
    else:  # 训练机
        dataset_root = '/media/ethan/DataStorage/WIDER/'
        record_dir = '/media/ethan/DataStorage/SSD_face_dataset/'
        batch_size = 64
        __debug = False

    trainset = create_dataset_generator(dataset_root, 'train', batch_size, shuffle=True, take=50 if __debug else None)
    validset = create_dataset_generator(dataset_root, 'valid', batch_size, shuffle=False, take=10 if __debug else None)

    train(1 if __debug else epochs, 1e-4, eval_steps=10 if __debug else 200)
