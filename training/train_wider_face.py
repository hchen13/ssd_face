import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from signal import signal, SIGINT
import cv2
from ethan_toolbox import draw_boxes
from training.loss import create_losses
from datetime import datetime

import prototype.pvanet as pvanet

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from training.tensorboard_monitor import Monitor
from training.tools import load_dataset_from_tfrecords, PROJECT_ROOT, preprocess_input


@tf.function
def train_on_batch(model, batch_data, optimizer):
    batch_images, labels_true, offsets_true = batch_data
    labels_true = tf.expand_dims(labels_true, -1)
    x = preprocess_input(batch_images)
    with tf.GradientTape() as tape:
        labels_pred, offsets_pred, _ = model(x)
        c_loss, l_loss = loss_function(labels_pred, offsets_pred, labels_true, offsets_true)
        total_loss = c_loss + l_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    report = dict(
        total_loss=total_loss,
        classification_loss=c_loss,
        localization_loss=l_loss
    )
    return report, gradients


def evaluate(model, dataset, limit=None):
    total_loss = tf.metrics.Mean(name='total_loss')
    class_loss = tf.metrics.Mean(name='class_loss')
    loc_loss = tf.metrics.Mean(name='loc_loss')
    accuracy = tf.metrics.BinaryAccuracy(name='acc')
    precision = tf.metrics.Precision(name='precision')
    recall = tf.metrics.Recall(name='recall')
    for i, (images, labels_true, offsets_true) in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        labels_true = tf.expand_dims(labels_true, -1)
        images = preprocess_input(images)
        labels_pred, offsets_pred, anchors = model.predict(images)
        c_loss, l_loss = loss_function(labels_pred, offsets_pred, labels_true, offsets_true)
        t_loss = c_loss + l_loss
        total_loss.update_state(t_loss)
        class_loss.update_state(c_loss)
        loc_loss.update_state(l_loss)
        accuracy.update_state(labels_true, labels_pred)
        precision.update_state(labels_true, labels_pred)
        recall.update_state(labels_true, labels_pred)
    return dict(
        total_loss=total_loss.result().numpy(),
        classification_loss=class_loss.result().numpy(),
        localization_loss=loc_loss.result().numpy(),
        accuracy=accuracy.result().numpy(),
        precision=precision.result().numpy(),
        recall=recall.result().numpy()
    )


def show_detection(model, monitor, step):
    for i in range(3):
        path = os.path.join(os.path.expanduser('~'), f'Pictures/things/face_test{i}.jpg')
        image = cv2.imread(path)
        image = cv2.resize(image, (512, 512))
        boxes, scores = model.detect(image, threshold=.5)
        disp = draw_boxes(image, boxes, mode='minmax', color=(0, 200, 20), thickness=2)
        show = np.expand_dims(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), axis=0)
        monitor.image(f'detections/{i + 1}', show, step)


def exit_handler(signal, frame):
    print("[info] exiting program, saving model weights...")
    p = os.path.join(PROJECT_ROOT, 'weights', 'temp')
    os.makedirs(p, exist_ok=True)
    p = os.path.join(p, 'killed.h5')
    ssd.model.save_weights(p)
    print('[info] terminated.')
    exit(0)


if __name__ == '__main__':
    if os.environ.get('AIBOX') is None:  # 本地开发
        dataset_root = '/Users/ethan/datasets/WIDER_SSD/'
        epochs = 1
        batch_size = 4
    else:  # 训练机
        dataset_root = '/media/ethan/DataStorage/WIDER_SSD/'
        epochs = 100
        batch_size = 32

    lr = 3e-4
    eval_steps = 1000
    _continue_train = True

    trainset, validset = load_dataset_from_tfrecords(dataset_root, batch_size, shuffle=True)

    image_size = 512
    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    num_boxes = len(aspect_ratios) + 1 if 1. in aspect_ratios else 0
    print('[info] construct and initialize SSD model...')
    ssd = pvanet.SSD(aspect_ratios=aspect_ratios)
    # ssd = original.SSD(aspect_ratios=aspect_ratios)

    print("[info] freezing feature extractor layers")
    ssd.freeze_layers(stop_name='conv2_3')
    if _continue_train:
        print("[info] continue training using previously saved weights.")
        p = os.path.join(PROJECT_ROOT, 'weights', 'temp', 'pause.h5')
        ssd.model.load_weights(p, by_name=True)
    else:
        print("[info] initialize model with pre-trained PVANet weights.")
        ssd.init_pvanet(os.path.join(PROJECT_ROOT, 'weights', 'pvanet_init.h5'))
        # ssd.init_vgg16()

    print(f'[info] model has {len(ssd.model.trainable_variables)} trainable weights.')
    print('[info] complete.\n')

    experiment_name = f'ssd-pvanet@{datetime.now().strftime("%-y%m%d-%H:%M:%S")}'
    monitor = Monitor(experiment_name)

    print('[info] preparing learning policy...')
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_function = create_losses(neg_ratio=3)
    print('[info] complete.\n')

    signal(SIGINT, exit_handler)
    print("[info] start training...")
    global_steps = 0
    _log_steps = 20
    for e in range(epochs):
        print(f"epoch #{e + 1}/{epochs}@{datetime.now()}:")
        for local_step, batch_data in tqdm(enumerate(trainset)):
            global_steps += 1
            train_report, gradients = train_on_batch(ssd.model, batch_data, opt)

            if global_steps % _log_steps == 0:
                monitor.write_reports(train_report, global_steps, prefix='train_')
                monitor.write_weights(ssd.model, step=global_steps)

            if global_steps % eval_steps == 0:
                print("\r[info] evaluating..", end='', flush=True)
                eval_report = evaluate(ssd.model, validset, limit=global_steps // 100)
                monitor.write_reports(eval_report, global_steps, prefix='valid_')
                print("\r[info] evaluation complete.", end='', flush=True)

        show_detection(ssd, monitor, e + 1)
        print(f'[info] saving intermediate weights at epoch #{e + 1}')
        p = os.path.join(PROJECT_ROOT, 'weights', 'temp')
        os.makedirs(p, exist_ok=True)
        ssd.model.save_weights(os.path.join(p, f'tmp_e{e+1}.h5'))

