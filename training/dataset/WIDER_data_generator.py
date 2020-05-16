import os
from functools import partial

import cv2
import numpy as np
import progressbar

from architecture.prototypes import SSD
from training.dataset.helper_funcs import is_file_name, ImageFile, random_patching, compute_target, single_patching
from utils.helper_funcs import draw_boxes, show_image, convert_box_range, convert_box_coordinates, horizontal_flip, \
    random_flip, non_max_suppression
from utils.preprocess import preprocess_image, deprocess_image

import tensorflow as tf

widgets = ["Progress: ", progressbar.Bar(), progressbar.Percentage(), " ", progressbar.ETA()]

class WIDER:
    def __init__(self, root_dir, image_size=300):
        self.root_dir = root_dir
        self.image_size = image_size
        ssd = SSD(backbone='vgg', input_shape=image_size)
        _, _, self.default_boxes = ssd.model.predict(np.random.uniform(-1., 1., size=(1, image_size, image_size, 3)))
        self.default_boxes = self.default_boxes[0]

        self.train_files = self.get_image_files('train')
        self.valid_files = self.get_image_files('val')

    def get_image_files(self, subset):
        if subset not in ['train', 'val']:
            raise TypeError("[load_ground_truths] param `subset` has to be in {}".format(['train', 'val']))

        print("[info] Processing [{}] dataset...".format(subset))

        file_name = 'wider_face_{}_bbx_gt.txt'.format(subset)
        file_path = os.path.join(self.root_dir, 'wider_face_split', file_name)
        with open(file_path) as f:
            lines = f.readlines()
        image_files = []
        current = None
        for i, line in enumerate(lines):
            if is_file_name(line):
                if current is not None:
                    current.correct_data()
                    image_files.append(current)
                current = ImageFile(self.root_dir, line, subset)
            else:
                current.parse(line)
        current.correct_data()
        image_files.append(current)
        image_files = list(filter(lambda f: f.box_number > 0, image_files))
        print("[info] {} images pre-loaded for [{}] dataset.\n".format(len(image_files), subset))
        return image_files

    def get_generator(self, subset):
        image_files = self.train_files if subset == 'train' else self.valid_files

        for i, file_object in enumerate(image_files):
            image = cv2.imread(file_object.file_path)
            boxes_corner = file_object.boxes

            mean_size = np.mean(boxes_corner[:, 2:])
            boxes_minmax = convert_box_coordinates(boxes_corner, mode_from='corner', mode_to='minmax')
            # if subset != 'train':
            #     choices = ['origin']
            # else:
            #     choices = ['patch']
            #     if mean_size > 50:
            #         choices.append('single')
            choices = ['origin', 'patch']
            augmentation = np.random.choice(choices)
            if mean_size > 50:
                augmentation = 'single'

            # patching methods will take bounding boxes as absolute values and return bounding boxes in proportional
            #  values, so that the image resizing (300x300) will not affect the bounding box values
            if augmentation == 'patch':
                image, boxes_minmax = random_patching(image, boxes_minmax, mode='minmax')
            elif augmentation == 'single':
                image, boxes_minmax = single_patching(image, boxes_minmax, mode='minmax')
            elif augmentation == 'origin':
                boxes_minmax = convert_box_range(boxes_minmax, image.shape, mode='absolute2relative')
            image = cv2.resize(image, (self.image_size, self.image_size))

            # if augmentation == 'single':
            #     d = draw_boxes(image, boxes_minmax, mode='minmax', color=(1, 200, 1), thickness=2)
            #     show_image(d, destroy=True, title=augmentation)

            image, boxes_minmax = random_flip(image, boxes_minmax, mode='minmax')

            # after patching, the bounding box values need to be restored as absolute values since we need to compute
            # the training target values which involves calculating the IOU between bounding boxes and default boxes,
            # and default boxes are represented as absolute values
            boxes_minmax = convert_box_range(boxes_minmax, image.shape, mode='relative2absolute')

            gt_conf, gt_offsets = compute_target(boxes_minmax, self.default_boxes)
            if np.sum(gt_conf) == 0:
                continue

            # if augmentation == 'single':
            d = draw_boxes(image, boxes_minmax, mode='minmax', color=(1, 200, 1), thickness=2)
            show_image(d, destroy=True, title=augmentation)

            # ids = gt_conf == 1
            # matched = self.default_boxes[ids]
            # d = draw_boxes(image, matched, mode='centroid')
            # show_image(d, destroy=True)
            image = preprocess_image(image)

            yield tf.constant(image), tf.expand_dims(gt_conf, axis=-1), tf.constant(gt_offsets)


def create_dataset_generator(root_dir, subset, batch_size, shuffle=True, take=None):
    wider = WIDER(root_dir)
    gen = partial(wider.get_generator, subset=subset)
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=40)
    dataset = dataset.batch(batch_size)
    if take:
        return dataset.take(take)
    return dataset


if __name__ == '__main__':
    if os.environ.get('AIBOX') is None:  # 本地开发
        dataset_root = '/Users/ethan/datasets/WIDER/'
    else:  # 训练机
        dataset_root = '/media/ethan/DataStorage/WIDER/'

    ssd = SSD()
    wider = WIDER(dataset_root)
    gen = wider.get_generator('train')
    for i in gen:
        pass
    # trainset = create_dataset_generator(dataset_root, 'train', 1, shuffle=False, take=5)
    # for i, batch in enumerate(trainset):
    #     images, conf_true, offset_true = batch
    #     conf_pred, offset_pred, anchors = ssd.model.predict(images)
    #     conf = conf_true.numpy().squeeze()
    #     offset = offset_true.numpy().squeeze()
    #     anchors = anchors.squeeze()
    #     box = np.empty_like(anchors)
    #     box[:, 0] = offset[:, 0] * anchors[:, 2] + anchors[:, 0]
    #     box[:, 1] = offset[:, 1] * anchors[:, 3] + anchors[:, 1]
    #     box[:, 2] = np.exp(offset[:, 2]) * anchors[:, 2]
    #     box[:, 3] = np.exp(offset[:, 3]) * anchors[:, 3]
    #     positive_indices = np.where(conf > .8)
    #     positive_boxes = box[positive_indices]
    #     scores = conf[positive_indices]
    #     positive_boxes_minmax = convert_box_coordinates(positive_boxes, mode_from='centroid', mode_to='minmax')
    #     boxes, scores = non_max_suppression(scores, positive_boxes_minmax, method='union', threshold=.5)
    #     d = draw_boxes(deprocess_image(images.numpy()[0]), boxes, mode='minmax', color=(0, 255, 2))
    #     show_image(d, destroy=True)

