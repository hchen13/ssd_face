import os
import re

import numpy as np
import tensorflow as tf

from architecture.default_box import DefaultBox
from utils.helper_funcs import convert_box_coordinates, calculate_overlap_ratio, draw_boxes, show_image, crop_image, \
    convert_box_range


def create_ssd_labels(bounding_boxes, feature_map_shapes, scales):
    db_list = []
    for i, (height, width) in enumerate(feature_map_shapes):
        inp = np.random.uniform(size=(1, height, width, 3))
        boxes_raw = DefaultBox(scale=scales[i], scale_next=scales[i + 1])(inp)
        default_boxes_grid = boxes_raw.numpy().squeeze(axis=0)
        default_boxes = default_boxes_grid.reshape((-1, 4))
        db_list.append(default_boxes)
    default_boxes = np.vstack(db_list)

    db_minmax = convert_box_coordinates(default_boxes, mode_from='centroid', mode_to='minmax')
    iou = calculate_overlap_ratio(db_minmax, bounding_boxes, method='union')
    scores = (np.max(iou, axis=1) > .5).astype('uint8')

    bounding_box_ids = np.argmax(iou, axis=1)
    ground_truths = bounding_boxes[bounding_box_ids]
    ground_truths = convert_box_coordinates(ground_truths, mode_from='minmax', mode_to='centroid')
    delta = np.empty_like(default_boxes)
    delta[:, 0] = (ground_truths[:, 0] - default_boxes[:, 0]) / default_boxes[:, 2]
    delta[:, 1] = (ground_truths[:, 1] - default_boxes[:, 1]) / default_boxes[:, 3]
    delta[:, 2] = np.log(ground_truths[:, 2] / default_boxes[:, 2])
    delta[:, 3] = np.log(ground_truths[:, 3] / default_boxes[:, 3])
    return scores, delta


def is_file_name(text):
    pattern = r'^.*/.*\.jpg$'
    m = re.match(pattern, text)
    return m is not None


class DatasetWriter:
    def __init__(self, root_dir, subset, buffer_size=64):
        self.file_dir = os.path.join(root_dir, subset)
        self.buffer = []
        self.max_size = buffer_size
        self.file_count = 0
        os.makedirs(self.file_dir, exist_ok=True)

    def update(self, record):
        self.buffer.append(record)
        if len(self.buffer) >= self.max_size:
            self.dump(self.max_size)

    def dump(self, size=None):
        if size is None:
            size = len(self.buffer)
        if size == 0:
            return
        filename = "data{}.tfrecords".format(self.file_count)
        path = os.path.join(self.file_dir, filename)
        with tf.io.TFRecordWriter(path) as writer:
            for serialized in self.buffer[:size]:
                writer.write(serialized)
        self.file_count += 1
        self.buffer = self.buffer[size:]


class ImageFile:
    def __init__(self, root_dir, file_path, subset):
        file_path = file_path.strip()
        self.file_path = os.path.join(root_dir, 'WIDER_{}'.format(subset), 'images', file_path)
        self.box_number = 0
        self.boxes = []

    def _is_box_number(self, line):
        pattern = r'^\d+$'
        return re.match(pattern, line) is not None

    def _is_box_info(self, line):
        pattern = r'^(\d+\s){10}$'
        return re.match(pattern, line) is not None

    def parse(self, line):
        if self._is_box_number(line):
            self.box_number = int(line)

        elif self._is_box_info(line):
            if self.box_number == 0:
                return
            items = list(map(int, line.split()))
            box = items[:4]
            if box[2] == 0 or box[3] == 0:
                return
            if min(box[2], box[3]) < 10:
                return
            self.boxes.append(box)

    def correct_data(self):
        if self.box_number != len(self.boxes):
            self.box_number = len(self.boxes)

        self.boxes = np.array(self.boxes, dtype=np.float32)


def generate_patch(boxes_centroid, threshold):
    boxes_minmax = convert_box_coordinates(boxes_centroid, mode_from='centroid', mode_to='minmax')
    while True:
        patch_w = np.random.uniform(boxes_centroid[:, 2:].min(), 1.)
        patch_h = patch_w
        patch_x0 = np.random.uniform(0, 1 - patch_w)
        patch_y0 = np.random.uniform(0, 1 - patch_h)
        patch_x1 = patch_x0 + patch_w
        patch_y1 = patch_y0 + patch_h
        patch = np.array([
            [patch_x0, patch_y0, patch_x1, patch_y1]
        ], dtype=np.float32)
        patch = np.clip(patch, 0., 1.)
        ious = calculate_overlap_ratio(patch, boxes_minmax, method='min')
        if tf.math.reduce_any(ious >= threshold):
            break
    return patch[0], ious[0]


def random_patching(image, bounding_boxes, mode):
    box_centroid = convert_box_coordinates(bounding_boxes, mode_from=mode, mode_to='centroid')
    box_minmax = convert_box_coordinates(bounding_boxes, mode_from=mode, mode_to='minmax')
    image_height, image_width = image.shape[:2]
    while True:
        patch_w = np.random.uniform(.1, 1.) * image_width
        aspect_ratio = np.random.uniform(.5, 2)
        patch_h = min(patch_w / aspect_ratio, image_height - 1)
        patch_x0 = np.random.uniform(0, image_width - patch_w)
        patch_y0 = np.random.uniform(0, image_height - patch_h)
        patch_x1 = patch_x0 + patch_w
        patch_y1 = patch_y0 + patch_h
        patch = np.array([
            [patch_x0, patch_y0, patch_x1, patch_y1]
        ], dtype=np.int)
        ious = calculate_overlap_ratio(patch, box_minmax, method='min')
        if np.any(ious >= .7):
            break

    patch = convert_box_range(patch, image.shape, mode='absolute2relative')
    box_centroid = convert_box_range(box_centroid, image.shape, mode='absolute2relative')
    patch = patch[0]
    ious = ious[0]

    keep_indices = (
        (ious > .3) &
        (box_centroid[:, 0] > patch[0]) &
        (box_centroid[:, 1] > patch[1]) &
        (box_centroid[:, 0] < patch[2]) &
        (box_centroid[:, 1] < patch[3])
    )
    if not np.any(keep_indices):
        return image, bounding_boxes
    patch_image = crop_image(image, patch, box_mode='minmax')
    box_centroid = box_centroid[keep_indices]
    patch_corner = convert_box_coordinates(np.expand_dims(patch, axis=0), mode_from='minmax', mode_to='corner')[0]

    box_centroid[:, 0] = (box_centroid[:, 0] - patch_corner[0]) / patch_corner[2]
    box_centroid[:, 1] = (box_centroid[:, 1] - patch_corner[1]) / patch_corner[3]
    box_centroid[:, 2] = box_centroid[:, 2] / patch_corner[2]
    box_centroid[:, 3] = box_centroid[:, 3] / patch_corner[3]
    box_minmax = convert_box_coordinates(box_centroid, mode_from='centroid', mode_to='minmax')
    box_minmax = np.clip(box_minmax, 0., 1.)
    return patch_image, convert_box_coordinates(box_minmax, mode_from='minmax', mode_to=mode)


def single_patching(image, bounding_boxes, mode):
    image_height, image_width = image.shape[:2]
    boxes_absolute = bounding_boxes

    boxes_minmax = convert_box_coordinates(boxes_absolute, mode_from=mode, mode_to='minmax')
    box_areas = (boxes_minmax[:, 2] - boxes_minmax[:, 0]) * (boxes_minmax[:, 3] - boxes_minmax[:, 1])
    random_box = boxes_minmax[[box_areas.argmax()]]
    width, height = random_box[0, 2] - random_box[0, 0], random_box[0, 3] - random_box[0, 1]
    # while True:
    #     patch = np.empty_like(random_box)
    #     patch_width = int(width * np.random.uniform(1., 2.))
    #     patch_height = int(height * np.random.uniform(1., 2.))
    #     patch[:, 0] = max(np.random.randint(random_box[:, 2] - patch_width - 1, random_box[:, 0]), 0)
    #     patch[:, 1] = max(np.random.randint(random_box[:, 3] - patch_height - 1, random_box[:, 1]), 0)
    #     patch[:, 2] = min(patch[:, 0] + patch_width, image_width - 1)
    #     patch[:, 3] = min(patch[:, 1] + patch_height, image_height - 1)
    #     ious = calculate_overlap_ratio(patch, random_box, method='min')
    #     if ious.max() > .8:
    #         break
    patch = np.empty_like(random_box)
    patch_width = int(width * np.random.uniform(1., 2.))
    patch_height = int(height * np.random.uniform(1., 2.))
    patch[:, 0] = max(np.random.randint(random_box[:, 2] - patch_width - 1, random_box[:, 0]), 0)
    patch[:, 1] = max(np.random.randint(random_box[:, 3] - patch_height - 1, random_box[:, 1]), 0)
    patch[:, 2] = min(patch[:, 0] + patch_width, image_width - 1)
    patch[:, 3] = min(patch[:, 1] + patch_height, image_height - 1)


    ious = calculate_overlap_ratio(patch, boxes_minmax, method='min')[0]
    keep_indices = ious > .7
    visible_boxes = boxes_minmax[keep_indices]

    patch = convert_box_range(patch, image.shape, mode='absolute2relative')
    visible_boxes = convert_box_range(visible_boxes, image.shape, mode='absolute2relative')

    patch_image = crop_image(image, patch[0], box_mode='minmax')
    patch_corner = convert_box_coordinates(patch, mode_from='minmax', mode_to='corner')[0]
    box_centroid = convert_box_coordinates(visible_boxes, mode_from='minmax', mode_to='centroid')
    box_centroid[:, 0] = (box_centroid[:, 0] - patch_corner[0]) / patch_corner[2]
    box_centroid[:, 1] = (box_centroid[:, 1] - patch_corner[1]) / patch_corner[3]
    box_centroid[:, 2] = box_centroid[:, 2] / patch_corner[2]
    box_centroid[:, 3] = box_centroid[:, 3] / patch_corner[3]
    visible_boxes = convert_box_coordinates(box_centroid, mode_from='centroid', mode_to='minmax')
    visible_boxes = np.clip(visible_boxes, 0., 1.)

    return patch_image, visible_boxes


def compute_target(bounding_boxes_minmax, anchor_boxes, image=None):
    """
    compute ground truth targets used for SSD model training
    :param bounding_boxes_minmax: ground truth bounding boxes represented in minmax format (x0, y0, x1, y1)
    :param anchor_boxes:
    :return:
    """
    anchor_minmax = convert_box_coordinates(anchor_boxes, mode_from='centroid', mode_to='minmax')
    iou = calculate_overlap_ratio(anchor_minmax, bounding_boxes_minmax, method='union')
    scores = (np.max(iou, axis=1) > .5).astype('uint8')
    bounding_box_ids = np.argmax(iou, axis=1)
    ground_truths = bounding_boxes_minmax[bounding_box_ids]
    ground_truths = convert_box_coordinates(ground_truths, mode_from='minmax', mode_to='centroid')
    offset = np.empty_like(anchor_boxes)
    offset[:, 0] = (ground_truths[:, 0] - anchor_boxes[:, 0]) / anchor_boxes[:, 2]
    offset[:, 1] = (ground_truths[:, 1] - anchor_boxes[:, 1]) / anchor_boxes[:, 3]
    offset[:, 2] = np.log(ground_truths[:, 2] / anchor_boxes[:, 2])
    offset[:, 3] = np.log(ground_truths[:, 3] / anchor_boxes[:, 3])

    return scores, offset