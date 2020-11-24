import cv2
import numpy as np
from ethan_toolbox import convert_box_coordinates, calculate_overlap_ratio, convert_box_range, is_box_inside, \
    crop_from_boxes
import tensorflow as tf


def make_ssd_sample(image, boxes, default_boxes):
    default_boxes = convert_box_range(default_boxes, image_shape=image.shape, mode='relative2absolute').astype('float32')
    anchor_minmax = convert_box_coordinates(default_boxes, mode_from='centroid', mode_to='minmax')
    ious = calculate_overlap_ratio(anchor_minmax, boxes, method='union')
    scores = (np.max(ious, axis=1) > .5).astype('uint8')

    box_ids = np.argmax(ious, axis=1)
    gt = boxes[box_ids].astype('float32')
    gt = convert_box_coordinates(gt, mode_from='minmax', mode_to='centroid')
    delta = np.empty_like(default_boxes)
    delta[:, 0] = (gt[:, 0] - default_boxes[:, 0]) / default_boxes[:, 2]
    delta[:, 1] = (gt[:, 1] - default_boxes[:, 1]) / default_boxes[:, 3]
    delta[:, 2] = np.log(gt[:, 2] / default_boxes[:, 2])
    delta[:, 3] = np.log(gt[:, 3] / default_boxes[:, 3])
    return scores, delta


def generate_sub_images(image, boxes, image_size):
    if not len(boxes):
        return [], []
    if isinstance(image, str):
        image = cv2.imread(image)
    height, width, _ = image.shape
    boxes_corner = np.array(boxes)
    boxes = convert_box_coordinates(
        boxes_corner,
        mode_from='corner',
        mode_to='minmax'
    )
    max_region_size = min(height, width)
    regions = []
    boxes_of_regions = []
    for region_size in np.array([.5, .67, 1.]) * max_region_size:
        region_size = int(region_size) - 1
        for x in range(0, width - region_size, region_size // 3):
            for y in range(0, height - region_size, region_size // 3):
                region = [x, y, x + region_size, y + region_size]
                valid_boxes = []
                for box in boxes:
                    if is_box_inside(box, region):
                        relative_box = box - np.tile(region[:2], 2)
                        valid_boxes.append(relative_box)
                if len(valid_boxes) > 0:
                    regions.append(region)
                    valid_boxes = np.array(valid_boxes)
                    valid_boxes = convert_box_range(valid_boxes, (region_size + 1, region_size + 1),
                                                    mode='absolute2relative')
                    boxes_of_regions.append(valid_boxes)
    assert len(regions) == len(boxes_of_regions)
    if len(regions) == 0:
        boxes = convert_box_range(boxes, image.shape, mode='absolute2relative')
        image = cv2.resize(image, (image_size, image_size))
        return [image], [boxes]
    crops = crop_from_boxes(image, regions, image_size)
    return crops, boxes_of_regions


def augment(dataset):
    def _transform(image, label, offset):
        x = tf.image.random_brightness(image, max_delta=.3)
        x = tf.image.random_saturation(x, .8, 1.2)
        x = tf.image.random_contrast(x, .8, 1.2)
        x = tf.image.random_hue(x, max_delta=.1)
        return x, label, offset
    return dataset.map(_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
