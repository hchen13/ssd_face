import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_project_root)

import re
import warnings

import cv2
import numpy as np
import progressbar

from architecture.prototypes import SSD
from training.dataset.helper_funcs import is_file_name, create_ssd_labels, DatasetWriter
from training.dataset.tfannotation import create_tfrecord
from utils.helper_funcs import convert_box_coordinates, convert_box_range, crop_from_boxes, \
    is_box_inside, draw_boxes, show_image, random_flip

widgets = ["Progress: ", progressbar.Bar(), progressbar.Percentage(), " ", progressbar.ETA()]


class ImageFile:
    def __init__(self, file_path, subset):
        file_path = file_path.strip()
        self.file_path = os.path.join(dataset_root, 'WIDER_{}'.format(subset), 'images', file_path)
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
            self.boxes.append(box)

    def check_integrity(self):
        if self.box_number != len(self.boxes):
            warnings.warn("[ImageFile] box count doesn\'t match specified number")
            self.box_number = len(self.boxes)
            return False

        if not os.path.exists(self.file_path):
            warnings.warn("[ImageFile] image not found: \n{}".format(self.file_path))
            return False

        return True

    def generate_sub_images(self, image_size=300):
        if not self.box_number:
            return
        image = cv2.imread(self.file_path)
        height, width, _ = image.shape
        boxes_corner = np.array(self.boxes)
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
                        valid_boxes = convert_box_range(valid_boxes, (region_size + 1, region_size + 1), mode='absolute2relative')
                        boxes_of_regions.append(valid_boxes)
        assert len(regions) == len(boxes_of_regions)
        if len(regions) == 0:
            boxes = convert_box_range(boxes, image.shape, mode='absolute2relative')
            image = cv2.resize(image, (image_size, image_size))
            return [image], [boxes]
        crops = crop_from_boxes(image, regions, image_size)
        return crops, boxes_of_regions


def load_ground_truths(root_dir, subset):
    if subset not in ['train', 'val']:
        raise TypeError("[load_ground_truths] param `subset` has to be in {}".format(['train', 'val']))
    file_name = 'wider_face_{}_bbx_gt.txt'.format(subset)
    file_path = os.path.join(root_dir, 'wider_face_split', file_name)
    with open(file_path) as f:
        lines = f.readlines()

    image_files = []
    current = None
    for i, line in enumerate(lines):
        if is_file_name(line):
            if current is not None:
                image_files.append(current)
            current = ImageFile(line, subset)
        else:
            current.parse(line)
    image_files.append(current)
    return image_files


def process_dataset(subset, image_size=300):
    print("[info] Processing [{}] dataset...".format(subset))

    print("[info] loading ground truth txt file...")
    train_files = load_ground_truths(dataset_root, subset)
    print("[info] loading complete, checking data integrity...")
    error_count = 0

    pbar = progressbar.ProgressBar(maxval=len(train_files), widgets=widgets, term_width=100)
    pbar.start()
    for i, image_file in enumerate(train_files):
        error_count += 0 if image_file.check_integrity() else 1
        pbar.update(i)
    pbar.finish()
    print("[info] all {} {} images are checked, {} errors found.\n".format(len(train_files), subset, error_count))

    print("[info] getting model feature map shapes based on input image size {0}x{0}...".format(image_size))
    ssd = SSD(backbone='squeeze', input_shape=image_size)
    feature_map_shapes = []
    for i in range(6):
        map_layer = ssd.model.get_layer(name='feature_map{}'.format(i)).output
        feature_map_shapes.append(map_layer.shape[1:3].as_list())
    print("done.\n")

    print("[info] preparing data for training...")
    pbar = progressbar.ProgressBar(maxval=len(train_files), widgets=widgets, term_width=100)
    writer = DatasetWriter(record_dir, subset, buffer_size=200)
    pbar.start()
    sample_count = 0
    for i, image_file in enumerate(train_files):
        if image_file.box_number == 0:
            continue
        images, labels = image_file.generate_sub_images(image_size=image_size)
        sample_count += len(images)
        for j, image in enumerate(images):
            bounding_boxes = labels[j]
            # randomly flip the image and its bounding boxes horizontally with a probability of .5
            image, bounding_boxes = random_flip(image, bounding_boxes, mode='minmax')

            label, delta = create_ssd_labels(bounding_boxes, feature_map_shapes, ssd.scales)
            record = create_tfrecord(image, label, delta)
            writer.update(record)
        pbar.update(i)
    writer.dump()
    pbar.finish()
    print("\n[info] {} training samples in total.".format(sample_count))


if __name__ == '__main__':
    if os.environ.get('AIBOX') is None:  # 本地开发
        dataset_root = '/Users/ethan/datasets/WIDER/'
        record_dir = '/Users/ethan/datasets/SSD_face_dataset/'
    else:  # 训练机
        dataset_root = '/media/ethan/DataStorage/WIDER/'
        record_dir = '/media/ethan/DataStorage/SSD_face_dataset/'

    process_dataset('train')
    # process_dataset('val')

