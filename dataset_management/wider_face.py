import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from glob import glob
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from prototype.original import SSD
import re
import cv2
import numpy as np
import tensorflow as tf
from ethan_toolbox import convert_box_coordinates, is_box_inside, convert_box_range, \
    crop_from_boxes, random_flip, show_image
from tqdm import tqdm

from dataset_management.tfannotation import create_tfrecord, read_tfrecord
from dataset_management.utils import make_ssd_sample


class ImageFile:
    BOX_COUNT_ERROR = -1
    IMAGE_NOT_EXIST = -2

    def __init__(self, file_path):
        self.file_path = file_path.strip()
        self.box_number = 0
        self.boxes = []

    def __repr__(self):
        return f"{self.file_path}  #boxes: {len(self.boxes)}"

    @classmethod
    def is_file_name(cls, text):
        pattern = r'^.*/.*\.jpg$'
        m = re.match(pattern, text)
        return m is not None

    @classmethod
    def _is_box_number(self, line):
        pattern = r'^\d+$'
        return re.match(pattern, line) is not None

    @classmethod
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
            return self.BOX_COUNT_ERROR
        if not os.path.exists(self.file_path):
            return self.IMAGE_NOT_EXIST
        return True

    def generate_sub_images(self, image_size):
        if not len(self.boxes):
            return [], []
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


class WIDER:
    def __init__(self, root_dir, min_face_size=32):
        self.root = root_dir
        self.label_dir = os.path.join(self.root, 'wider_face_split')
        self.min_face = min_face_size
        self.load_ground_truths()
        self.check_integrity()
        self.filter()

    def __repr__(self):
        lens = len(self.train_images), len(self.valid_images), len(self.train_images) + len(self.valid_images)
        return "training: {} | validation: {} | total: {}".format(*lens)

    def load_ground_truths(self):
        print("[WIDER] loading ground truths...")
        self.train_images = self._load_gt('train')
        self.valid_images = self._load_gt('val')
        print("[WIDER] ground truths loading complete.\n")

    def _load_gt(self, subset):
        gt_file = f'wider_face_{subset}_bbx_gt.txt'
        gt_path = os.path.join(self.label_dir, gt_file)
        with open(gt_path) as f:
            lines = f.readlines()
        image_files = []
        current = None
        for i, line in enumerate(lines):
            if ImageFile.is_file_name(line):
                if current is not None:
                    image_files.append(current)
                image_path = os.path.join(self.root, f"WIDER_{subset}", "images", line)
                current = ImageFile(image_path)
            else:
                current.parse(line)
        image_files.append(current)
        return image_files

    def check_integrity(self):
        print("[WIDER] self checking data integrity...")
        for image_list in [self.train_images, self.valid_images]:
            count_box_error = 0
            count_file_error = 0
            for image_file in tqdm(image_list):
                result = image_file.check_integrity()
                if result == ImageFile.BOX_COUNT_ERROR:
                    count_box_error += 1
                if result == ImageFile.IMAGE_NOT_EXIST:
                    count_file_error += 1
            print(f"\t box number error:\t{count_box_error}")
            print(f"\t file not exist error:\t{count_file_error}")
        print("[WIDER] checking complete.\n")

    def filter(self):
        print(f"[WIDER] filtering out images that have faces which are too small (less than {self.min_face} pixels). ")
        buffer = []
        for item in tqdm(self.train_images):
            small = False
            for box in item.boxes:
                x0, y0, width, height = box
                if min(width, height) < self.min_face:
                    small = True
                    break
            if not small:
                buffer.append(item)
        self.train_images = buffer

        buffer = []
        for item in tqdm(self.valid_images):
            small = False
            for box in item.boxes:
                x0, y0, width, height = box
                if min(width, height) < self.min_face:
                    small = True
                    break
            if not small:
                buffer.append(item)
        self.valid_images = buffer
        print("[WIDER] filtering complete.\n")


def create_ssd_dataset(image_size, aspect_ratios):
    if os.environ.get('AIBOX') is None:  # 本地开发
        dataset_root = '/Users/ethan/datasets/WIDER/'
        record_dir = '/Users/ethan/datasets/WIDER_SSD/'
    else:  # 训练机
        dataset_root = '/media/ethan/DataStorage/WIDER/'
        record_dir = '/media/ethan/DataStorage/WIDER_SSD/'

    for f in ['train', 'valid']:
        os.makedirs(os.path.join(record_dir, f), exist_ok=True)

    print("[info] construct and initialize SSD model...")
    ssd = SSD(aspect_ratios=aspect_ratios)
    print("[info] complete.\n")

    print("[info] creating default boxes from model...")
    outputs = ssd.model.predict(np.random.uniform(-1, 1, size=(1, image_size, image_size, 3)).astype('float32'))
    default_boxes = outputs[2].squeeze()
    print(f'[info] complete, default boxes shape: {default_boxes.shape}\n')

    wider = WIDER(dataset_root)
    global_count = 0

    print("\n[info] start generating samples for training set...")
    for sample in tqdm(wider.train_images):
        images, labels = sample.generate_sub_images(image_size=image_size)
        if len(images) == 0:
            continue
        for i, image in enumerate(images):
            boxes = labels[i]  # boxes are in minmax mode
            # random flip horizontally
            image, boxes = random_flip(image, boxes, mode='minmax')
            boxes = convert_box_range(boxes, image.shape, mode='relative2absolute')
            score, delta = make_ssd_sample(image, boxes, default_boxes)
            filename = f'data{global_count:06d}.tfrecords'
            path = os.path.join(record_dir, 'train', filename)
            with tf.io.TFRecordWriter(path) as writer:
                serialized = create_tfrecord(image, score, delta)
                writer.write(serialized)
            global_count += 1

    print("\n[info] start generating samples for validation set...")
    global_count = 0
    for sample in tqdm(wider.valid_images):
        images, labels = sample.generate_sub_images(image_size=image_size)
        if len(images) == 0:
            continue
        for i, image in enumerate(images):
            boxes = labels[i]
            boxes = convert_box_range(boxes, image.shape, mode='relative2absolute')
            score, delta = make_ssd_sample(image, boxes, default_boxes)
            filename = f'data{global_count:06d}.tfrecords'
            path = os.path.join(record_dir, 'valid', filename)
            with tf.io.TFRecordWriter(path) as writer:
                serialized = create_tfrecord(image, score, delta)
                writer.write(serialized)
            global_count += 1


def load_dataset(batch_size):
    if os.environ.get('AIBOX') is None:  # 本地开发
        dataset_root = '/Users/ethan/datasets/WIDER_SSD/'
    else:  # 训练机
        dataset_root = '/media/ethan/DataStorage/WIDER_SSD/'
    print("[info] preparing training datasets...")
    train_dir = os.path.join(dataset_root, 'train')
    valid_dir = os.path.join(dataset_root, 'valid')
    train_images, valid_images = [glob(os.path.join(p, '*.tfrecords')) for p in [train_dir, valid_dir]]
    raw = tf.data.TFRecordDataset(train_images)
    trainset = raw.map(read_tfrecord)
    raw = tf.data.TFRecordDataset(valid_images)
    validset = raw.map(read_tfrecord)
    trainset = trainset.batch(batch_size).prefetch(AUTOTUNE)
    validset = validset.batch(batch_size).prefetch(AUTOTUNE)
    print(f'[info] datasets preparation complete, '
          f'training size: {len(train_images)}, validation size: {len(valid_images)}.\n'
          f'batch size: {batch_size}.\n')
    return trainset, validset


if __name__ == '__main__':
    image_size = 512
    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    create_ssd_dataset(image_size, aspect_ratios)
