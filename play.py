import os
from datetime import datetime

import cv2
import numpy as np
from imutils import resize

from architecture.prototypes import SSD
from training.dataset.WIDER_data_generator import create_dataset_generator
from utils.helper_funcs import draw_boxes, show_image, non_max_suppression, convert_box_coordinates
from utils.preprocess import preprocess_image, deprocess_image


def detect_faces(ssd, image):
    input_image = preprocess_image(image)
    # input_image = image
    # image = deprocess_image(image)
    if input_image.ndim != 4:
        input_image = np.expand_dims(input_image, axis=0)
    conf_pred, offsets_pred, anchors = ssd.model.predict(input_image)
    conf_pred = conf_pred.squeeze(axis=-1)
    conf_pred = conf_pred.squeeze(axis=0)
    offsets_pred = offsets_pred.squeeze(axis=0)
    anchors = anchors.squeeze(axis=0)
    box = np.empty_like(anchors)
    box[:, 0] = offsets_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
    box[:, 1] = offsets_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
    box[:, 2] = np.exp(offsets_pred[:, 2]) * anchors[:, 2]
    box[:, 3] = np.exp(offsets_pred[:, 3]) * anchors[:, 3]
    positive_indices = np.where(conf_pred > .8)
    positive_boxes = box[positive_indices]
    scores = conf_pred[positive_indices]

    positive_boxes_minmax = convert_box_coordinates(positive_boxes, mode_from='centroid', mode_to='minmax')
    boxes, scores = non_max_suppression(scores, positive_boxes_minmax, method='union', threshold=.5)
    return boxes


if __name__ == '__main__':
    batch_size = 1

    if os.environ.get('AIBOX') is None:  # 本地开发
        dataset_root = '/Users/ethan/datasets/WIDER/'
    else:  # 训练机
        dataset_root = '/media/ethan/DataStorage/WIDER/'

    ssd = SSD()
    ssd.load_model("initial.h5")

    # validset = create_dataset_generator(dataset_root, 'valid', batch_size, shuffle=False, take=50)
    # for batch in validset:
    #     images, conf_gt, offsets_gt = batch
    #     image = deprocess_image(images.numpy()[0])
    #     boxes, scores = ssd.detect_faces(image, threshold=.8)
    #     d = draw_boxes(image, boxes, mode='minmax', color=(1, 200, 1))
    #     show_image(d, destroy=True)

    for i in range(3):
        path = os.path.join(os.path.expanduser('~'), 'Pictures/things/face_test{}.jpg'.format(i))
        image = cv2.imread(path)
        # image = cv2.resize(image, (400, 400))
        image = resize(image, height=500)
        tick = datetime.now()
        boxes, scores = ssd.detect_faces(image, threshold=.7)
        tock = datetime.now()
        print("prediction takes {}".format(tock - tick))
        d = draw_boxes(image, boxes, mode='minmax', color=(1, 200, 1))
        show_image(d, destroy=True)
