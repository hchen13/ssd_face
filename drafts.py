import os
from datetime import datetime

import cv2
import numpy as np
from imutils import resize

from architecture.prototypes import SSD
from utils.helper_funcs import draw_boxes, show_image, convert_box_coordinates, non_max_suppression
from utils.preprocess import preprocess_image


ssd = SSD()
ssd.load_model("initial.h5")


def detect_faces(image, threshold=.5):
    input_image = preprocess_image(image)
    if input_image.ndim != 4:
        input_image = np.expand_dims(input_image, axis=0)
    conf_pred, offsets_pred, anchors, a1, a2, a3, a4, a5, a6 = ssd.model.predict(input_image)
    conf_pred = conf_pred.squeeze(axis=-1)
    conf_pred = conf_pred.squeeze(axis=0)
    offsets_pred = offsets_pred.squeeze(axis=0)
    anchors = anchors.squeeze(axis=0)
    box = np.empty_like(anchors)
    box[:, 0] = offsets_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
    box[:, 1] = offsets_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
    box[:, 2] = np.exp(offsets_pred[:, 2]) * anchors[:, 2]
    box[:, 3] = np.exp(offsets_pred[:, 3]) * anchors[:, 3]
    positive_indices = np.where(conf_pred > threshold)
    positive_boxes = box[positive_indices]
    scores = conf_pred[positive_indices]
    positive_anchors = anchors[positive_indices]

    # dbs = [a1, a2, a3, a4, a5, a6]
    # for a in dbs:
    #     print(a.shape)

    # for i, anchor in enumerate(positive_anchors):
    #     d = draw_boxes(image, [anchor], mode='centroid', color=(200, 1, 2))
    #     d = draw_boxes(d, [positive_boxes[i]], mode='centroid', color=(2, 200, 2))
    #     show_image(d, destroy=True)

    positive_boxes_minmax = convert_box_coordinates(positive_boxes, mode_from='centroid', mode_to='minmax')
    boxes, scores = non_max_suppression(scores, positive_boxes_minmax, method='min', threshold=.5)
    return boxes, scores

if __name__ == '__main__':
    path = os.path.join(os.path.expanduser('~'), 'Pictures/things/face_test0.jpg')
    image = cv2.imread(path)
    image = resize(image, height=500)
    tick = datetime.now()
    boxes, scores = detect_faces(image, threshold=.8)
    tock = datetime.now()
    print("prediction takes {}".format(tock - tick))

    # d = draw_boxes(image, boxes, mode='minmax', color=(1, 200, 1))
    # show_image(d, destroy=True)

    # from mtcnn import MTCNN
    # mtcnn = MTCNN()
    # detections = mtcnn.detect_faces(image)
    # print(detections)
    # d = draw_boxes(image, [detections[0]['box']], mode='corner')
    # show_image(d)