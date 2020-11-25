import os
from datetime import datetime

from ethan_toolbox import draw_boxes, show_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf

from prototype import pvanet, original

if __name__ == '__main__':
    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    h = os.path.expanduser('~')
    image_path = os.path.join(h, 'Pictures/things/face_test2.jpg')
    image = cv2.imread(image_path)

    ssd = pvanet.SSD(aspect_ratios=aspect_ratios)
    ssd.model.load_weights('weights/ssd_pvanet.h5', by_name=True)

    # ssd = original.SSD(aspect_ratios=aspect_ratios)
    # ssd.model.load_weights('weights/ssd_vgg16.h5', by_name=True)

    boxes, scores = ssd.detect(image, threshold=.5)

    n = 20
    tick = datetime.now()
    for _ in range(n):
        boxes, scores = ssd.detect(image, threshold=.5)
    tock = datetime.now()
    elapsed = (tock - tick) / n
    print(f"time consumed: {elapsed.total_seconds():.2f} seconds")
    # disp = draw_boxes(image, boxes, mode='minmax')
    # show_image(disp, width=800)

