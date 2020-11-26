from datetime import datetime

import cv2
from ethan_toolbox import draw_boxes
from imutils import resize
import numpy as np

from prototype import pvanet

if __name__ == '__main__':
    path = '/Users/ethan/Movies/test.mkv'
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ssd = pvanet.SSD(aspect_ratios=[1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3])
    ssd.model.load_weights('weights/ssd_pvanet.h5', by_name=True)

    for _ in range(total_frames):
        t0 = datetime.now()
        success, frame = cap.read()
        if not success:
            break
        frame_resized = resize(frame, width=1000)

        boxes, scores = ssd.detect(frame)
        disp = draw_boxes(frame_resized, boxes, mode='minmax', thickness=2, color=(10, 200, 19))

        t1 = datetime.now()
        delta = (t1 - t0).total_seconds()
        fps = 1 / delta
        cv2.putText(disp, f"FPS {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, .7, (10, 200, 19), thickness=1)

        cv2.imshow('live', disp)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

