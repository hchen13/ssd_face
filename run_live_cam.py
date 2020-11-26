import time
from datetime import datetime
import numpy as np

import cv2
from ethan_toolbox import draw_boxes
from imutils import resize
from imutils.video import VideoStream

from prototype import pvanet

if __name__ == '__main__':
    print('[info] initializing...')
    vs = VideoStream().start()
    time.sleep(2.)
    ssd = pvanet.SSD(aspect_ratios=[1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3])
    ssd.model.load_weights('weights/ssd_pvanet.h5', by_name=True)

    print("[info] start streaming...")
    while True:
        t0 = datetime.now()
        frame = vs.read()
        frame_resized = resize(frame, width=1000)

        boxes, scores = ssd.detect(frame)
        disp = draw_boxes(frame_resized, boxes, mode='minmax', thickness=2, color=(10, 200, 19))
        disp = np.array(disp[:, ::-1, :])

        t1 = datetime.now()
        delta = (t1 - t0).total_seconds()
        fps = 1 / delta
        cv2.putText(disp, f"FPS {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, .7, (10, 200, 19), thickness=1)

        cv2.imshow('live', disp)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
