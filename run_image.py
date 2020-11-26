import cv2
from ethan_toolbox import draw_boxes, show_image, convert_box_coordinates

from prototype import original, pvanet

if __name__ == '__main__':
    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    m1 = pvanet.SSD(aspect_ratios)
    m1.model.load_weights('weights/ssd_pvanet.h5', by_name=True)

    m2 = original.SSD(aspect_ratios)
    m2.model.load_weights('weights/ssd_vgg16.h5', by_name=True)

    image = cv2.imread('/Users/ethan/Pictures/things/face_test2.jpg')
    b1, _ = m1.detect(image)

    b2, _ = m2.detect(image)

    disp = draw_boxes(image, b1, mode='minmax', thickness=2, color=(10, 20, 200))
    disp = draw_boxes(disp, b2, mode='minmax', thickness=2)
    show_image(disp, width=1000)