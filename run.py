import cv2
from ethan_toolbox import draw_boxes, show_image, convert_box_coordinates

from prototype import original

if __name__ == '__main__':
    aspect_ratios = [1., 1 / 2, 2, 2 / 3, 3 / 2, 3 / 4, 4 / 3]
    ssd = original.SSD(aspect_ratios)
    ssd.model.load_weights('weights/ssd_vgg16.h5')

    image = cv2.imread('/Users/ethan/Pictures/things/face_test2.jpg')
    boxes, scores = ssd.detect(image)
    disp = draw_boxes(image, boxes, mode='minmax')
    show_image(disp, width=800)