import cv2
import numpy as np


def draw_boxes(image, boxes, mode, color=(255, 255, 255), thickness=1):
    if len(boxes) == 0:
        return image
    available_modes = ['corner', 'centroid', 'minmax']
    if mode not in available_modes:
        raise TypeError("[draw_boxes] unknown mode: {}, has to be `corner` or `centroid`".format(mode, available_modes))
    canvas = image.copy()

    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)

    absolute_box = False if boxes.max() - boxes.min() <= 10. else True

    if not absolute_box:
        boxes = convert_box_range(boxes, image.shape, mode='relative2absolute')

    if mode != 'minmax':
        boxes = convert_box_coordinates(boxes, mode_from=mode, mode_to='minmax')

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box[:4]
        x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)
    return canvas


def show_image(*images, width=None, col=None, wait=0, title=None, destroy=False):
    import imutils

    if title is None:
        title = 'image'
    if len(images) == 1:
        image = images[0]
        if width is not None:
            image = imutils.resize(image, width=width)
        cv2.imshow(title, image)
        key = cv2.waitKey(wait) & 0xff
        if destroy:
            cv2.destroyAllWindows()
        return

    if width is None:
        width = 800
    if col is None:
        col = len(images)
    row = np.math.ceil(len(images) / col)
    _width = int(width / col)

    montages = imutils.build_montages(images, (_width, _width), (col, row))
    for montage in montages:
        cv2.imshow(title, montage)
        cv2.waitKey(wait)
        if destroy:
            cv2.destroyAllWindows()


def convert_box_coordinates(bounding_boxes, mode_from, mode_to):
    """
    converting bounding box modes back and forth

    :param bounding_boxes: expect the bounding boxes are represented as (#boxes, 4) 2D numpy arrays
    :param mode_from: original mode
    :param mode_to: expected output mode
    :return: converted boxes represented in the same way as the input
    """
    available_modes = ['corner', 'centroid', 'minmax']
    if mode_from not in available_modes or mode_to not in available_modes:
        raise TypeError("[convert_box_coordinates] unknown mode, has to be in {}".format(available_modes))

    if not isinstance(bounding_boxes, np.ndarray):
        boxes = np.array(bounding_boxes)
    else:
        boxes = bounding_boxes.copy()
    dtype = boxes.dtype
    boxes = boxes.astype('float')

    if mode_from == 'corner' and mode_to == 'centroid':
        boxes[:, 0] += boxes[:, 2] / 2
        boxes[:, 1] += boxes[:, 3] / 2

    if mode_from == 'corner' and mode_to == 'minmax':
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    if mode_from == 'centroid' and mode_to == 'minmax':
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    if mode_from == 'centroid' and mode_to == 'corner':
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2

    if mode_from == 'minmax' and mode_to == 'corner':
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]

    if mode_from == 'minmax' and mode_to == 'centroid':
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] += boxes[:, 2] / 2
        boxes[:, 1] += boxes[:, 3] / 2

    return boxes.astype(dtype)


def convert_box_range(bounding_boxes, image_shape, mode):
    available_modes = ["absolute2relative", 'relative2absolute']
    if mode not in available_modes:
        raise TypeError("[convert_box_range] mode has to be in {}".format(available_modes))

    height, width = image_shape[:2]
    boxes = bounding_boxes.copy().astype('float')
    if mode == 'absolute2relative':
        boxes[:, 0] = boxes[:, 0] / width
        boxes[:, 2] = boxes[:, 2] / width
        boxes[:, 1] = boxes[:, 1] / height
        boxes[:, 3] = boxes[:, 3] / height
    else:
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        boxes = np.floor(boxes).astype('int')
    return boxes


def crop_from_boxes(image, boxes, image_size):
    height, width, channels = image.shape
    patches = np.empty([len(boxes), image_size, image_size, channels], dtype=image.dtype)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = list(map(int, box[:4]))
        patch = image[max(0, y0): min(y1, height), max(0, x0): min(x1, width)]
        patch = cv2.resize(patch, (image_size, image_size), interpolation=cv2.INTER_AREA)
        patches[i] = patch
    return patches


def crop_image(image, crop_box, box_mode):
    height, width, _ = image.shape
    assert crop_box.ndim == 1
    crop_box = np.expand_dims(crop_box, axis=0)
    if box_mode != 'minmax':
        crop_box = convert_box_coordinates(crop_box, mode_from=box_mode, mode_to='minmax')
    if crop_box.dtype != np.int:
        crop_box = convert_box_range(crop_box, image.shape, 'relative2absolute')
    x0, y0, x1, y1 = crop_box[0]
    patch = image[max(0, y0): min(y1, height), max(0, x0): min(x1, width)]
    return patch


def is_box_inside(inner_box, outer_box):
    delta = outer_box - inner_box
    return delta[0] < 0 and delta[1] < 0 and delta[2] > 0 and delta[3] > 0

def calculate_overlap_ratio(boxes1, boxes2, method='union'):
    if method not in ['union', 'min']:
        raise TypeError("[calculate_overlap_ratio] method `{}` not supported".format(method))

    b1_areas = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    b2_areas = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x0 = np.maximum.outer(boxes1[:, 0], boxes2[:, 0])
    inter_y0 = np.maximum.outer(boxes1[:, 1], boxes2[:, 1])
    inter_x1 = np.minimum.outer(boxes1[:, 2], boxes2[:, 2])
    inter_y1 = np.minimum.outer(boxes1[:, 3], boxes2[:, 3])

    widths = np.maximum(0., inter_x1 - inter_x0)
    heights = np.maximum(0., inter_y1 - inter_y0)
    inter_areas = np.multiply(widths, heights)

    if method == 'union':
        d1 = np.tile(np.expand_dims(b1_areas, axis=-1), (1, boxes2.shape[0]))
        d2 = np.tile(np.expand_dims(b2_areas, axis=0), (boxes1.shape[0], 1))
        overlaps = inter_areas / (d1 + d2 - inter_areas)
    else:
        overlaps = inter_areas / np.minimum.outer(b1_areas, b2_areas)
    return overlaps


def random_flip(image, bounding_boxes, mode, prob=.5):
    if np.random.rand() > prob:
        return image, bounding_boxes
    return horizontal_flip(image, bounding_boxes, mode)


def horizontal_flip(image, bounding_boxes, mode):
    if mode != 'centroid':
        bounding_boxes = convert_box_coordinates(bounding_boxes, mode_from=mode, mode_to='centroid')
    bounding_boxes[:, 0] = 1 - bounding_boxes[:, 0]
    bounding_boxes = convert_box_coordinates(bounding_boxes, mode_from='centroid', mode_to=mode)
    image[:, :, :] = image[:, ::-1, :]
    return image, bounding_boxes


def non_max_suppression(scores, bounding_boxes, threshold=.5, method='union'):
    if method not in ['union', 'min']:
        raise TypeError("[non_max_suppression] method `{}` not supported".format(method))
    x0, y0 = bounding_boxes[:, 0], bounding_boxes[:, 1]
    x1, y1 = bounding_boxes[:, 2], bounding_boxes[:, 3]
    area = (x1 - x0) * (y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        top = order[0]
        keep.append(top)
        inter_x0 = np.maximum(x0[top], x0[order[1:]])
        inter_y0 = np.maximum(y0[top], y0[order[1:]])
        inter_x1 = np.minimum(x1[top], x1[order[1:]])
        inter_y1 = np.minimum(y1[top], y1[order[1:]])
        inter_area = np.maximum(0., inter_x1 - inter_x0) * np.maximum(0., inter_y1 - inter_y0)
        if method == 'union':
            overlap = inter_area / (area[top] + area[order[1:]] - inter_area)
        else:
            overlap = inter_area / np.minimum(area[top], area[order[1:]])
        indices = np.where(overlap <= threshold)[0]
        order = order[indices + 1]
    return bounding_boxes[keep], scores[keep]
