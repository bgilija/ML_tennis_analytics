import math


def get_center_of_bounding_box(bounding_box):
    x1, y1, x2, y2 = bounding_box[:4]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_foot_position(bounding_box):
    x1, y1, x2, y2 = bounding_box[:4]
    return (int((x1 + x2) / 2), int(y2))


def get_height_of_bounding_box(bounding_box):
    return bounding_box[3] - bounding_box[1]


def measure_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
