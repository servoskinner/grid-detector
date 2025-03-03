import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi
from random import random, randint

def map_perspective(vertices : dict, x0 = 0, y0 = 0) -> np.ndarray:
    """
    Perspective maps a point by its relative coordinates.
    x, y must be within [1; 0] interval;
    x sets "left-right" positioning and
    y sets "top-bottom" positioning

    vertices: dict<str : np.array>
        tl - top left
        tr - top right
        bl - bottom left
        br - bottom right 

    the perspective matrix follows:
        [[Xx, Xy, X0],
         [Yx, Yy, Y0],
         [Wx, Wy, W0]]
    the transform is applied by multiplying it by
        [x, y, 1].T
    then dividing the resulting x', y' by third coordinate w.
    """
    return (1 - y0 - x0 + x0*y0)*vertices["tl"] + (x0 - x0*y0)*vertices["tr"] + (y0 - x0*y0)*vertices["bl"] + x0*y0*vertices["br"]



def map_points(vertices : dict, shape : tuple) -> list:
    """
    Generates a set of perspective warped grid vertices with set width and height.

    vertices: dict<str : np.array>
        tl - top left
        tr - top right
        bl - bottom left
        br - bottom right 
    """
    points = []

    if shape[0] == 1:
        if shape[1] == 1:
            return vertices["tl"]
        else:
            return [[map_perspective(vertices, x0 = i/(shape[1]-1), y0 = 0)] for i in range(shape[1])] 
    else:
        if shape[1] == 1:
            return [[map_perspective(vertices, x0 = 0, y0 = i/(shape[0]-1)) for i in range(shape[0])]]
        else:
            for i in range(shape[0]):
                layer = list(map_perspective(vertices, x0 = j/(shape[1]-1), y0 = i/(shape[0]-1)) for j in range(shape[1]))
                points.append(layer)
            return points



def rotate_point(point, center, angle):
    translated_point = point - center
    
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    rotated_point = np.matmul(rotation_matrix, translated_point) + center
    return rotated_point



def random_rotate_points(points, center, angle_range=(-pi/6, pi/6)):
    rotated_points = []

    angle = np.random.uniform(angle_range[0], angle_range[1])
    rotation_matrix = np.array([[cos(angle), -sin(angle)],
                              [sin(angle), cos(angle)]])
    for point in points:
        translated_point = point - center
        rotated_point = np.matmul(rotation_matrix, translated_point) + center
        rotated_points.append(rotated_point)
    return np.array(rotated_points)

def generate_grid():

    MIN_SIZE = 4
    MAX_SIZE = 10
    CORNER_MARGIN = 0.125
    NOISE = 0.05
    PROB_MISSING = 0.33
    MAX_MISSING = 24
    ROTATE_DEG = 40

    vertices = {"tl": np.array([random()*CORNER_MARGIN, random()*CORNER_MARGIN]),
                        "tr": np.array([(1-CORNER_MARGIN)+random()*CORNER_MARGIN, random()*CORNER_MARGIN]),
                        "bl": np.array([random()*CORNER_MARGIN, (1-CORNER_MARGIN)+random()*CORNER_MARGIN]),
                        "br": np.array([(1-CORNER_MARGIN)+random()*CORNER_MARGIN, (1-CORNER_MARGIN)+random()*CORNER_MARGIN]),
               }
    width, height = randint(MIN_SIZE, MAX_SIZE), randint(MIN_SIZE, MAX_SIZE) #randint(MIN_SIZE, MAX_SIZE)

    pts = map_points(vertices, (width, height))
    pts_flattened = []
    for row in pts:
        pts_flattened += row
    pts = pts_flattened

    if NOISE > 1e-5:
        for j, _ in enumerate(pts):
            pts[j] += (np.random.normal(0, NOISE, 2) / np.array([height, width], dtype=np.float64))
    if PROB_MISSING != 0:
        for i in range(MAX_MISSING):
            roll = random()
            if roll >= PROB_MISSING:
                break
            exclude = randint(0, len(pts)-1)
            pts.pop(exclude)

    rotate_rad = pi * ROTATE_DEG / 180
    pts = random_rotate_points(pts, np.array([0.5, 0.5]), (-rotate_rad, rotate_rad))

    return pts, np.array([height, width], dtype=np.int32)