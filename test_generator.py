import numpy as np
from math import sin, cos, pi
from random import random, randint

#######################################################
#                                                     #
# Generates test cases for the grid detection module. #
#                                                     #
#######################################################

def map_perspective(vertices : dict, x0 = 0, y0 = 0) -> np.ndarray:
    """
    Perspective maps a point by bilinear interpolating using the base 4 vertices.

    x, y must be within [1; 0] interval;
    x sets "left-right" positioning and
    y sets "top-bottom" positioning

    vertices: dict<str, np.array>
        tl - top left
        tr - top right
        bl - bottom left
        br - bottom right 
    """
    return (1 - y0 - x0 + x0*y0)*vertices["tl"] + (x0 - x0*y0)*vertices["tr"] + (y0 - x0*y0)*vertices["bl"] + x0*y0*vertices["br"]



def map_grid(vertices : dict, shape : tuple) -> list:
    """
    Generates a set of perspective warped grid vertices with set width and height.

    vertices: dict<str : np.array>
        tl - top left
        tr - top right
        bl - bottom left
        br - bottom right 
    """
    points = []
    max_spacing = max(shape) - 1

    if shape[0] == 1:
        if shape[1] == 1:
            return [[0.5, 0.5]]
        else:
            return [[map_perspective(vertices, x0 = 0.5 + (i - max_spacing/2.0)/max_spacing, y0 = 0.5)] for i in range(shape[1])]
    else:
        if shape[1] == 1:
            return [[map_perspective(vertices, 
                                     x0 = 0.5, 
                                     y0 = 0.5 + (i - max_spacing/2.0)/max_spacing) for i in range(shape[0])]]
        else:
            for i in range(shape[0]):
                layer = list(map_perspective(vertices, 
                                             x0 = 0.5 + (j - (shape[1] - 1)/2.0)/max_spacing, 
                                             y0 = 0.5 + (i - (shape[0] - 1)/2.0)/max_spacing) for j in range(shape[1]))
                points.append(layer)
            return points



def rotate_point(point, center, angle):
    translated_point = point - center
    
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    rotated_point = np.matmul(rotation_matrix, translated_point) + center
    return rotated_point



def rotate_points(points, center, angle):
    rotated_points = []
    rotation_matrix = np.array([[cos(angle), -sin(angle)],
                              [sin(angle), cos(angle)]])
    for point in points:
        translated_point = point - center
        rotated_point = np.matmul(rotation_matrix, translated_point) + center
        rotated_points.append(rotated_point)
    return np.array(rotated_points)


# Returns a random grid and its dimensions.
def generate_grid():

    MIN_SIZE = 4    # Lower bound on each grid dim (inclusive).  Should be not less than 2
    MAX_SIZE = 10   # Upper bound on each grid dim (also inclusive)

    CORNER_MARGIN = 0.125   # [0.0; 0.5): regulates perspective warp strength.
    NOISE = 0.05            # Regulates gaussian noise. it's also divided by 
    PROB_LINE = 0.1         # Chance of generating a 1d grid (noise has to be handled in a different way)
    PROB_MISSING = 0.02     # Chance that a random point gets excluded. Derived from detector accuracy
    ROTATE_DEG = 40         # Random rotation, should not exceed 45 to avoid false error due to confusing vertical and horizontal axes
    MIN_SCALE = 0.7         # One scale is selected at random and multiplied by random scale from MIN_SCALE to 1.0

    vertices = {"tl": np.array([random()*CORNER_MARGIN, random()*CORNER_MARGIN]),
                        "tr": np.array([(1-CORNER_MARGIN)+random()*CORNER_MARGIN, random()*CORNER_MARGIN]),
                        "bl": np.array([random()*CORNER_MARGIN, (1-CORNER_MARGIN)+random()*CORNER_MARGIN]),
                        "br": np.array([(1-CORNER_MARGIN)+random()*CORNER_MARGIN, (1-CORNER_MARGIN)+random()*CORNER_MARGIN]),
               }
    
    generate_line = random() < PROB_LINE
    vertical = random() < 0.5

    height, width = randint(MIN_SIZE, MAX_SIZE), randint(MIN_SIZE, MAX_SIZE)

    rand_scale = MIN_SCALE + random() * (1 - MIN_SCALE)
    stretch_x = random() < 0.5

    max_dim = max(height, width)

    if generate_line:
        if vertical:
            width = 1
        else:
            height = 1

    pts = map_grid(vertices, (height, width))
    pts_flattened = []
    for row in pts:
        pts_flattened += row
    pts = pts_flattened

    # Add noise
    if NOISE > 1e-5:
        for j, _ in enumerate(pts):
            pts[j] += (np.random.normal(0, NOISE, 2) / max_dim)

    # Simulate false negatives
    if PROB_MISSING != 0 and not generate_line:
        rand = [random() for _ in pts]
        n_pts = len(pts)
        for i in range(0, n_pts): # Deleting from behind to account for index shifts
            if rand[i] < PROB_MISSING:
                pts.pop(-(i+1) % n_pts)

    # Scale grid down over one of axes
    for p in pts:
        p[:] *= np.array([rand_scale, 1.0]) if stretch_x else np.array([1.0, rand_scale])

    # Apply random rotation
    rotate_rad = pi * ROTATE_DEG / 180
    angle = np.random.uniform(-rotate_rad, rotate_rad)
    pts = rotate_points(pts, np.array([0.5, 0.5]), angle)

    return pts, np.array([width, height], dtype=np.int32)