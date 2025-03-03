import numpy as np
from math import atan2, acos

####################################################################
#                                                                  #
# Miscellaneous functions that are not specific for the algorithm. #
#                                                                  #
####################################################################

REALLY_SMALL = 1e-6



def idx_extreme(iterable, find_min=True, key=lambda x: x,):
    extr_idx, extr_val = 0, key(iterable[0])
    for i in range(1, len(iterable)):
        to_compare = key(iterable)

        if (to_compare < extr_val) and find_min or \
           (to_compare > extr_val) and not find_min:

            extr_val = to_compare
            extr_idx = i

    return extr_idx

def idx_min(iterable, key=lambda x: x):
    return idx_extreme(iterable, True, key)

def idx_max(iterable, key=lambda x: x):
    return idx_extreme(iterable, False, key)



def mean(iterable, acc=0):
    for item in iterable:
        acc += item
    return acc / len(iterable)



def mode(values, tie_handling="mean"):
    """
    Finds the mode of given values. If there are multiple, return the one according to tie_handling:
    - \"mean\" -- return the one closer to mean.
    - \"min\" -- return maximum.
    - \"max\" -- return minimum.
    """
    unique = np.unique(values)
    # Make a list of (value, count) pairs and sort it by count.
    occurrencies = sorted([(val, np.count_nonzero(values == val)) for val in unique], key = lambda x: x[1], reverse=True)

    max_count = occurrencies[0][1]
    fin = 0
    for i, pair in enumerate(occurrencies):
        if pair[1] != max_count:
            break
        fin = i+1

    modes = occurrencies[:fin]
    if len(modes) == 1:
        return modes[0][0]
    else:
        if tie_handling == "mean":
            mean = np.mean(values)
            return min(modes, key = lambda x: abs(mean - x[0]))[0]
        elif tie_handling == "min":
            return min(modes)[0]
        elif tie_handling == "max":
            return max(modes)[0]



def adjacent(iterable, n_adjacent : int=2, cycle=True, center=True):
    """
    Yield tuples of adjacent items.
    params:
    - cycle: loops the iterable 
    - center: captures elements by both sides of every index instead of in front of it
    """
    if n_adjacent == 0:
        return None
    if n_adjacent == 1 or n_adjacent == -1:
        for item in iterable:
            yield item
    
    iter_size = len(iterable)
    idx_start, idx_end = 0, iter_size

    if cycle: # Capture elements before given index if n_adj is negative
        if n_adjacent < 0:
            idx_start -= n_adjacent
            idx_end -= n_adjacent
        if center:
            offset = int((n_adjacent - 1) / 2) if n_adjacent > 0 else -int((n_adjacent + 1) / 2)
            idx_start -= offset
            idx_end -= offset
    else:
        idx_end -= n_adjacent

    for i in range(idx_start, idx_end):
        yield tuple(iterable[(i + j) % iter_size] for j in range(n_adjacent))



def normalize(arr : np.ndarray) -> np.ndarray:
    return arr / np.linalg.norm(arr)

def cos_dist(vec1 : np.ndarray, vec2 : np.ndarray) -> float:
    return 1 - np.dot(normalize(vec1), normalize(vec2))

def get_angle(edge1: np.ndarray, vertex: np.ndarray, edge2: np.ndarray):
    vec1, vec2 = edge1 - vertex, edge2 - vertex
    angle_cos = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    return acos(angle_cos)



def clockwise_sort(points : list[np.ndarray]):
    """
    Sort points in-place in clockwise order with respect to centroid, starting from west:
        3
      2   4
    1---*   5
      8   6
        7
    """
    centroid = np.array([0.0, 0.0])
    for p in points:
        centroid += p
    centroid = centroid / len(points)
    points.sort(key=lambda p: -atan2((p - centroid)[1], (p - centroid)[0]))



def unique_pairs(set):
    """
    Iterate over a set of (n-1)n/2 unique pairs of given set's elements.
    """
    for i in range(len(set)-1):
        for j in range(i+1, len(set)):
            yield set[i], set[j]



def get_extreme_distance(points, minimum=True):
    """
    Returns minimum distance between two points in given set.
    """
    xtr_dist = None
    comparison = {False: lambda x, y: x > y, True: lambda x, y: x < y}
    # Consider all point pairs:
    for one, two in unique_pairs(points):
            distance = np.linalg.norm(one - two)
            if xtr_dist is None or comparison[minimum](distance, xtr_dist):
                xtr_dist = distance

    return xtr_dist