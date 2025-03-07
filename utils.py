import numpy as np
from math import atan2, acos, pi

####################################################################
#                                                                  #
# Miscellaneous functions that are not specific for the algorithm. #
#                                                                  #
####################################################################

REALLY_SMALL = 1e-6


# Generic orthogonal basis (not necessarily normalized)
class Basis_ortho:
    def __init__(self):
        self.origin = np.zeros(2)

        self.basis_u = np.array([1, 0])
        self.basis_v = np.array([0, 1])

    def to_local(self, point : np.ndarray) -> np.ndarray:
        """
        Map global point to local coords.
        """
        proj_mat = np.linalg.inv(np.column_stack([self.basis_u, self.basis_v]))
        return np.matmul(proj_mat, point - self.origin)

    def to_global(self, coords : np.ndarray) -> np.ndarray:
        """
        Map local point to global coords.
        """
        return self.origin + self.basis_u * coords[0] + self.basis_v * coords[1]



def idx_extreme(iterable, find_min=True, key=lambda x: x,):
    extr_idx, extr_val = 0, key(iterable[0])
    for i in range(1, len(iterable)):
        to_compare = key(iterable[i])

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
        idx_end -= n_adjacent - 1

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
    Sort points in-place in clockwise order, starting with one with min abs(x + y):
        2
      1   3
    8   *   4
      7   5
        6
    """
    centroid = np.array([0.0, 0.0])
    for p in points:
        centroid += p
    centroid = centroid / len(points)

    points.sort(key=lambda p: -atan2((p - centroid)[1], (p - centroid)[0]))
    first_idx = idx_min(points, key=lambda p: p[0] - p[1])
    points[:] = points[first_idx:] + points[:first_idx]



def unique_pairs(set):
    """
    Iterate over a set of (n-1)n/2 unique pairs of given set's elements.
    """
    for i in range(len(set)-1):
        for j in range(i+1, len(set)):
            yield set[i], set[j]



import numpy as np
import math

def min_distance(points):
    """
    Finds minimum distance between points in given set with n*log(n) time complexity.
    """
    points_sorted_x = sorted(points, key=lambda x: x[0])
    
    # Recursively find min distances between ordered subsets of points.
    def subset_min_distance(subset):
        num_points = len(subset)
        if num_points <= 3:
            min_dist = None
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    dist = np.linalg.norm(subset[i] - subset[j])
                    if min_dist == None or dist < min_dist:
                        min_dist = dist
            return min_dist
        
        mid_idx = num_points // 2
        mid_point = subset[mid_idx]
        
        min_left = subset_min_distance(subset[:mid_idx])
        min_right = subset_min_distance(subset[mid_idx:])
        min_subset = min(min_left, min_right)
        
        # Check points adjacent to border
        boundary = [point for point in subset if abs(point[0] - mid_point[0]) < min_subset]
        
        strip_sorted_y = sorted(boundary, key=lambda x: x[1])
        
        min_boundary = min_subset
        for i in range(len(strip_sorted_y)):
            for j in range(i + 1, len(strip_sorted_y)):
                if (strip_sorted_y[j][1] - strip_sorted_y[i][1]) >= min_boundary:
                    break
                dist = np.linalg.norm(strip_sorted_y[i] - strip_sorted_y[j])
                if dist < min_boundary:
                    min_boundary = dist
        
        return min(min_subset, min_boundary)
    
    return subset_min_distance(points_sorted_x)



def get_principal_axes(points : list[np.ndarray]) -> Basis_ortho | None:
    """
    Get mass center of given set of points and orthogonal axes with most and least variance.
    """
    centroid = np.array([0.0, 0.0])
    for p in points:
        centroid += p
    centroid = centroid / len(points)
    centered_pts = np.array(points) - centroid

    cov_mat = np.cov(centered_pts.transpose())
    D_sqrt = np.sqrt((cov_mat[0][0] - cov_mat[1][1])**2 + 4*cov_mat[0][1]**2)

    eigenvalue1 = ((cov_mat[0][0] + cov_mat[1][1]) + D_sqrt) / 2
    eigenvalue2 = ((cov_mat[0][0] + cov_mat[1][1]) - D_sqrt) / 2

    major_value = eigenvalue1 if abs(eigenvalue1) > abs(eigenvalue2) else eigenvalue2
    if major_value == 0:
        return None
    
    major_axis, minor_axis = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    
    if eigenvalue1 != eigenvalue2: # If they are equal, the axes can be chosen arbitrarily (think of a circle instead of ellipsoid)
        major_axis = normalize(np.array([cov_mat[0][1], -(cov_mat[0][0] - major_value)]))
        minor_axis = np.array([-major_axis[1], major_axis[0]])

    axes = Basis_ortho()
    axes.basis_u = major_axis
    axes.basis_v = minor_axis
    axes.origin = centroid

    return axes
