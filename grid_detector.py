import numpy as np
import copy
from scipy.spatial import ConvexHull
from math import atan2, pi

from utils import normalize, get_angle, adjacent, clockwise_sort, get_extreme_distance, mode
from plotters import dump_points

#################################################
#                                               #
# Module for detecting grid on a set of points. #
#                                               #
#################################################



class Grid:
    def __init__(self):
        self.offset = np.zeros(2)
        self.dims = np.ones(2, dtype=np.int32)

        self.basis_u = np.array([1, 0])
        self.basis_v = np.array([0, 1])

        self.proj_u = np.array([1, 0])
        self.proj_v = np.array([0, 1])

    def to_local(self, point : np.ndarray) -> np.ndarray:
        """
        Map global point to local integer coords.
        """
        pos_u = np.dot(self.proj_u, point - self.offset)
        pos_v = np.dot(self.proj_v, point - self.offset)

        return np.round(np.array([pos_u, pos_v])).astype(np.int32)

    def to_global(self, coords : np.ndarray) -> np.ndarray:
        """
        Map a grid cell's row and column to global coords.
        """
        return self.offset + self.basis_u * coords[0] + self.basis_v * coords[1]



class Principal_axes:
    def __init__(self):
        self.origin = np.zeros(2)
        self.major_axis = np.zeros(2)
        self.minor_axis = np.zeros(2)

    def to_local(self, point : np.ndarray) -> np.ndarray:
        proj_mat = np.linalg.inv(np.column_stack([self.major_axis, self.minor_axis]))
        return np.matmul(proj_mat, point)



class Bounding_polygon:
    def __init__(self):
        self.top_left = np.zeros(2)
        self.top_right = np.zeros(2)
        self.bottom_right = np.zeros(2)
        self.bottom_left = np.zeros(2)



def get_principal_axes(points : list[np.ndarray]) -> Principal_axes | None:
    """
    Get center of given set of points and orthogonal axes with most and least variance.
    """
    centroid = np.array([0.0, 0.0])
    for p in points:
        centroid += p
    centroid = centroid / len(points)
    centered_pts = np.array(points) - centroid

    cov_mat = np.cov(centered_pts.transpose())
    D_sqrt = np.sqrt((cov_mat[0][0] - cov_mat[1][1])**2 + 4*cov_mat[0][1]**2)

    eigenvalue1 = ((cov_mat[0][0] + cov_mat[1][1]) - D_sqrt) / 2
    eigenvalue2 = ((cov_mat[0][0] + cov_mat[1][1]) - D_sqrt) / 2

    major_value = eigenvalue1 if abs(eigenvalue1) > abs(eigenvalue2) else eigenvalue2
    if eigenvalue1 == 0:
        return None

    major_axis = normalize(np.array([cov_mat[0][1], -(cov_mat[0][0] - major_value)]))
    minor_axis = np.array([-major_axis[1], major_axis[0]])

    axes = Principal_axes()
    axes.major_axis = major_axis
    axes.minor_axis = minor_axis
    axes.origin = centroid

    return axes



def get_bounding_polygon(points : list[np.ndarray],
                         acuteness_thr = 5*pi/6) -> Bounding_polygon | None:
    """
    Build a 4-point (somewhat) convex hull.
    """

    convex_hull = ConvexHull(points)
    hull_points = [points[i] for i in convex_hull.vertices]

    indexed_angles = [(i, get_angle(*angle_pts)) for i, angle_pts in enumerate(adjacent(hull_points, 3))]
    acute_angles = [angle for angle in indexed_angles if angle[1] < acuteness_thr]
    hull_points = [hull_points[a[0]] for a in acute_angles]

    def regenerate_angles():
        nonlocal acute_angles
        clockwise_sort(hull_points)
        acute_angles = [(i, get_angle(*angle_pts)) for i, angle_pts in enumerate(adjacent(hull_points, 3))]
        acute_angles.sort(key=lambda a: a[1])

    regenerate_angles()

    if len(acute_angles) < 2: # irreparable
        return None

    elif len(acute_angles) == 3: # triangle; symmetrize the least acute vertex against the line formed by remaining two
        obtuse_vertex = hull_points[acute_angles[-1][0]]
        acute_vertex_1 = hull_points[acute_angles[0][0]]
        acute_vertex_2 = hull_points[acute_angles[1][0]]

        hull_points.append(acute_vertex_1 + acute_vertex_2 - obtuse_vertex)
        regenerate_angles()

    elif len(acute_angles) == 4: # no repair needed.
        pass

    elif 4 < len(acute_angles) < 9:
        # Find the least acute vertex, pair it with closest
        # neighbor, then merge them by extending 
        # adjacent edges to form a new vertex.
        while len(hull_points) > 4:

            vertex_id = acute_angles[-1][0]

            vertex = hull_points[vertex_id]
            left_neighbor = hull_points[(vertex_id - 1) % len(hull_points)]
            right_neighbor = hull_points[(vertex_id + 1) % len(hull_points)]
            # assuming left neighbor is closer
            neighbor_id, far_edge_id, near_edge_id = None, None, None

            if np.linalg.norm(vertex - right_neighbor) < \
               np.linalg.norm(vertex - left_neighbor):
                
                neighbor_id = (vertex_id + 1) % len(hull_points)
                far_edge_id = (vertex_id + 2) % len(hull_points)
                near_edge_id = (vertex_id - 1) % len(hull_points)
            else:
                neighbor_id = (vertex_id - 1) % len(hull_points)
                far_edge_id = (vertex_id - 2) % len(hull_points)
                near_edge_id = (vertex_id + 1) % len(hull_points)
            # Next code finds the intersection point.
            # It's better to draw a diagram on paper that explain it with words
            near_vector = vertex - hull_points[near_edge_id]
            far_vector = hull_points[neighbor_id] - hull_points[far_edge_id]
            neighbor_vector = hull_points[neighbor_id] - vertex

            proj_matrix = np.linalg.inv(np.column_stack([near_vector, far_vector]))
            decomposed = np.matmul(proj_matrix, neighbor_vector)

            new_point = vertex + decomposed[0] * near_vector

            hull_points.pop(vertex_id)
            if vertex_id > neighbor_id:
                hull_points.pop(neighbor_id)
            else:
                hull_points.pop(neighbor_id - 1) # Adjust for shifting indices

            hull_points.append(new_point)
            regenerate_angles()

    else: # >8, irreparable.
        return None
    
    # Points are sorted as iteration step of repair regenerates angles.
    polygon = Bounding_polygon()

    polygon.top_left = hull_points[0]
    polygon.top_right = hull_points[1]
    polygon.bottom_right = hull_points[2]
    polygon.bottom_left = hull_points[3]
    return polygon

        

def detect_grid(points : list[np.ndarray],
                acuteness_thr = 5*pi/6,
                line_sideways_deviation_thr = 0.33,
                line_spacing_deviation_thr = 0.25,
                bin_width = 0.33,
                cell_xy_ratio_estimate = 1.0) -> Grid | None:
    
    if len(points) == 0: # Trivial cases
        return None
    if len(points) == 1:
        grid = Grid()
        grid.offset = points[0]
        return grid

    # Check if points belong to a line (asserting there are >=2)
    principal_axes = get_principal_axes(points)
    pa_points = [principal_axes.to_local(p) for p in points]

    # Sort by major axis
    pa_points.sort(key=lambda p: p[0])

    min_major_spacing = min([pair[1][0] - pair[0][0] for pair in adjacent(pa_points, cycle=False, center=False)])
    max_major_spacing = max([pair[1][0] - pair[0][0] for pair in adjacent(pa_points, cycle=False, center=False)])
    max_minor_deviation = abs(max(pa_points, key=lambda p: abs(p[1]))[1])

    if min_major_spacing * line_sideways_deviation_thr >= max_minor_deviation:
        # Possibly a line structure, need to check spacing evenness
        if max_major_spacing / min_major_spacing - 1 > line_spacing_deviation_thr:
            return None # Unevenly spaced
        else:
            direction = atan2(principal_axes.major_axis[1], principal_axes.major_axis[0])
            mean_index = (len(points) - 1) / 2
            index_variance = mean_index**2 /3 
            # Mean of major coordinates is 0
            major_coordinates = [p[0] for p in pa_points]

            coords_index_correlation = 0
            for i, u in enumerate(major_coordinates):
                coords_index_correlation += (i - mean_index) * u

            spacing = coords_index_correlation / len(points) / index_variance
            flat_offset = -spacing * mean_index

            principal_axes.major_axis *= spacing
            principal_axes.minor_axis *= spacing
            principal_axes.offset += principal_axes.major_axis * flat_offset

            proj_matrix = np.linalg.inv(np.column_stack([principal_axes.major_axis, principal_axes.minor_axis]))
            grid = Grid()

            if direction > 3*pi/4 or direction < pi/4: # Nx1, horizontal
                # Make sure axes increase from top to bottom and left to right
                if principal_axes.major_axis[0] < 0:
                    principal_axes.major_axis *= -1
                if principal_axes.minor_axis[1] < 0:
                    principal_axes.minor_axis *= -1

                grid.dims = np.array([len(points), 1], dtype=np.int32)
                grid.basis_u = principal_axes.major_axis
                grid.basis_v = principal_axes.minor_axis

                grid.proj_u = proj_matrix[0]
                grid.proj_v = proj_matrix[1]

            else: # 1xN, vertical
                # Make sure axes increase from top to bottom and left to right
                if principal_axes.major_axis[1] < 0:
                    principal_axes.major_axis *= -1
                if principal_axes.minor_axis[0] < 0:
                    principal_axes.minor_axis *= -1

                grid.dims = np.array([1, len(points)], dtype=np.int32)
                grid.basis_u = principal_axes.minor_axis
                grid.basis_v = principal_axes.major_axis

                grid.proj_u = proj_matrix[1]
                grid.proj_v = proj_matrix[0]

            return grid
        
    # At this point points are guaranteed to not form a line
    bounding_polygon = get_bounding_polygon(points, acuteness_thr)

    if bounding_polygon == None:
        return None
    
    horizontal_axis = ((bounding_polygon.top_right - bounding_polygon.top_left) + \
                       (bounding_polygon.bottom_right - bounding_polygon.bottom_left)) / 2
    
    vertical_axis = ((bounding_polygon.top_left - bounding_polygon.bottom_left) + \
                     (bounding_polygon.top_right - bounding_polygon.bottom_right)) / 2
    
    centroid = (bounding_polygon.top_left + bounding_polygon.top_right + \
                bounding_polygon.bottom_left + bounding_polygon.bottom_right) / 4
    
    offset = centroid - (horizontal_axis + vertical_axis) / 2

    proj_matrix = np.linalg.inv(np.column_stack([horizontal_axis, vertical_axis]))
    normalized_points = [np.dot(proj_matrix, p) for p in points]

    # Calculate bin sizes for each axis
    min_norm_distance = get_extreme_distance(normalized_points, minimum=True)
    x_bin_size, y_bin_size = min_norm_distance, min_norm_distance

    if cell_xy_ratio_estimate < 1: # min distance is likely from X axis neighbors
        y_bin_size *= cell_xy_ratio_estimate  
    else: # likely from Y axis neighbors
        x_bin_size *= cell_xy_ratio_estimate

    # Group by bins to infer grid size
    def group_by_bin(list, key, bin_size):
        list.sort(key=key)
        bins, current_bin, upper_bound = [], [], None
        for item in list:
            if upper_bound == None:
                current_bin.append(item)
                upper_bound = key(item) + bin_size

            elif key(item) <= upper_bound:
                current_bin.append(item)
            else:
                bins.append(current_bin)
                current_bin = [item]
                upper_bound = key(item) + bin_size

        if len(current_bin) != 0:
            bins.append(current_bin)
        
        return bins
    
    x_bins = group_by_bin(normalized_points, key=lambda p: p[0], bin_size=x_bin_size)
    y_bins = group_by_bin(normalized_points, key=lambda p: p[1], bin_size=x_bin_size)

    # check for discrepancies
    x_size = mode([len(b) for b in y_bins], "max")
    y_size = mode([len(b) for b in x_bins], "max")

    if x_size != len(x_bins) or y_size != len(y_bins):
        # print("x:", x_size, len(y_bins), "y:", y_size, len(x_bins))
        return None
    
    grid = Grid()
    
    grid.dims = np.array([x_size, y_size], dtype=np.int32)
    grid.offset = offset
    grid.basis_u = horizontal_axis / (x_size - 1)
    grid.basis_v = vertical_axis / (y_size - 1)

    proj_mat = np.linalg.inv(np.column_stack([grid.basis_u, grid.basis_v]))
    grid.proj_u = proj_mat[0]
    grid.proj_v = proj_mat[1]

    return grid
    