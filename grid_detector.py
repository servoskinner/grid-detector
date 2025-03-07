import numpy as np
from scipy.spatial import ConvexHull
from math import atan2, pi

from utils import Basis_ortho, get_principal_axes, get_angle, adjacent, clockwise_sort, min_distance, mode

#################################################
#                                               #
# Module for detecting grid on a set of points. #
#                                               #
#################################################

# Basis plus exact number of cells horizontally and vertically
class Grid(Basis_ortho):
    def __init__(self):
        super()
        self.dims = np.ones(2, dtype=np.int32)



# A polygon with 4 points with known orientation in space
class Bounding_polygon:
    def __init__(self):
        self.top_left = np.zeros(2)
        self.top_right = np.zeros(2)
        self.bottom_right = np.zeros(2)
        self.bottom_left = np.zeros(2)



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
                line_spacing_deviation_thr = 0.5,
                bin_width = 0.5,
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
            major_axis = principal_axes.basis_u
            minor_axis = principal_axes.basis_v
            
            direction = atan2(major_axis[1], major_axis[0])
            mean_index = (len(points) - 1) / 2
            index_variance = mean_index**2 /3 
            # Mean of major coordinates is 0
            major_coordinates = [p[0] for p in pa_points]

            coords_index_correlation = 0
            for i, u in enumerate(major_coordinates):
                coords_index_correlation += (i - mean_index) * u

            spacing = coords_index_correlation / len(points) / index_variance

            major_axis *= spacing
            minor_axis *= spacing

            grid = Grid()

            if direction > 3*pi/4 or direction < pi/4: # Nx1, horizontal
                # Make sure axes increase from top to bottom and left to right
                if major_axis[0] < 0:
                    major_axis *= -1
                if minor_axis[1] < 0:
                    minor_axis *= -1

                grid.dims = np.array([len(points), 1], dtype=np.int32)
                grid.basis_u = major_axis
                grid.basis_v = minor_axis

            else: # 1xN, vertical
                # Make sure axes increase from top to bottom and left to right
                if major_axis[1] < 0:
                    major_axis *= -1
                if minor_axis[0] < 0:
                    minor_axis *= -1

                grid.dims = np.array([1, len(points)], dtype=np.int32)
                grid.basis_u = minor_axis
                grid.basis_v = major_axis

            grid.origin = principal_axes.origin - major_axis * (len(points) - 1)/2.0

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

    norm_basis = Basis_ortho()
    norm_basis.basis_u, norm_basis.basis_v, norm_basis.origin = horizontal_axis, vertical_axis, offset

    normalized_points = [norm_basis.to_local(p) for p in points]

    # Calculate bin sizes for each axis
    min_norm_distance = min_distance(normalized_points)

    x_bin_size, y_bin_size = min_norm_distance * bin_width, min_norm_distance * bin_width

    xy_scale_ratio = cell_xy_ratio_estimate * np.linalg.norm(vertical_axis) / np.linalg.norm(horizontal_axis)

    if xy_scale_ratio < 1: # min distance is likely from normalized X axis neighbors
        y_bin_size /= xy_scale_ratio  
    else: # likely from normalized Y axis neighbors
        x_bin_size *= xy_scale_ratio

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
    y_bins = group_by_bin(normalized_points, key=lambda p: p[1], bin_size=y_bin_size)

    # check for discrepancies (max bin length mode vs. other axis' bin count)
    x_size = mode([len(b) for b in y_bins], "max")
    y_size = mode([len(b) for b in x_bins], "max")

    if x_size != len(x_bins) or y_size != len(y_bins):
        return None
    
    grid = Grid()
    
    grid.dims = np.array([x_size, y_size], dtype=np.int32)
    grid.origin = offset
    grid.basis_u = horizontal_axis / (x_size - 1)
    grid.basis_v = vertical_axis / (y_size - 1)

    return grid
    