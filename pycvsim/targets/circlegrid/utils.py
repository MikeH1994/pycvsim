import numpy as np
from ..utils import combine_submeshes


def create_circle_in_square(centre, square_size, circle_radius, square_colour, circle_colour, n_segments=250):
    square_v, square_t, square_c = create_square_with_circle_cutout(centre, square_size, circle_radius, square_colour, n_segments)
    circle_v, circle_t, circle_c = create_circle(centre, circle_radius, circle_colour, n_segments)
    return combine_submeshes([square_v, circle_v], [square_t, circle_t], [square_c, circle_c])


def create_square_with_circle_cutout(centre, square_size, radius, colour, n_segments):
    vertices, triangles, colours = [], [], []
    for q in range(4):
        v, t, c = create_square_quadrant_with_circle_cutout(centre, square_size, radius, colour, q, n_segments)
        vertices.append(v)
        triangles.append(t)
        colours.append(c)
    return combine_submeshes(vertices, triangles, colours)


def create_square_quadrant_with_circle_cutout(centre, square_size, radius, colour, quadrant, n_segments):
    w, h = square_size
    r = radius
    theta_min = quadrant*np.pi/2
    theta_max = (quadrant+1)*np.pi/2
    angles = np.linspace(theta_min, theta_max, n_segments, endpoint=True)
    curve_vertices = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(shape=n_segments)])

    square_corners = [
        np.array([w/2, h/2, 0]),
        np.array([-w/2, h/2, 0]),
        np.array([-w/2, -h/2, 0]),
        np.array([w/2, -h/2, 0])
    ]
    square_midpoints = np.array([
        np.array([0, h / 2, 0]),
        np.array([w / 2, 0, 0]),
        np.array([0, -h / 2, 0]),
        np.array([-w / 2, 0, 0]),
        np.array([0, h / 2, 0])
    ])
    if quadrant %2 == 1:
        square_midpoints *= -1
    midpoint_1 = square_midpoints[quadrant]
    corner = square_corners[quadrant]
    midpoint_2 = square_midpoints[quadrant+1]
    vertices = np.vstack([midpoint_1, corner, midpoint_2, curve_vertices]) #
    i0 = 3
    imax = vertices.shape[0]-1
    if quadrant %2 == 0:
        triangles = [[2, 1, i0], [imax, 1, 0]]
    else:
        triangles = [[1, i0, 2], [0, imax, 1]]

    for i in range(i0, vertices.shape[0]-1):
        triangles.append([1, i+1, i])

    colours = [colour for _ in range(vertices.shape[0])]
    return np.array(vertices, dtype=np.float32) + centre, np.array(triangles, dtype=np.int32), np.array(colours, dtype=np.float32)

def create_circle(centre,  radius, colour, n_segments):
    r = radius
    angles = np.linspace(0, 2*np.pi, n_segments, endpoint=True)
    curve_vertices = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(shape=n_segments)])
    vertices = np.vstack([np.array([0, 0, 0]), curve_vertices]) #

    triangles = []
    for i in range(1, vertices.shape[0]-1):
        triangles.append([i+1, 0, i])

    colours = [colour for _ in range(vertices.shape[0])]
    return np.array(vertices, dtype=np.float32) + centre, np.array(triangles, dtype=np.int32), np.array(colours, dtype=np.float32)

def create_square(top_left, bottom_right, colour):
    top_right = np.array([bottom_right[0], top_left[1], top_left[2]])
    bottom_left = np.array([top_left[0], bottom_right[1], bottom_right[2]])

    vertices = np.array([top_left,top_right,bottom_left,bottom_right])
    triangles = [
        [2, 0, 1], [3, 2, 1]
    ]
    colours = [colour for _ in range(vertices.shape[0])]
    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32), np.array(colours, dtype=np.float32)

def create_circle_grid(board_size, grid_size, radius, boundary, circle_colour, bkg_colour):
    grid_width, grid_height = grid_size
    vertices = []
    triangles = []
    colors = []
    for i in range(board_size[0]):
        for j in range(board_size[1]):
            x0 = i*grid_width
            y0 = j*grid_height
            circle_centre = np.array([x0, y0, 0])
            v, t, c = create_circle_in_square(circle_centre, (grid_width, grid_height), radius, bkg_colour, circle_colour, 250)
            vertices.append(v)
            triangles.append(t)
            colors.append(c)
    for i in range(4):
        x0 = -grid_width/2
        y0 = -grid_height/2
        x1 = (board_size[0]-0.5)*grid_width
        y1 = (board_size[1]-0.5)*grid_height
        dx = boundary[0]
        dy = boundary[1]

        if i == 0:
            # left hand side
            v, t, c = create_square([x0-dx, y1+dy, 0], [x0, y0-dy, 0], bkg_colour)
        elif i == 1:
            # right hand side
            v, t, c = create_square([x1, y1+dy, 0], [x1+dx, y0-dy, 0], bkg_colour)
        elif i == 2:
            # top
            v, t, c = create_square([x0, y1+dy, 0], [x1, y1, 0], bkg_colour)
        else:
            # bottom
            v, t, c = create_square([x0, y0, 0], [x1, y0-dy, 0], bkg_colour)
        vertices.append(v)
        triangles.append(t)
        colors.append(c)

    return combine_submeshes(vertices, triangles, colors)
