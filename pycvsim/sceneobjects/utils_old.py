import open3d as o3d
import numpy as np
from numpy.typing import NDArray
from typing import List
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter, GeomNode, NodePath

"""
def create_square(points: List[NDArray], color: NDArray,
                  vertices=None, vertex_colors=None, triangle_indices=None):
    p0, p1, p2, p3 = points

    vertices = [] if vertices is None else vertices
    vertex_colors = [] if vertex_colors is None else vertex_colors
    triangle_indices = [] if triangle_indices is None else triangle_indices
    i0 = len(vertices)

    vertices.extend([p0, p1, p2, p3])
    vertex_colors.extend([color, color, color, color])
    triangle_indices.extend([[i0, i0+1, i0+2], [i0, i0+2, i0+3]])
    return {
        "vertices": vertices,
        "vertex_colors": vertex_colors,
        "triangle_indices": triangle_indices
    }


def create_square_with_circle(circle_point: NDArray,
                              square_points: List[NDArray],
                              radius: float,
                              circle_color: NDArray,
                              square_color: NDArray,
                              vertices=None, vertex_colors=None,
                              triangle_indices=None, n_elems=1200):
    p_c = circle_point
    vertices = [] if vertices is None else vertices
    vertex_colors = [] if vertex_colors is None else vertex_colors
    triangle_indices = [] if triangle_indices is None else triangle_indices

    # create circle
    for theta in np.linspace(0, 2*np.pi, n_elems):
        dtheta = 2*np.pi / (n_elems-1)
        p0 = p_c
        p1 = np.array([p0[0]+radius*np.sin(theta), p0[1]+radius*np.cos(theta), p_c[2]])
        p2 = np.array([p0[0]+radius*np.sin(theta+dtheta), p0[1]+radius*np.cos(theta+dtheta), p_c[2]])

        i0 = len(vertices)
        vertices.extend([p0, p1, p2])
        vertex_colors.extend([circle_color, circle_color, circle_color])
        triangle_indices.append([i0, i0+1, i0+2])

    # create

    bl, br, tr, tl = square_points

    return {
        "vertices": vertices,
        "vertex_colors": vertex_colors,
        "triangle_indices": triangle_indices
    }

"""