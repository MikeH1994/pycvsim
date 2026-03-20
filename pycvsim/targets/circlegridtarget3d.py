import open3d as o3d
import math
import numpy as np
from utils import create_box
from pycvsim.sceneobjects.utils import tensor_mesh_to_legacy_mesh



def create_plate_with_hole(width, height, radius, z_offset=0, n_segments: int = 32, index_offset=0):
    assert(n_segments % 4 == 0)
    corner_points = [[width/2, height/2, z_offset], [width/2, -height/2, z_offset], # top right, bottom right
                     [-width/2, -height/2, z_offset], [-width/2, height/2, z_offset]] # bottom left, top left
    circle_points = []
    angles = np.linspace(0, 360.0, n_segments+1)[:-1]
    return None # points,




