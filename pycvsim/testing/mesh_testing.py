import open3d as o3d
import numpy as np
from pycvsim.scene_objects.utils import create_box, create_cylinder, create_box_with_hole

square_pos = [0.0, 0.0, 0.0]
circle_pos = [0.5, 0.5, 0.0]
square = create_box(square_pos, width=1.0, height=1.0, depth=0.4, color=[0.0, 1.0, 0.0])
circle = create_cylinder(circle_pos, 0.3, 0.4, color=[1.0, 0.0, 0.0])
square_cutout = create_box_with_hole(square_pos, circle_pos, square_width=1.0, square_height=1.0, depth=0.4,
                                     circle_radius=0.15, square_color=[0.0,0.0,1.0])
sphere = o3d.geometry.TriangleMesh.create_sphere(0.1).translate(circle_pos)
# o3d.visualization.draw_geometries([square_cutout]) # square


#sphere = o3d.t.geometry.TriangleMesh.create_sphere(0.1)
#sphere_t = tensor_mesh_to_legacy_mesh(sphere)