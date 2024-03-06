from direct.showbase.ShowBase import ShowBase
from typing import Union
import panda3d
from panda3d.core import WindowProperties, NodePath
import open3d as o3d
from direct.showbase.Loader import Loader
import math
from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from pycvsim.scene_objects.utils import o3d_mesh_to_pandas3d


class SceneObject:
    node_path: panda3d.core.NodePath = None
    mesh: o3d.geometry.TriangleMesh = None

    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        geom_node = o3d_mesh_to_pandas3d(mesh)
        node_path = NodePath()
        self.node_path = node_path.attachNewNode(geom_node)
        self.node_path.setTwoSided(True)
        self.mesh = mesh

    def set_pos(self, pos):
        self.node_path.setPos(*pos)
