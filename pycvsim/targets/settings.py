from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Tuple
from numpy.typing import NDArray
import numpy as np


@dataclass
class CheckerboardSettings:
    board_dimensions: Tuple[int, int] = (11, 10)
    checker_size: Tuple[float, float] = (0.05, 0.05)
    boundary_size: Tuple[float, float] = (0.05, 0.05)
    colour_1: Tuple[int, int, int] = (255, 255, 255)
    colour_2: Tuple[int, int, int] = (0, 0, 0)
    colour_boundary: Tuple[int, int, int] = (128, 128, 128)
    object_points: NDArray = None

    def __post_init__(self):
        board_width, board_height = self.board_dimensions
        checker_width, checker_height = self.checker_size
        self.object_points = np.zeros((board_height + 2, board_width + 2, 3))
        for i in range(board_width+2):
            for j in range(board_height+2):
                self.object_points[j, i] = [(i-1)*checker_width, (j-1)*checker_height, 0.0]

