from unittest import TestCase
import numpy as np
from pycvsim.targets.checkerboardtarget import CheckerbordTarget


def create_checkerboard_points(board_size, dx):
    board_size = board_size
    dx = dx
    object_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_points[:, :2] = np.indices(board_size).T.reshape(-1, 2)
    object_points *= dx
    return object_points


class TestCheckerboard(TestCase):
    def test_checkerboard_points(self):
        for board_size in [(5,5), (7, 6), (6, 7), (20, 20)]:
            for point_width in [0.05, 1.0, 5.0]:
                tgt = CheckerbordTarget(board_size, (point_width, point_width))
                points_expected = create_checkerboard_points(board_size, point_width)
                points_calculated = tgt.get_object_points(transformed=False)
