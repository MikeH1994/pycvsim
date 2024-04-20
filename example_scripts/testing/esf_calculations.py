import numpy as np


def distance_to_edge(m: float, c: float, x: float, y: float):
    return (y - m * x - c) / np.sqrt(m ** 2 + 1)


def gradient_to_angle(m: float):
    return np.degrees(np.arctan(m))


def run():
    y = m*x + c


if __name__ == "__main__":
    run()