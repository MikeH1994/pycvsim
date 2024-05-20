import numpy as np
import matplotlib.pyplot as plt


def angle_to_gradient(angle: float):
    return np.tan(np.radians(angle))


def gradient_to_angle(gradient: float):
    return np.degrees(np.arctan(gradient))


def distance_to_line(gradient: float, pixel_x0: float):
    return pixel_x0*np.sin(np.radians(gradient_to_angle(gradient)))

def calc_sampling(cx, cy, m, c=0, width=1.0, n_samples=100):
    x = np.linspace(cx - width/2.0, cx + width/2.0, n_samples)
    y = np.linspace(cy - width/2.0, cy + width/2.0, n_samples)
    xx, yy = np.meshgrid(x, y)
    yy_line = xx*m + c
    n = np.count_nonzero(yy > yy_line) / np.size(yy)
    return xx, yy, yy_line, n

def visualise():
    for angle in [55.0, 20.0, 40.0]:
        m = angle_to_gradient(angle)
        x_samples = np.linspace(1.0, 0.0, 5)
        for x in x_samples:
            xx, yy, yy_line, n = calc_sampling(x, 0.0, m)
            plt.plot([x-0.5, x+0.5, x+0.5, x-0.5, x-0.5], [-0.5, -0.5, 0.5, 0.5, -0.5])
            plt.plot([-0.5, 1.0], [-0.5*m, m])
            plt.scatter(xx[yy > yy_line], yy[yy > yy_line], s=1, color='r')
            plt.scatter(xx[yy < yy_line], yy[yy < yy_line], s=1, color='b')
            plt.axis('equal')
            plt.show()


def run():
    for angle in [89.0, 20.0, 40.0]:
        m = angle_to_gradient(angle)
        x_samples = np.linspace(-2.0, 2.0, 100)
        y = []
        for x in x_samples:
            xx, yy, yy_line, n = calc_sampling(x, 0.0, m, n_samples=1000)
            y.append(n)
        plt.scatter(x_samples, y)
        plt.show()


if __name__ == "__main__":
    # visualise()
    run()
