import numpy as np
import matplotlib.pyplot as plt


class KnifeEdge:
    def __init__(self, p, angle):
        self.p = np.array(p)
        self.angle = angle
        self.dirn = np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
        self.normal = np.array([self.dirn[1], -self.dirn[0]])

    def distance(self, p):
        return np.sqrt((p[0] - self.p[0])**2 + (p[1] - self.p[1])**2)

    def get_pos(self, distance: float = 1.0):
        return self.p + self.normal*distance

    def get_draw_pos(self, d=1.0, use_normal=False):
        if use_normal:
            x = [self.p[0] - d*self.normal[0], self.p[0] + d*self.normal[0]]
            y = [self.p[1] - d*self.normal[1], self.p[1] + d*self.normal[1]]
        else:
            x = [self.p[0] - d*self.dirn[0], self.p[0] + d*self.dirn[0]]
            y = [self.p[1] - d*self.dirn[1], self.p[1] + d*self.dirn[1]]
        return x, y

    def point_above_line(self, x, y):
        # if line is vertical
        if self.dirn[0] == 0:
            return x > self.p[0]
        else:
            dx = (x-self.p[0])/self.dirn[0]
            y_line = self.p[1] + dx*self.dirn[1]
            return y < y_line

    def sample_points(self, pixel_center, width = 1.0, n_samples = 100):
        x0, y0 = pixel_center
        x = np.linspace(x0 - width / 2.0, x0 + width / 2.0, n_samples)
        y = np.linspace(y0 - width / 2.0, y0 + width / 2.0, n_samples)
        xx, yy = np.meshgrid(x, y)
        p = self.point_above_line(xx, yy)
        n = np.count_nonzero(p) / np.size(p)
        return xx, yy, p, n

    def get_intersection_points(self, pixel_center, width=1.0):
        cx, cy = pixel_center
        # check vertical intersections


def visualise():
    for angle in [45.0]:
        edge = KnifeEdge([0.0, 0.0], angle)
        d_samples = np.linspace(-0.8, 1.0, 5)
        for d in d_samples:
            x, y = edge.get_pos(d)
            xx, yy, p, n = edge.sample_points([x, y])
            plt.plot([x-0.5, x+0.5, x+0.5, x-0.5, x-0.5], [y-0.5, y-0.5, y+0.5, y+0.5, y-0.5])
            a, b = edge.get_draw_pos()
            plt.plot(a, b)
            plt.scatter(xx[p], yy[p], s=1, color='b')
            plt.scatter(x, y, s=40, color='black')
            # plt.scatter(xx[~p], yy[~p], s=1, color='b')
            plt.axis('equal')
            plt.show()


def run_distance_correction():
    angle = 45.0
    edge = KnifeEdge([0.0, 0.0], angle)
    distances = np.linspace(-1.0, 1.0, 1000)
    intensities = []
    for d in distances:
        x, y = edge.get_pos(d)
        xx, yy, p, n = edge.sample_points([x, y], n_samples=1000)
        intensities.append(n)
    plt.plot(distances, intensities, label="Without distance correction")
    plt.plot(distances*np.cos(np.radians(angle)), intensities, label="With distance correction")
    plt.legend(loc=0)
    plt.show()


def run():
    interp_fns = []
    for angle in [0.0, 45.0]:
        edge = KnifeEdge([0.0, 0.0], angle)
        distances = np.linspace(-10.0, 10.0, 10000)
        intensities = []
        for d in distances:
            x, y = edge.get_pos(d)
            xx, yy, p, n = edge.sample_points([x, y], n_samples=1000)
            intensities.append(n)
        # distances *= np.cos(np.radians(angle))
        plt.plot(distances, intensities, label="Angle = {}".format(angle))
    plt.title("Edge Spread function")
    plt.xlabel("Distance from pixel to edge")
    plt.ylabel("Normalised intensity")
    plt.legend(loc=0)
    plt.show()


if __name__ == "__main__":
    # visualise()
    # run_distance_correction()
    run()
