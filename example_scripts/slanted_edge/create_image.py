from pycvsim.targets.slantededgetarget import SlantedEdgeTarget
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax_1 = fig.add_subplot(111)
im_1 = ax_1.imshow(np.zeros((512, 640, 3), dtype=np.uint8))
plt.ion()

while True:
    for theta in np.linspace(0, 360.0, 31):
        img = SlantedEdgeTarget.create_image((640, 512), theta, (320, 256), n_samples=1).astype(np.int32)
        im_1.set_data(img)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.title("{:.3f}".format(theta))
        plt.pause(0.01)
