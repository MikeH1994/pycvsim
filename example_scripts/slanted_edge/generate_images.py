import numpy as np
from pycvsim.camera.basecamera import BaseCamera
from pycvsim.routines.slantededge.slantededgeroutine import SlantedEdgeRoutine
import cv2
import os
import matplotlib.pyplot as plt


def run(output_folder):
    camera = BaseCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=20.0)
    for angle in np.linspace(0.0, 45.0, 46):
        routine = SlantedEdgeRoutine(camera, angle=angle)
        image, p0, p1 = routine.generate_image()
        # cv2.imwrite(os.path.join(output_folder, "slant_edge_{:.2f}.png".format(angle)), image.astype(np.uint8))
        plt.imshow(image[:, :, 0])
        plt.show()
        print("{} written".format(angle))


if __name__ == "__main__":
    run(r"C:\Users\mh18\OneDrive - National Physical Laboratory\Documents\Projects\NMS TI 2023\data\simulated_data\pycvsim")