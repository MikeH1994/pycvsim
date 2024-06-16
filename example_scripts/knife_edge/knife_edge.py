import numpy as np
from pycvsim.sceneobjects.targets.knifeedgetarget import KnifeEdgeTarget
from pycvsim.sceneobjects.sceneobject import SceneObject
from pycvsim.rendering.panda3drenderer import Panda3DRenderer
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.image_utils import overlay_points_on_image
from pycvsim.routines.knifeedge.knifeedgeroutine import KnifeEdgeRoutine
import matplotlib.pyplot as plt
import scipy.stats as st


def gkern(kernlen=5, nsig=1):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


blurring_kernel = gkern()
camera = SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=20.0)

routine = KnifeEdgeRoutine(camera, angle=5.0)
esf_x, esf_f, image = routine.run(normalize=False)

esf_x_blurred, esf_f_blurred, _ = routine.run(blurring_kernel=blurring_kernel, normalize=False)
plt.title("Edge profile with and without PSF")
plt.plot(esf_x*np.cos(np.radians(5)), esf_f/255.0, label="No PSF")
plt.plot(esf_x_blurred*np.cos(np.radians(5)), esf_f_blurred/255.0, label="With gaussian PSF")
plt.xlabel("Distance from pixel to edge")
plt.ylabel("Intensity")
plt.legend(loc=0)
plt.figure()

esf_x_5, esf_f_5, _ = KnifeEdgeRoutine(camera, angle=5.0).run(blurring_kernel=blurring_kernel, normalize=False)
esf_x_45, esf_f_45, _ = KnifeEdgeRoutine(camera, angle=44.0).run(blurring_kernel=blurring_kernel, normalize=False)
plt.title("Edge profile at different angles")
plt.plot(esf_x_5*np.cos(np.radians(5)), esf_f_5/255.0, label="5 degrees, with gaussian psf")
plt.plot(esf_x_45*np.cos(np.radians(44)), esf_f_45/255.0, label="44 degrees, with gaussian psf")
plt.xlabel("Distance from pixel to edge")
plt.ylabel("Intensity")
plt.legend(loc=0)
plt.show()
