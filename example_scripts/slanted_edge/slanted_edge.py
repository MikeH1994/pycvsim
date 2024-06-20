import numpy as np
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.routines.slantededge.slantededgeroutine import SlantedEdgeRoutine
import matplotlib.pyplot as plt
import scipy.stats as st


def gkern(kernlen=5, nsig=1):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


blurring_kernel = gkern()
camera = SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=20.0)
camera_blur = SceneCamera(pos=np.array([0.0, 0.0, -1.5]), res=(800, 800), hfov=20.0)

"""
plt.title("ESF (no blurring)")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=5.0).run(normalize=False)
plt.plot(esf_x, esf_f, label="5 degrees")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=25.0).run(normalize=False)
plt.plot(esf_x, esf_f, label="25 degrees")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=44.0).run(normalize=False)
plt.plot(esf_x, esf_f, label="44 degrees")
plt.xlim((-0.75, 0.75))
plt.legend(loc=0)
plt.figure()

plt.title("ESF (no blurring)- cos theta correction")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=5.0).run(normalize=False)
plt.plot(esf_x*np.cos(np.radians(5.0)), esf_f, label="5 degrees")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=25.0).run(normalize=False)
plt.plot(esf_x*np.cos(np.radians(25.0)), esf_f, label="25 degrees")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=44.0).run(normalize=False)
plt.plot(esf_x*np.cos(np.radians(44.0)), esf_f, label="44 degrees")
plt.xlim((-0.75, 0.75))
plt.legend(loc=0)
plt.show()"""

plt.title("ESF with blurring")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera_blur, angle=5.0).run(normalize=False, convert_to_8_bit=True, apply_gaussian=True)
plt.plot(esf_x, esf_f, label="5 degrees")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera_blur, angle=25.0).run(normalize=False, convert_to_8_bit=True, apply_gaussian=True)
plt.plot(esf_x, esf_f, label="25 degrees")
esf_x, esf_f, _ = SlantedEdgeRoutine(camera_blur, angle=44.0).run(normalize=False, convert_to_8_bit=True, apply_gaussian=True)
plt.plot(esf_x, esf_f, label="44 degrees")
plt.xlim((-3, 3))
plt.legend(loc=0)
plt.show()
