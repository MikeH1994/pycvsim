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



routine = SlantedEdgeRoutine(camera, angle=5.0)
esf_x, esf_f, image = routine.run(normalize=False)

esf_x_blurred, esf_f_blurred, _ = routine.run(blurring_kernel=blurring_kernel, normalize=False)
plt.title("Edge profile with and without PSF")
plt.plot(esf_x*np.cos(np.radians(5)), esf_f/255.0, label="No PSF")
plt.plot(esf_x_blurred*np.cos(np.radians(5)), esf_f_blurred/255.0, label="With gaussian PSF")
plt.xlabel("Distance from pixel to edge")
plt.ylabel("Intensity")
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

if True:
    plt.title("Simulated Edge Profile (no blurring)")
    esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=5.0).run(normalize=False)
    plt.plot(esf_x, esf_f, label="5 degree edge")
    # esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=25.0).run(normalize=False)
    # plt.plot(esf_x, esf_f, label="25 degrees")
    esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=44.0).run(normalize=False)
    #plt.plot(esf_x, esf_f, label="44 degree edge")
    plt.xlim((-0.75, 0.75))
    plt.xlabel("Distance from pixel to edge (px)")
    plt.ylabel("Normalised intensity")
    #plt.legend(loc=0)
    plt.show()


if False:
    plt.title("ESF (no blurring)- cos theta correction")
    esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=5.0).run(normalize=False)
    plt.plot(esf_x*np.cos(np.radians(5.0)), esf_f, label="5 degrees")
    esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=25.0).run(normalize=False)
    plt.plot(esf_x*np.cos(np.radians(25.0)), esf_f, label="25 degrees")
    esf_x, esf_f, _ = SlantedEdgeRoutine(camera, angle=44.0).run(normalize=False)
    plt.plot(esf_x*np.cos(np.radians(44.0)), esf_f, label="44 degrees")
    plt.xlim((-0.75, 0.75))
    plt.legend(loc=0)
    plt.show()


if False:
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


if False:
    plt.title("Edge Profile")
    plt.xlabel("Distance to edge (pixels)")
    plt.ylabel("Normalised intensity")
    esf_x, esf_f, _ = SlantedEdgeRoutine(camera_blur, angle=5.0).run(normalize=False, convert_to_8_bit=True, apply_gaussian=True)
    esf_f += np.random.normal(scale=0.01, size=esf_f.size)
    plt.scatter(esf_x[::25], esf_f[::25], label="5 degrees")
    plt.xlim((-15, 15))
    plt.show()
