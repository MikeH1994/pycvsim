import numpy as np
from pycvsim.rendering.scenecamera import SceneCamera
from pycvsim.core.pinhole_camera_maths import focal_length_to_fov, calc_closest_y_direction
from scipy.optimize import minimize
import open3d as o3d

def run():
    camera = SceneCamera()

    rays_total = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=camera.hfov, center=camera.get_lookpos(), eye=camera.pos,
        up=camera.get_up(), width_px=camera.xres, height_px=camera.yres).numpy()

    rays_stitched = np.zeros(rays_total.shape)
    lookpos = camera.get_pixel_direction(np.array([(camera.xres-1)/4, (camera.yres-1)/2]))
    rays_stitched[:, :camera.xres // 2, :] = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=focal_length_to_fov(camera.get_focal_length()[0], camera.xres//2), center=lookpos, eye=pos,
        up=up, width_px=camera.xres//2, height_px=camera.yres).numpy()


    lookpos = camera.get_pixel_direction(np.array([3*(camera.xres-1)/4, (camera.yres-1)/2]))
    rays_stitched[:, camera.xres//2:, :] = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=focal_length_to_fov(camera.get_focal_length()[0], camera.xres//2), center=lookpos, eye=pos,
        up=up, width_px=camera.xres//2, height_px=camera.yres).numpy()

    rays_err = rays_total - rays_stitched
    print(np.mean(rays_err))


def do_optimisation():
    camera = SceneCamera()

    rays_total = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=camera.hfov, center=camera.get_lookpos(), eye=camera.pos,
        up=camera.get_up(), width_px=camera.xres, height_px=camera.yres).numpy()
    rays_left = rays_total[:, :camera.xres//2]

    # u, v = (camera.xres - 1) / 4, (camera.yres - 1) / 2
    def minimisation_fn(X):
        u, v, hfov = X
        lookpos = camera.get_pixel_direction(np.array([u, v])) + camera.pos
        up = calc_closest_y_direction(lookpos, camera.get_up())
        rays_calc = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=hfov, center=lookpos, eye=camera.pos,
            up=up, width_px=camera.xres // 2, height_px=camera.yres).numpy()
        return np.mean(np.abs(rays_calc - rays_left))

    hfov_guess = focal_length_to_fov(camera.get_focal_length()[0], camera.xres//2)
    x0 = np.array([150, 150.0, 30.0])
    bounds = [(x0[0]-1.0, x0[0]+1.0)]
    result = minimize(minimisation_fn, x0, method='Nelder-Mead', tol=1e-7)
    print(x0)
    print(result.x)
    print(minimisation_fn(result.x))
    foo = 1



if __name__ == "__main__":
    do_optimisation()