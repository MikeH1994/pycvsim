import tkinter
import tkinter.filedialog
from PIL import ImageTk, Image
import numpy as np
import threading
import time
from typing import Tuple
from pycvsim.rendering import SceneRenderer, SceneCamera
from pycvsim.sceneobjects import SceneObject
import pycvsim.core


class GUI:
    renderer: SceneRenderer = None
    displayed_image_size: Tuple[int, int] = (640, 512)

    def __init__(self, window: tkinter.Tk):
        # Create and display a test window for viewing the menus
        self.window = window
        self.renderer = SceneRenderer()

        # setup variables
        self.last_mousepos = (0, 0)
        self.displayed_image_size = (640, 512)
        self.camera_drag_sensitivity = 90.0

        # Create left and right frames
        self.frame_tl = tkinter.Frame(self.window, width=640, height=512, bg='grey')
        self.frame_tl.grid(row=0, column=0, padx=10, pady=5)
        self.frame_tr = tkinter.Frame(self.window, width=200, height=512, bg='grey')
        self.frame_tr.grid(row=0, column=1, padx=10, pady=5)
        self.frame_bl = tkinter.Frame(self.window, width=640, height=200, bg='grey')
        self.frame_bl.grid(row=1, column=0, padx=10, pady=5)
        self.frame_br = tkinter.Frame(self.window, width=200, height=200, bg='grey')
        self.frame_br.grid(row=1, column=1, padx=10, pady=5)

        self.image_label = tkinter.Label(master=self.frame_tl)
        self.image_label.grid(row=0, column=0)
        self.image_label.bind("<Button-1>", self.mouse_pressed)
        self.image_label.bind("<B1-Motion>", self.mouse_drag)

        # setup default render
        armadillo = SceneObject.load_armadillo()
        armadillo.set_pos(np.array([0, 0, 4.0]))
        self.add_object(armadillo)
        camera = SceneCamera(res=(720, 720), name="Default")
        self.add_camera(camera)
        self.render()


    def mouse_drag(self, event):
        dx = self.camera_drag_sensitivity*(event.x - self.last_mousepos[0])/self.displayed_image_size[0]
        dy = self.camera_drag_sensitivity*(event.y - self.last_mousepos[1])/self.displayed_image_size[1]
        angles = np.array([-dy, dx, 0])
        self.renderer.cameras[self.get_camera_index()].rotate(angles)

        self.last_mousepos = (event.x, event.y)
        self.render()

    def mouse_pressed(self, event):
        self.last_mousepos = (event.x, event.y)
        self.render()

    def get_camera_index(self):
        return 0

    def add_camera(self, camera: SceneCamera):
        self.renderer.add_camera(camera)

    def add_object(self, obj: SceneObject):
        self.renderer.add_object(obj)

    def update_image_label(self, image):
        image = pycvsim.core.resize_image(image, self.displayed_image_size)
        image = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.image_label.configure(image=image)
        self.image_label.img = image

    def render(self):
        image = self.renderer.render_image(self.get_camera_index())
        self.update_image_label(image)

    def render_loop(self):
        while True:
            time.sleep(0.005)
            if self.request_render:
                self.request_render = False
                self.render()


def run():
    root = tkinter.Tk()
    root.call('wm', 'attributes', '.', '-topmost', True)
    application = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    run()
