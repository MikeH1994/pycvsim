import tkinter
import tkinter.filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
import threading
import time
from typing import Tuple
from pycvsim.rendering import SceneRenderer, SceneCamera
from pycvsim.sceneobjects import SceneObject
from pycvsim.sceneobjects.calibrationtargets import CheckerbordTarget, CircleGridTarget
import pycvsim.core


class GUI:
    renderer: SceneRenderer
    displayed_image_size: Tuple[int, int]

    def __init__(self, window: tkinter.Tk):
        # Create and display a test window for viewing the menus
        self.window = window
        self.renderer = SceneRenderer()

        # setup variables
        self.last_mousepos = (0, 0)
        self.displayed_image_size = (800, 600)
        self.k_camera_angle = 90.0
        self.k_camera_pos = 0.5

        # Create left and right frames
        self.frame_tl = tkinter.Frame(self.window, width=self.displayed_image_size[0],
                                      height=self.displayed_image_size[1],
                                      bg='grey')
        self.frame_tl.grid(row=0, column=0, padx=10, pady=5)
        self.frame_tr = tkinter.Frame(self.window, width=200,
                                      height=self.displayed_image_size[1],
                                      bg='grey')
        self.frame_tr.grid(row=0, column=1, padx=10, pady=5)
        self.frame_bl = tkinter.Frame(self.window, width=self.displayed_image_size[0],
                                      height=200, bg='grey')
        self.frame_bl.grid(row=1, column=0, padx=10, pady=5)
        self.frame_br = tkinter.Frame(self.window, width=200, height=200, bg='grey')
        self.frame_br.grid(row=1, column=1, padx=10, pady=5)

        self.image_label = tkinter.Label(master=self.frame_tl)
        self.image_label.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        self.image_label.bind("<Button-1>", self.mouse_pressed)
        self.image_label.bind("<Button-3>", self.mouse_pressed)
        self.image_label.bind("<B1-Motion>", self.mouse_drag_function("left"))
        self.image_label.bind("<B3-Motion>", self.mouse_drag_function("right"))
        self.image_scrollbar = tkinter.Scrollbar(self.frame_tl, orient=tkinter.HORIZONTAL)
        self.image_scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)

        # create tabs
        self.tabs = ttk.Notebook(self.frame_tr, height=self.displayed_image_size[1]-23)
        self.cameras_tab = tkinter.Frame(self.tabs, width=200)
        self.objects_tab = tkinter.Frame(self.tabs, width=200)
        self.tabs.add(self.cameras_tab, text="camera")
        self.tabs.add(self.objects_tab, text="objects")
        self.tabs.add(self.objects_tab, text="calibration target")
        self.tabs.pack()

        # create camera listbox and scrollbar
        self.camera_listbox_frame = tkinter.Frame(self.cameras_tab, height=400)
        self.camera_listbox_frame.grid(row=0, column=0)
        self.camera_selected_list = tkinter.Listbox(self.camera_listbox_frame)
        self.camera_selected_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.camera_selected_scrollbar = tkinter.Scrollbar(self.camera_listbox_frame)
        self.camera_selected_scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.BOTH)
        self.camera_selected_list.config(yscrollcommand=self.camera_selected_scrollbar.set)
        self.camera_selected_scrollbar.config(command=self.camera_selected_list.yview)

        # create camera buttons
        self.camera_options_frame = tkinter.Frame(self.cameras_tab)
        self.camera_options_frame.grid(row=1, column=0)
        self.add_camera_button = tkinter.Button(self.camera_options_frame, text="Add",
                                                command=self.add_camera_button_pressed)
        self.add_camera_button.grid(row=0, column=0)
        self.edit_camera_button = tkinter.Button(self.camera_options_frame, text="Edit",
                                                 command=self.edit_camera_button_pressed)
        self.edit_camera_button.grid(row=0, column=1)
        self.remove_camera_button = tkinter.Button(self.camera_options_frame, text="Remove",
                                                   command=self.remove_camera_button_pressed)
        self.remove_camera_button.grid(row=0, column=2)

        # create object listbox and scrollbar
        self.object_listbox_frame = tkinter.Frame(self.objects_tab, height=400)
        self.object_listbox_frame.grid(row=0, column=0)
        self.object_selected_list = tkinter.Listbox(self.object_listbox_frame)
        self.object_selected_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.object_selected_scrollbar = tkinter.Scrollbar(self.object_listbox_frame)
        self.object_selected_scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.BOTH)
        self.object_selected_list.config(yscrollcommand=self.object_selected_scrollbar.set)
        self.object_selected_scrollbar.config(command=self.object_selected_list.yview)

        # create camera buttons
        self.object_options_frame = tkinter.Frame(self.objects_tab)
        self.object_options_frame.grid(row=1, column=0)
        self.add_object_button = tkinter.Button(self.object_options_frame, text="Add",
                                                command=self.add_object_button_pressed)
        self.add_object_button.grid(row=0, column=0)
        self.edit_object_button = tkinter.Button(self.object_options_frame, text="Edit",
                                                 command=self.edit_object_button_pressed)
        self.edit_object_button.grid(row=0, column=1)
        self.remove_object_button = tkinter.Button(self.object_options_frame, text="Remove",
                                                   command=self.remove_object_button_pressed)
        self.remove_object_button.grid(row=0, column=2)

        # setup default render
        default_obj = CheckerbordTarget((7, 6), (0.05, 0.05), board_thickness=0.02, name="Default target")
        default_obj.set_pos(np.array([0, 0, 2.0]))
        self.add_object(default_obj)

        # create default cameras
        camera = SceneCamera(res=(720, 720), name="Default")
        self.add_camera(camera)
        self.render()

    def mouse_drag_function(self, kind):
        def func(event):
            dx = (event.x - self.last_mousepos[0])/self.displayed_image_size[0]
            dy = (event.y - self.last_mousepos[1])/self.displayed_image_size[1]
            if kind == "left":
                angles = np.array([-self.k_camera_angle * dy,
                                   self.k_camera_angle * dx, 0])
                self.renderer.cameras[self.get_camera_index()].rotate(angles)
            elif kind == "right":
                pos = np.array([self.k_camera_pos*dx,
                                self.k_camera_pos*dy, 0.0])
                self.renderer.cameras[self.get_camera_index()].translate(pos)
            self.last_mousepos = (event.x, event.y)
            self.render()
        return func

    def mouse_pressed(self, event):
        self.last_mousepos = (event.x, event.y)
        self.render()

    def add_camera_button_pressed(self):
        print("Add camera presed")

    def edit_camera_button_pressed(self):
        print("Edit camera presed")

    def remove_camera_button_pressed(self):
        print("Remove camera presed")

    def add_object_button_pressed(self):
        print("Add object presed")

    def edit_object_button_pressed(self):
        print("Edit object presed")

    def remove_object_button_pressed(self):
        print("Remove object presed")

    def add_target_button_pressed(self):
        print("Add target pressed")

    def edit_target_button_pressed(self):
        print("Edit target pressed")

    def remove_target_button_pressed(self):
        print("Remove target pressed")

    def get_camera_index(self):
        if self.camera_selected_list.size() == 0:
            return -1
        index = 0
        return index

    def add_camera(self, camera: SceneCamera):
        self.renderer.add_camera(camera)
        self.camera_selected_list.insert(self.camera_selected_list.size(), camera.name)

    def add_object(self, obj: SceneObject):
        self.renderer.add_object(obj)
        self.object_selected_list.insert(self.camera_selected_list.size(), obj.name)

    def update_image_label(self, image):
        image = pycvsim.core.resize_image(image, self.displayed_image_size)
        image = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.image_label.configure(image=image)
        self.image_label.img = image

    def render(self):
        index = self.get_camera_index()
        if index == -1:
            image = np.zeros(())
            self.update_image_label(image)
            return

        image = self.renderer.render_image(index)
        self.update_image_label(image)


def run():
    root = tkinter.Tk()
    root.call('wm', 'attributes', '.', '-topmost', True)
    application = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    run()
