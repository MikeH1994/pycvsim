import tkinter
import tkinter.filedialog
from tkinter import ttk
from typing import Tuple

import numpy as np
from PIL import ImageTk, Image

import pycvsim.core
from pycvsim.rendering import SceneRenderer, SceneCamera
from pycvsim.sceneobjects import SceneObject
from pycvsim.sceneobjects.calibrationtargets import CheckerbordTarget
from camera_window import AddCameraWindow


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
        self.frame_tl.grid(row=0, column=0, padx=10, pady=10)
        self.frame_tr = tkinter.Frame(self.window, width=200,
                                      height=self.displayed_image_size[1],
                                      bg='grey')
        self.frame_tr.grid(row=0, column=1, padx=10, pady=10)
        self.frame_bl = tkinter.Frame(self.window, width=self.displayed_image_size[0],
                                      height=200, bg='grey')
        self.frame_bl.grid(row=1, column=0, padx=10, pady=10)
        self.frame_br = tkinter.Frame(self.window, width=200, height=200, bg='grey')
        self.frame_br.grid(row=1, column=1, padx=10, pady=10)

        self.image_label = tkinter.Label(master=self.frame_tl)
        self.image_label.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        self.image_label.bind("<Button-1>", self.mouse_pressed)
        self.image_label.bind("<Button-3>", self.mouse_pressed)
        self.image_label.bind("<B1-Motion>", self.mouse_drag_function("left"))
        self.image_label.bind("<B3-Motion>", self.mouse_drag_function("right"))
        self.camera_selected_scrollbar = tkinter.Scale(self.frame_tl, from_=0, to=0, orient=tkinter.HORIZONTAL,
                                                       resolution=1, command=self.camera_selected_changed)
        self.camera_selected_scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)

        # create tabs
        self.tabs = ttk.Notebook(self.frame_tr, height=self.displayed_image_size[1]-23)
        self.cameras_tab = tkinter.Frame(self.tabs, width=200)
        self.objects_tab = tkinter.Frame(self.tabs, width=200)
        self.targets_tab = tkinter.Frame(self.tabs, width=200)
        self.tabs.add(self.cameras_tab, text="camera")
        #self.tabs.add(self.objects_tab, text="objects")
        self.tabs.add(self.targets_tab, text="calibration target")
        self.tabs.pack()

        # create camera listbox and scrollbar
        self.camera_listbox_frame = tkinter.Frame(self.cameras_tab, height=400)
        self.camera_listbox_frame.grid(row=0, column=0)
        self.camera_selected_list = self.add_listbox(self.camera_listbox_frame)

        # create camera buttons
        self.camera_options_frame = tkinter.Frame(self.cameras_tab)
        self.camera_options_frame.grid(row=1, column=0)
        self.add_buttons(self.camera_options_frame, button_labels=["Add", "Edit", "Remove"],
                         button_commands=[self.cb_add_camera, self.cb_edit_camera, self.cb_remove_camera])

        # create object listbox and scrollbar
        self.object_listbox_frame = tkinter.Frame(self.objects_tab, height=400)
        self.object_listbox_frame.grid(row=0, column=0)
        self.object_selected_list = self.add_listbox(self.object_listbox_frame)

        # create object buttons
        self.object_options_frame = tkinter.Frame(self.objects_tab)
        self.object_options_frame.grid(row=1, column=0)
        self.add_buttons(self.object_options_frame, button_labels=["Add", "Edit", "Remove"],
                         button_commands=[self.cb_add_object, self.cb_edit_object, self.cb_remove_object])

        # create object listbox and scrollbar
        self.target_listbox_frame = tkinter.Frame(self.targets_tab, height=400)
        self.target_listbox_frame.grid(row=0, column=0)
        self.target_selected_list = self.add_listbox(self.target_listbox_frame)

        # create object buttons
        self.target_options_frame = tkinter.Frame(self.targets_tab)
        self.target_options_frame.grid(row=1, column=0)
        self.add_buttons(self.target_options_frame, button_labels=["Add", "Edit", "Remove"],
                         button_commands=[self.cb_add_target, self.cb_edit_target, self.cb_remove_target])

        # setup default render
        default_obj = CheckerbordTarget((7, 6), (0.05, 0.05), board_thickness=0.02, name="Default target")
        default_obj.set_pos(np.array([0, 0, 2.0]))
        self.add_object(default_obj)

        # create default cameras
        camera = SceneCamera(res=(720, 720), name="Default camera 1")
        self.add_camera(camera)
        camera = SceneCamera.create_camera_from_lookpos(np.array([0.5, 0.0, 0.0]),
                                                        np.array([0.0, 0.0, 2.0]),
                                                        np.array([0.0, 1.0, 0.0]),
                                                        res=(720, 720), hfov=40.0,
                                                        name="Default camera 2")
        self.add_camera(camera)
        self.render()
        self.camera_selected_scrollbar.configure(to=len(self.renderer.cameras)-1)
        self.update_camera_scrollbar()

    def add_buttons(self, master, button_labels, button_commands):
        for i in range(len(button_labels)):
            button = tkinter.Button(master, text=button_labels[i], command=button_commands[i])
            button.grid(row=0, column=i)

    def add_listbox(self, master):
        frame = tkinter.Frame(master, height=400)
        frame.grid(row=0, column=0)
        listbox = tkinter.Listbox(frame)
        listbox.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        scrollbar = tkinter.Scrollbar(frame)
        scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.BOTH)
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        return listbox

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

    def camera_selected_changed(self, _):
        self.render()

    def update_camera_scrollbar(self):
        self.camera_selected_scrollbar.configure(to=len(self.renderer.cameras)-1)
        n = min(self.get_camera_index(), len(self.renderer.cameras)-1)
        self.camera_selected_scrollbar.set(n)

    def cb_close_popup_window(self):
        pass

    def cb_add_camera(self):
        win = AddCameraWindow(self.window)
        win.attributes('-topmost', 'true')
        print("Add camera pressed")

    def cb_edit_camera(self):
        print("Edit camera presed")

    def cb_remove_camera(self):
        selections = self.camera_selected_list.curselection()
        if len(selections) == 0:
            return
        assert(len(selections) == 1)
        selection = selections[0]
        self.renderer.remove_camera(selection)
        self.camera_selected_list.delete(selection)
        self.update_camera_scrollbar()
        self.render()

    def cb_add_object(self):
        print("Add object presed")

    def cb_edit_object(self):
        print("Edit object presed")

    def cb_remove_object(self):
        print("Remove object presed")

    def cb_add_target(self):
        print("Add target pressed")

    def cb_edit_target(self):
        print("Edit target pressed")

    def cb_remove_target(self):
        print("Remove target pressed")

    def get_camera_index(self):
        if self.camera_selected_list.size() == 0:
            return -1
        return self.camera_selected_scrollbar.get()

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
            image = np.zeros((self.displayed_image_size[1], self.displayed_image_size[0]))
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
