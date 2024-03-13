import tkinter


class AddCameraWindow(tkinter.Toplevel):
    def __init__(self, win):
        super().__init__(win)
        self.geometry("750x250")
        self.title("Child Window")
        tkinter.Label(self, text="Hello World!", font=('Mistral 18 bold')).place(x=150, y=80)
