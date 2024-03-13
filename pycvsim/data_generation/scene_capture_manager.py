

class SceneCaptureManager:
    def __init__(self, **kwargs):
        self.camera_mode = "fixed"
        self.n_images = 15

    def run(self):
        if self.camera_mode == "fixed":
            for n in range(self.n_images):
                pass
        else:
            pass