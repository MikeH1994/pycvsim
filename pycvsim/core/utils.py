import psutil
import numpy as np


def get_suggested_array_size(dtype=np.float32):
    available_memory = psutil.virtual_memory().available