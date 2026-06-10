import os

import numpy as np


def available_memory_bytes():
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return None


def complex_itemsize(dtype):
    return np.dtype(dtype).itemsize


def real_dtype_for_complex(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.complex64):
        return np.float32
    if dtype == np.dtype(np.complex128):
        return np.float64
    raise ValueError("dtype must be np.complex64 or np.complex128")
