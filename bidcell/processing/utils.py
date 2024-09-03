import numpy as np
import multiprocessing as mp
import warnings

def get_patches_coords(size, size_patch):
    """Get start and end locations of patches in a large image given the image patch sizes"""

    if size <= size_patch:
        max_size = size
        coords = [(0, max_size)]
    else:
        max_size = size_patch
        starts = list(np.arange(0, size, size_patch))
        ends = [x + size_patch if x + size_patch <= size else size for x in starts]
        coords = list(zip(starts, ends))

    return coords, max_size


def get_n_processes(n_processes):
    """Number of CPUs for multiprocessing"""
    if n_processes is None:
        return mp.cpu_count()
    else:
        return n_processes if n_processes <= mp.cpu_count() else mp.cpu_count()

map_bit_depth = {np.uint64: np.uint16, np.uint32: np.uint16, np.int64: np.int16, np.int32: np.int16}
def check_bit_depth(img, disallowed=map_bit_depth): 
    for dtype_from, dtype_to in map_bit_depth.items():
        if img.dtype == dtype_from:
            warnings.warn(f"Image dtype was {img.dtype}, which is incompatible; converting to {dtype_to}")
            img.astype(dtype_to)
            break
    return img
