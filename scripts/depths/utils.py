import os

import liblzfse
import numpy as np
from scipy.ndimage import zoom


def read_depth_from_bin(data_path):
    # Binary files to depth array
    with open(os.path.join(data_path, "compressed_np_depth_float32.bin"), "rb") as file:
        depth = file.read()
        depth = np.frombuffer(liblzfse.decompress(depth), dtype=np.float32)
        depth = depth.reshape((-1, 192, 256))
    return depth


def bilinear_interpolation(depth, target_shape=(256, 256)):
    interpolated_depths = []
    for d in depth:
        scale = [t / s for t, s in zip(target_shape, d.shape)]
        interpolated_depth = zoom(
            d, scale, order=1
        )  # order=1 for bilinear interpolation
        interpolated_depths.append(interpolated_depth)
    return np.stack(interpolated_depths)


def depth_to_file(depth, data_dir):
    BOUND_FACTOR = 20
    # Depth Array to .npy files
    # Interpoltat
    depth_interpolat = bilinear_interpolation(depth, target_shape=(256, 256))
    depth_interpolat = BOUND_FACTOR * depth_interpolat

    os.makedirs(os.path.join(data_dir, "depth"),exist_ok=True)

    with open(os.path.join(data_dir, "depth_list.txt"), "w") as file:
        for i, d in enumerate(depth_interpolat):
            file_path = os.path.join(data_dir, "depth", f"depth_{i}.npy")
            np.save(file_path, d)
            file.write(file_path + "\n")
