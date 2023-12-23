# HoNY 数据集内有相机的位姿
# json file -> projection matrix >> cams_poses.npy
# orginal npy file -> intric, distortion >> cams_poses.npy
# bin file -> depth >> depth_list.txt, depth_xxx.npy
import json
import os
from os.path import join as pjoin

import click
import liblzfse
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

BOUND_FACTOR = 20

original_file = "/home/DISCOVER_summer2022/cuily/workspace/f2-nerf/data/Air_Fryer_Opening/haritheja_Env1_2023-11-21--13-34-39/evo/cams_meta_colmap.npy"  # 读取 intric 等参数


# Function to process the JSON data
def process_json_data(json_data):
    data_list = []

    for key, value in json_data.items():
        xyz = np.array(value["xyz"])  # [3]
        quat = np.array(value["quats"])  # [4]
        rot_matrix = R.from_quat(quat).as_matrix()  # [3,3]
        combined_matrix = np.hstack((rot_matrix, xyz.reshape(-1, 1)))  # [3,3] [3,1]

        # Inverse
        # projection_matrix = np.vstack([combined_matrix, np.array([0,0,0,1])]) # [4,4]
        # combined_matrix = np.linalg.inv(projection_matrix)[0:3,:]
        # print(f"{combined_matrix.shape=}")

        data_list.append(combined_matrix.reshape(-1))

    final_array = np.array(data_list)
    return final_array


def depth_to_bound(depth):
    depth = depth.reshape((-1, 192 * 256))
    depth_min = np.min(depth, axis=-1)[..., np.newaxis]
    depth_max = np.max(depth, axis=-1)[..., np.newaxis]
    return np.hstack([depth_min, depth_max])


def viz_bound(bound_data, data_dir):
    colors = np.arange(bound_data.shape[0])

    # 绘制散点图
    plt.scatter(bound_data[:, 0], bound_data[:, 1], c=colors, cmap="viridis")
    plt.colorbar(label="Element Index in the List")
    plt.title("2D Scatter Plot with Colors Representing Element Order")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.savefig(pjoin(data_dir, "viz_bound.png"))


@click.command()
@click.option("--data_dir", type=str)
def main(data_dir):
    # Read JSON data from a file
    with open(pjoin(data_dir, "labels.json"), "r") as file:
        json_data = json.load(file)

        # Process the data
        processed_data = process_json_data(json_data)

    n_images = processed_data.shape[0]
    camera_pose = np.load(original_file)

    # Get other parameters from colmap result
    other_params = np.broadcast_to(camera_pose[1, 12:], (n_images, 15)).copy()

    # Depth to Bound
    with open(pjoin(data_dir, "compressed_np_depth_float32.bin"), "rb") as file:
        depth = file.read()
        depth = np.frombuffer(liblzfse.decompress(depth), dtype=np.float32)
        bound = depth_to_bound(depth)
        viz_bound(bound, data_dir)
        other_params[:, -2:] = bound * BOUND_FACTOR # Change Bound

    result_pose = np.hstack((processed_data, other_params))

    print(f"{result_pose.shape=}")
    np.save(pjoin(data_dir, "cams_meta.npy"), result_pose)


if __name__ == "__main__":
    main()
