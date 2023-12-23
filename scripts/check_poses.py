import logging
import sys
from os.path import join as pjoin

import click
import numpy as np
from matplotlib import pyplot as plt

#python ./scripts/check_poses.py --data_dir /home/DISCOVER_summer2022/cuily/workspace/f2-nerf/data/Air_Fryer_Opening/haritheja_Env1_2023-11-21--13-34-39/evo/


def init_logger(data_dir, file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(pjoin(data_dir, file + ".log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


def viz_bound(bound_data, data_dir):
    colors = np.arange(bound_data.shape[0])

    # 绘制散点图
    plt.scatter(bound_data[:, 0], bound_data[:, 1], c=colors, cmap='viridis')
    plt.colorbar(label='Element Index in the List')
    plt.title('2D Scatter Plot with Colors Representing Element Order')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(pjoin(data_dir, "viz_colmap_bound.png"))


@click.command()
@click.option("--data_dir", type = str)
@click.option("--file", type=str, default="cams_meta_colmap")
def main(data_dir, file):
    logger = init_logger(data_dir,file)
    cams = np.load(pjoin(data_dir,file + ".npy"))

    # cams_pose
    # [0:12] projection matrix
    # [12:21] 9 intrinsics
    # [21:25] 4 distortion
    # [25:27] 2 bounds, near / far
    poses = cams[:, 0:12].reshape(-1, 3, 4)
    intri = cams[:, 12:21]
    dist = cams[:, 21:25]
    bound = cams[:,-2:]

    xyz = poses[:, :, 3]


    logger.info(f"{xyz=}")
    logger.info(f"{np.max(xyz)=}")
    logger.info(f"{np.min(xyz)=}")

    logger.info(f"{intri=}")
    logger.info(f"{dist=}")

    logger.info(f"{bound=}")
    logger.info(f"{np.max(bound)=}")
    logger.info(f"{np.min(bound)=}")

    viz_bound(bound, data_dir)

if __name__ == "__main__":
    main()