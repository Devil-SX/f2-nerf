import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import click
from os.path import join as pjoin

# cams_path = "./data/Air_Fryer_Opening/haritheja_Env1_2023-11-21--13-34-39/cams_meta.npy"

# def visualize_poses(poses, path="pic"):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     for pose in poses:
#         x,y,z = pose[:,3]
#         # print(pose[:,3])
#         rot_matrix = pose[:,:3]

#         direction = np.dot(rot_matrix, np.array([1, 0, 0]))

#         ax.scatter(x, y, z, color='blue')

#         # 绘制方向射线
#         ax.quiver(x, y, z, direction[0], direction[1], direction[2], length=0.05, color='red')

#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     plt.title('3D Pose Visualization with Directions')

#     plt.savefig(path)


@click.command()
@click.option("--data_dir", type=str)
@click.option("--cam_file", type=str, default="cams_meta.npy")
def main(data_dir,cam_file):
    cams = np.load(pjoin(data_dir, cam_file))
    cams_pose = cams[:, :12].reshape(-1, 3, 4)
    
    translation = cams_pose[:, :, 3]
    mean = np.mean(translation, axis=0)
    bias = translation - mean
    radius = np.linalg.norm(bias, ord=2, axis=-1).max()
    norm_translation = bias / radius
    norm_cams_pose = np.concatenate((cams_pose[:,:,:3],norm_translation[...,np.newaxis]), axis=-1)

    np.savetxt(pjoin(data_dir,cam_file.split(".")[0] + ".kitti"), cams_pose.reshape(-1,12))
    np.savetxt(pjoin(data_dir,cam_file.split(".")[0] + "_norm.kitti"), norm_cams_pose.reshape(-1,12))

    # visualize_poses(cams_pose, pjoin(data_dir,'viz.png'))
    # visualize_poses(norm_cams_pose, pjoin(data_dir,'viz_norm.png'))

if __name__ == "__main__":
    main()
