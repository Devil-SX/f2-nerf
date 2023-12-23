# 计算 bound 和 depth 之间线性拟合的值
# 通过 COLMAP 和 iPhone 采集数据做比较
import os
from copy import deepcopy
from glob import glob
from os.path import join as pjoin
from typing import Mapping, Optional, Sequence, Text, Tuple, Union

import camera_utils
import click
import cv2 as cv
import liblzfse
import numpy as np
from colmap_warpper.pycolmap.scene_manager import SceneManager
from depths import *
from matplotlib import pyplot as plt


# This implementation is from MipNeRF360
class NeRFSceneManager(SceneManager):
    """COLMAP pose loader.

    Minor NeRF-specific extension to the third_party Python COLMAP loader:
    google3/third_party/py/pycolmap/scene_manager.py
    """

    def __init__(self, data_dir):
        # COLMAP
        if os.path.exists(pjoin(data_dir, "sparse", "0")):
            sfm_dir = pjoin(data_dir, "sparse", "0")
        # if os.path.exists(pjoin(data_dir, 'sparse', '1')):
        # sfm_dir = pjoin(data_dir, 'sparse', '1')

        # hloc
        else:
            sfm_dir = pjoin(data_dir, "hloc_sfm")

        assert os.path.exists(sfm_dir)
        super(NeRFSceneManager, self).__init__(sfm_dir)

    def process(
        self,
    ) -> Tuple[
        Sequence[Text],
        np.ndarray,
        np.ndarray,
        Optional[Mapping[Text, float]],
        camera_utils.ProjectionType,
    ]:
        """Applies NeRF-specific postprocessing to the loaded pose data.

        Returns:
          a tuple [image_names, poses, pixtocam, distortion_params].
          image_names:  contains the only the basename of the images.
          poses: [N, 4, 4] array containing the camera to world matrices.
          pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
          distortion_params: mapping of distortion param name to distortion
            parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
        """

        self.load_cameras()
        self.load_images()
        self.load_points3D()

        # Assume shared intrinsics between all cameras.
        # print(self.cameras)
        cam = self.cameras[1]

        # Extract focal lengths and principal point parameters.
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :3, :4]

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        names = [imdata[k].name for k in imdata]

        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        poses = poses @ np.diag([1, -1, -1, 1])
        # pixtocam = np.diag([1, -1, -1]) @ pixtocam

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 1 or type_ == "PINHOLE":
            params = None
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        if type_ == 2 or type_ == "SIMPLE_RADIAL":
            params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
            params["k1"] = cam.k1
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 3 or type_ == "RADIAL":
            params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
            params["k1"] = cam.k1
            params["k2"] = cam.k2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 4 or type_ == "OPENCV":
            params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
            params["k1"] = cam.k1
            params["k2"] = cam.k2
            params["p1"] = cam.p1
            params["p2"] = cam.p2
            camtype = camera_utils.ProjectionType.PERSPECTIVE

        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            params = {k: 0.0 for k in ["k1", "k2", "k3", "k4"]}
            params["k1"] = cam.k1
            params["k2"] = cam.k2
            params["k3"] = cam.k3
            params["k4"] = cam.k4
            camtype = camera_utils.ProjectionType.FISHEYE

        return names, poses, pixtocam, params, camtype


class Dataset:
    def __init__(self, data_dir):
        scene_manager = NeRFSceneManager(data_dir)
        (
            self.names,
            self.poses,
            self.pix2cam,
            self.params,
            self.camtype,
        ) = scene_manager.process()
        self.cam2pix = np.linalg.inv(self.pix2cam)
        self.n_images = len(self.poses)

        # re-permute images by name
        sorted_image_names = sorted(deepcopy(self.names))
        sort_img_idx = []
        for i in range(self.n_images):
            sort_img_idx.append(self.names.index(sorted_image_names[i]))
        img_idx = np.array(sort_img_idx, dtype=np.int32)
        self.poses = self.poses[sort_img_idx]

        # calc near-far bounds
        self.bounds = np.zeros([self.n_images, 2], dtype=np.float32)
        name_to_ids = scene_manager.name_to_image_id
        points3D = scene_manager.points3D
        points3D_ids = scene_manager.point3D_ids
        point3D_id_to_images = scene_manager.point3D_id_to_images
        # image_id_to_image_idx = np.zeros(self.n_images + 10, dtype=np.int32)
        image_id_to_image_idx = np.zeros(
            len(os.listdir(pjoin(data_dir, "images"))) + 10, dtype=np.int32
        )
        for image_name in self.names:
            image_id_to_image_idx[name_to_ids[image_name]] = sorted_image_names.index(
                image_name
            )
        vis_arr = []
        for pts_i in range(len(points3D)):
            cams = np.zeros([self.n_images], dtype=np.uint8)
            images_ids = point3D_id_to_images[points3D_ids[pts_i]]
            for image_info in images_ids:
                image_id = image_info[0]
                image_idx = image_id_to_image_idx[image_id]
                cams[image_idx] = 1
            vis_arr.append(cams)

        vis_arr = np.stack(vis_arr, 1)  # [n_images, n_pts ]

        self.xyz_list = []
        for img_i in range(self.n_images):
            vis = vis_arr[img_i]
            pts = points3D[vis == 1]
            # print(pts.shape) # [n_pts, 3]
            c2w = np.diag([1.0, 1.0, 1.0, 1.0])
            c2w[:3, :4] = self.poses[img_i]
            w2c = np.linalg.inv(c2w)
            xyz = (w2c[None, :3, :3] @ pts[..., None])[..., 0] + w2c[
                None, :3, 3
            ]  # [n_pts, 3] of camera coordinate
            cam2pix = np.linalg.inv(self.pix2cam) # [3,3]
            xyz_pix = (cam2pix @ xyz[...,None])[...,0] / xyz[:,2:]
            xyz = np.hstack([xyz_pix[:,:2], xyz[:,2:]])
            self.xyz_list.append(xyz)

        # Move all to numpy
        def proc(x):
            return np.ascontiguousarray(np.array(x).astype(np.float64))

        self.poses = proc(self.poses)
        self.cam2pix = proc(np.tile(self.cam2pix[None], (len(self.poses), 1, 1)))
        self.bounds = proc(self.bounds)
        if self.params is not None:
            dist_params = [
                self.params["k1"],
                self.params["k2"],
                self.params["p1"],
                self.params["p2"],
            ]
        else:
            dist_params = [0.0, 0.0, 0.0, 0.0]
        dist_params = np.tile(np.array(dist_params), len(self.poses)).reshape(
            [len(self.poses), -1]
        )
        self.dist_params = proc([dist_params])

    def get_xyz_list(self):
        return self.xyz_list


def depth_to_bound(depth):
    depth = depth.reshape((-1, 192 * 256))
    depth_min = np.min(depth, axis=-1)[..., np.newaxis]
    depth_max = np.max(depth, axis=-1)[..., np.newaxis]
    return np.hstack([depth_min, depth_max])  # [N,2]


if __name__ == "__main__":
    data_dir = "/home/DISCOVER_summer2022/cuily/workspace/f2-nerf/data/Air_Fryer_Opening/haritheja_Env1_2023-11-21--13-34-39"
    depth_file = pjoin(data_dir, "compressed_np_depth_float32.bin")
    colmap_file = pjoin(data_dir, "evo", "cams_meta_colmap.npy")

    # Read from Colmap
    dataset = Dataset(data_dir)
    xyz = dataset.get_xyz_list() # n_images of [n_pts, 3]

    colmap_image = np.zeros((256,256))
    for point in xyz[0]:
        colmap_image[int(point[0]), 255 - int(point[1])] = -point[2]
    plt.imshow(colmap_image)
    plt.savefig("colmap_depth.png")
    plt.clf()


    # Read from iPhone
    depth = read_depth_from_bin(data_dir)
    depth = bilinear_interpolation(depth, (256, 256))  # [n_images, 256, 256]

    plt.imshow(depth[0])
    plt.savefig("iphone_depth.png")
    plt.clf()

    iphone_depth = []
    
    for i,depth_per_image in enumerate(depth):
        # slice_x = 255 - xyz[i][:,0].astype(int) # [n_pts]
        slice_x = xyz[i][:,0].astype(int) # [n_pts]
        # slice_y = xyz[i][:,1].astype(int)
        slice_y = 255 - xyz[i][:,1].astype(int)
        iphone_depth.append(depth_per_image[slice_x, slice_y])
    
    t = np.arange(0, xyz[0].shape[0])
    plt.plot(t, -xyz[0][:,2], label="colmap")
    plt.savefig("colmap.png")
    plt.clf()
    plt.plot(t, iphone_depth[0], label="iphone")
    plt.savefig("iphone.png")


    colmap_depth = -np.vstack(xyz)[:,2]
    iphone_depth = np.concatenate(iphone_depth)
    
    
    x = iphone_depth
    y = colmap_depth

    coefficents = np.polyfit(x, y, 1)
    slope, intercept = coefficents
    print(f"{slope=} /t {intercept=}")
    # mean_rate = np.mean(bound_col/bound_dep, axis=0)

    y_predicted = slope * x + intercept
    total_variance = np.sum((y - np.mean(y)) ** 2)
    explained_variance = np.sum((y_predicted - np.mean(y)) ** 2)
    r_squared = explained_variance / total_variance
    print(f"R^2: {r_squared}")


