原本的 README 东西太多我把咱们实验的内容理在整理

# 工程目录管理

- `data`
  - `dataset_name` 如 Air_Fryer_Opening
    - `case_name` 如 haritheja_Env1_2023-11-21--13-34-39
      - 原始 dataset 文件
        - `images` 原始视频分割成的照片 (Homes of New York，256x256)
        - `compressed_np_depth_float32.bin` (Homes of New York) 每一帧深度信息(256x196)
        - `labels.json` (Homes of New York) 每一帧的 7 自由度信息
      - 生成文件
        - `dense` 深度信息，每一帧用一个 npy 保存，由 `compressed_np_depth_float32.bin` 插值到 256x256 生成
        - `images_x` 缩小的图片， x 代表缩小的倍数，训练时可用 `dataset.factor` 参数选择使用的图片集
        - `camera_mask.png` COLMAP Mask 图片
        - `sparse` COLMAP 生成的二进制相机模型
        - `evo` 手动保留不同方法生成的位姿 `npy` 文件
        - `cams_meta.npy` 保存相机位姿以及内参等信息，用于渲染
        - `pose_render.npy` 渲染时的相机位姿轨迹
        - `depth_list.txt` 训练时根据这个文件读取深度顺序
        - `image_list.txt` 训练时根据这个文件读取图片顺序
- `exp`
  - `case_name` （不同 dataset 中相同 case 会被覆盖）
    - `novel_images` 放 New View Synpthesis 图片和拼接的视频


# 流程

生成 `cams_meta.npy` 文件
- 方法1： `file2poses.py`  （要求存在 COLMAP 已生成好的 `cams_meta.npy`）
- 方法2: `my_colmap_script.sh`-> `colmap2poses.py` 

训练
- `run.py` 自动根据 `dataset.with_depth` 生成深度文件，然后生成 `runtime_config.yaml` C++ 读取这个文件运行

生成渲染视频
- `inter_poses.py` 生成渲染轨迹
- `run.py` 生成渲染图片
- `imgs2mp4.py` 生成渲染视频

# Case 1 

Air_Fryer_Opening / haritheja_Env1_2023-11-21--13-34-39

## Preprocess
export CUDA_VISIBLE_DEVICES=2
bash scripts/resize.sh ./data/Air_Fryer_Opening/haritheja_Env1_2023-11-21--13-34-39
python scripts/file2poses.py --data_dir ./data/Air_Fryer_Opening/haritheja_Env1_2023-11-21--13-34-39

## Train
python scripts/run.py --config-name=wanjinyou dataset_name=Air_Fryer_Opening case_name=haritheja_Env1_2023-11-21--13-34-39 mode=train +work_dir=$(pwd) dataset.factor=1  
//train.learning_rate=5e-3


python scripts/inter_poses.py --data_dir ./data/Air_Fryer_Opening/haritheja_Env1_2023-11-21--13-34-39 --key_poses 5,10,15,20,25,30,35,40,45,50,55,60 --n_out_poses 300
python scripts/run.py --config-name=wanjinyou dataset_name=Air_Fryer_Opening case_name=haritheja_Env1_2023-11-21--13-34-39 mode=render_path is_continue=true +work_dir=$(pwd) 
python scripts/run.py --config-name=wanjinyou dataset_name=Air_Fryer_Opening case_name=haritheja_Env1_2023-11-21--13-34-39 mode=render_path is_continue=true +work_dir=$(pwd) dataset.factor=1 

python scripts/imgs2mp4.py --data_dir /home/DISCOVER_summer2022/cuily/workspace/f2-nerf/exp/haritheja_Env1_2023-11-21--13-34-39/test/novel_images

