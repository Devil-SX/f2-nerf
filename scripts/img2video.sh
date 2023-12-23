#!/bin/bash

# 图片文件夹路径
image_folder="./exp/half_pick/test/novel_images"

# 视频输出文件的名称
output_video="./exp/half_pick/novel_view.mp4"

# 检查文件夹是否为空
if [ -z "$(ls -A $image_folder)" ]; then
   echo "指定的文件夹为空。"
   exit 1
fi

# 使用 ffmpeg 创建视频
ffmpeg -framerate 30 -pattern_type glob -i "$image_folder/*.png" -c:v mpeg4 -pix_fmt yuv420p "$output_video"

echo "视频已创建完成，保存为：$output_video"
