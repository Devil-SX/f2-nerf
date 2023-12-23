#!/bin/bash

# 图片文件夹路径
image_folder="./exp/half_pick/test/images"

# 输出文件的名称
output_image="output.png"

# 进入图片文件夹
cd "$image_folder"

# 检查文件夹是否为空
# if [ -z "$(ls -A $image_folder)" ]; then
#    echo "指定的文件夹为空。"
#    exit 1
# fi

# 如果不删除，convert 命令会直接在原来的图片上接着拼接
rm "$output_image"
# 使用 ImageMagick 的 convert 命令来拼接图片
convert -append $(ls -v) "$output_image"

echo "图片已经拼接完成，保存为：$output_image"
