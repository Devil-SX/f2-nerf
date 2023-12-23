#!/bin/bash

# 递归函数来处理子目录
rename_folders() {
  local dir="$1"
  for task in $(ls $dir) ; do # Task
    for scene in $(ls $dir/$task) ; do # Scene    
        mv "$dir/$task/$scene/image" "$dir/$task/$scene/images"
        echo "rename!"
    done
  done
}

# 调用递归函数从当前目录开始
rename_folders "${HOME}/dataset/HoNY_nerf"
