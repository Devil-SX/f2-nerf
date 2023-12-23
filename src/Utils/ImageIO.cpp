//
// Created by ppwang on 2023/4/4.
//

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <torch/torch.h>
#include "../Common.h"
#include "Utils.h"
#include "cnpy.h"

using Tensor = torch::Tensor;

Tensor Utils::ReadImageTensor(const std::string& path) {
  int w, h, n;
  unsigned char *idata = stbi_load(path.c_str(), &w, &h, &n, 0);

  Tensor img = torch::empty({ h, w, n }, CPUUInt8);
  std::memcpy(img.data_ptr(), idata, w * h * n);
  stbi_image_free(idata);

  img = img.to(torch::kFloat32).to(torch::kCPU) / 255.f;
  return img;
}

bool Utils::WriteImageTensor(const std::string &path, Tensor img) {
  Tensor out_img = (img * 255.f).clip(0.f, 255.f).to(torch::kUInt8).to(torch::kCPU).contiguous();
  stbi_write_png(path.c_str(), out_img.size(1), out_img.size(0), out_img.size(2), out_img.data_ptr(), 0);
  return true;
}

Tensor Utils::ReadDepthTensor(const std::string& path) {
  cnpy::NpyArray arr = cnpy::npy_load(path.c_str());
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  Tensor depth = torch::from_blob(arr.data<float>(), arr.num_vals, options).contiguous();
  return depth;
}

bool Utils::WriteDepthTensor(const std::string &path, Tensor img) {
  img = img.clamp(0.f, 2.0e16 - 1).to(torch::kInt32);
  Tensor imgRGB = torch::empty({ img.size(0), img.size(1), 3 }, CPUUInt8);
  imgRGB.select(2, 0) = img.bitwise_right_shift(16).bitwise_and(255).to(torch::kUInt8);
  imgRGB.select(2, 1) = img.bitwise_right_shift(8).bitwise_and(255).to(torch::kUInt8);
  imgRGB.select(2, 2) = img.bitwise_and(255).to(torch::kUInt8);
  imgRGB = imgRGB.contiguous();

  stbi_write_png(path.c_str(), img.size(1), img.size(0), img.size(2), imgRGB.data_ptr(), 0);
  return true;
}