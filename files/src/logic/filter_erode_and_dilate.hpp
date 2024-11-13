#pragma once


#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"
#include "../Compute.hpp"
// #include "labConverter.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void erode_cpp(const ImageView<float> input, lab* output, int width, int height, std::ptrdiff_t stride);
void dilate_cpp(const ImageView<float> input, lab* output, int width, int height, std::ptrdiff_t stride);

void erode_cu(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride);
void dilate_cu(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride);

void filter_init(Parameters *params);
void erode_process_frame(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride);
void dilate_process_frame(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride);

#ifdef __cplusplus
}
#endif
