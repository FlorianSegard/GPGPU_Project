#pragma once


#include <cmath>
#include "../../common/Image.hpp"
#include "../filter_impl.hpp"
#include "../../Compute.hpp"
// #include "labConverter.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void erode_cpp(const ImageView<float> input, lab* output, int width, int height, int opening_size);
void dilate_cpp(const ImageView<float> input, lab* output, int width, int height, int opening_size);

void erode_cu(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size);
void dilate_cu(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size);

void filter_init(Parameters *params);
void erode_process_frame(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size);
void dilate_process_frame(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size);

#ifdef __cplusplus
}
#endif
