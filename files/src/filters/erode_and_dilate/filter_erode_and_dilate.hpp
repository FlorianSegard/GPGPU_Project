#pragma once


#include <cmath>
#include "../../common/Image.hpp"
#include "../filter_impl.hpp"
#include "../../Compute.hpp"
// #include "labConverter.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void erode_cpp(const ImageView<float> input, lab* output, int width, int height);
void dilate_cpp(const ImageView<float> input, lab* output, int width, int height);

void erode_cu(ImageView<float> input, ImageView<float> output, int width, int height);
void dilate_cu(ImageView<float> input, ImageView<float> output, int width, int height);

void filter_init(Parameters *params);
void erode_process_frame(ImageView<float> input, ImageView<float> output, int width, int height);
void dilate_process_frame(ImageView<float> input, ImageView<float> output, int width, int height);

#ifdef __cplusplus
}
#endif
