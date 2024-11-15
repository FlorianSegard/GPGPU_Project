#pragma once

#include <cstddef>
#include "../Image.hpp"
#include "../filter_impl.h"
#include "../Compute.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// BASELINE
void hysteresis_cpp(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold);
void hysteresis_cu(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold);

// OPTIMIZED
void hysteresis_cu_optimized(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold);

// PIPELINE
void hysteresis_init(Parameters* params);
void hysteresis_process_frame(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold);

#ifdef __cplusplus
}
#endif
