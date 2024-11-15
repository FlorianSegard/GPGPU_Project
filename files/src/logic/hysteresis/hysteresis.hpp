#pragma once

#include <cstddef>
#include "../../Image.hpp"
#include "../../filters/filter_impl.hpp"
#include "../../Compute.hpp"

#ifdef __cplusplus
extern "C" {
#endif


void hysteresis_cpp(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold);
void hysteresis_cu(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold);

void hysteresis_init(Parameters* params);
void hysteresis_process_frame(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold);

#ifdef __cplusplus
}
#endif