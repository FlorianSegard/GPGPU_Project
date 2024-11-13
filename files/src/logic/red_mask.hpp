#pragma once


#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"
#include "../Compute.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void mask_init(Parameters* params);
void mask_process_frame(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride);


void red_mask_cu(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride);


#ifdef __cplusplus
}
#endif