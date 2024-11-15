#pragma once


#include <cmath>
#include "../../common/Image.hpp"
#include "../../filters/filter_impl.hpp"
#include "../../Compute.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void mask_init(Parameters* params);
void mask_process_frame(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height);


void red_mask_cu(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height);


#ifdef __cplusplus
}
#endif
