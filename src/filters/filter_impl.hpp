#pragma once

#include <cuda_runtime.h>

#include "../common/stb_image.h"
#include "../common/Image.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// RGB color structure
// struct rgb8 {
//     uint8_t r;
//     uint8_t g;
//     uint8_t b;
// };

// LAB color structure
struct lab {
    float L;
    float a;
    float b;
};

void filter_impl(uint8_t* pixels_buffer, int width, int height, int plane_stride, const char* bg_uri,
                     int opening_size, int th_low, int th_high, int bg_sampling_rate, int bg_number_frame);

#ifdef __cplusplus
}
#endif