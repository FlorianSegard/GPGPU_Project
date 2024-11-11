#pragma once

#include <cuda_runtime.h>
#include "Image.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <cstdint>

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

void filter_impl(uint8_t* pixels_buffer, int width, int height,
                 int plane_stride, int pixel_stride);

#ifdef __cplusplus
}
#endif