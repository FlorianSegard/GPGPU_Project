#pragma once

#include <cuda_runtime.h>

#include "stb_image.h"
#include "Image.hpp"

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

void filter_impl_cu(uint8_t* pixels_buffer, int width, int height,
                 int plane_stride);

#ifdef __cplusplus
}
#endif