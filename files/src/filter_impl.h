#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <gst/gst.h>

// RGB color structure
struct rgb8 {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

// LAB color structure
struct lab {
    float L;
    float a;
    float b;
};

// CUDA kernel declarations
__global__ void rgbtolab_converter_GPU(rgb8* input, size_t input_pitch,
                                       lab* output, size_t output_pitch,
                                       int width, int height);

__global__ void residual_image(lab* bg_ref, size_t bg_pitch,
                               lab* current, size_t current_pitch,
                               lab* residual, size_t residual_pitch,
                               int width, int height);

__global__ void check_background_GPU(lab* current, size_t current_pitch,
                                     lab* bg_ref, size_t bg_ref_pitch,
                                     lab* candidate_bg, size_t candidate_pitch,
                                     int* time_pixels, size_t time_pitch,
                                     int width, int height);

__global__ void erode(lab* input, lab* output,
                      int width, int height, size_t pitch);

__global__ void dilate(lab* input, lab* output,
                       int width, int height, size_t pitch);

__global__ void hysteresis_reconstruction(lab* input, bool* output,
                                          int width, int height, size_t pitch);

__global__ void apply_mask(rgb8* input, bool* mask,
                           int width, int height, size_t pitch);

#ifdef __cplusplus
extern "C" {
#endif

void filter_impl(uint8_t* pixels_buffer, int width, int height,
                 int plane_stride, int pixel_stride, GstClockTime timestamp);

#ifdef __cplusplus
}
#endif