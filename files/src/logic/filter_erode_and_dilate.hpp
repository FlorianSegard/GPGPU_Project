#pragma once


#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"
#include "../Compute.hpp"
// #include "labConverter.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void erode_cpp(const lab* input, lab* output, int width, int height, std::ptrdiff_t stride);
void dilate_cpp(const lab* input, lab* output, int width, int height, std::ptrdiff_t stride);

void erode_cu(float* input, float* output, int width, int height, std::ptrdiff_t stride);
void dilate_cu(float* input, float* output, int width, int height, std::ptrdiff_t stride);

void filter_init(Parameters *params);
void erode_process_frame(float* input, float* output, int width, int height, std::ptrdiff_t stride);
void dilate_process_frame(float* input, float* output, int width, int height, std::ptrdiff_t stride);

#ifdef __cplusplus
}
#endif
