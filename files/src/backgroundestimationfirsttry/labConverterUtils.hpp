#pragma once
#include <cmath>

// Shared function to convert RGB to XYZ color space.
__host__ __device__ void XYZ_color_space(float r_linear, float g_linear, float b_linear, float* result) {
    float matrix[9] = { 0.4124564, 0.3575761, 0.1804375,
                        0.2126729, 0.7151522, 0.0721750,
                        0.0193339, 0.1191920, 0.9503041 };

    result[0] = matrix[0] * r_linear + matrix[1] * g_linear + matrix[2] * b_linear;
    result[1] = matrix[3] * r_linear + matrix[4] * g_linear + matrix[5] * b_linear;
    result[2] = matrix[6] * r_linear + matrix[7] * g_linear + matrix[8] * b_linear;
}

// Function to convert sRGB to linear RGB.
__host__ __device__ float get_linear(float r_g_b) {
    if (r_g_b <= 0.04045f) {
        return r_g_b / 12.92f;
    }
    return powf((r_g_b + 0.055f) / 1.055f, 2.4f);
}

// Helper function for lab color space conversion.
__host__ __device__ float f(float t) {
    if (t > powf(6.0f/29.0f, 3.0f)) {
        return powf(t, 1.0f/3.0f);
    } else {
        return (1.0f/3.0f) * powf(29.0f/6.0f, 2.0f) * t + 4.0f/29.0f;
    }
}
