#include <iostream>
#include <cuda_runtime.h>
#include "red_mask.hpp"


__global__ void red_mask_kernel(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    bool hyst_value = (bool*)((std::byte*)hysteresis_buffer.buffer + y * hysteresis_buffer.stride)[x];
    rgb8 rgb_value = (rgb8*)((std::byte*)rgb_buffer.buffer + y * rgb_buffer.stride)[x];

    rgb_value.r = rgb_value.r / 2 + (hyst_value[x] ? 127 : 0);
    rgb_value.g = rgb_value.g / 2;
    rgb_value.b = rgb_value.b / 2;

    rgb8* lineptr_rgb= (rgb8*)((std::byte*)rgb_buffer.buffer + y * rgb_buffer.stride);
    lineptr_rgb[x] = rgb_value;
}

void red_mask_cu(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    red_mask_kernel<<<blocksPerGrid, threadsPerBlock>>>(hysteresis_buffer, rgb_buffer, width, height, stride);
}