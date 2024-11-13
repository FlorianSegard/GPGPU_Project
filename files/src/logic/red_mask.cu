#include <iostream>
#include <cuda_runtime.h>
#include "red_mask.hpp"


__global__ void red_mask_kernel(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    bool hyst_value = (bool*)((std::byte*)hysteresis_buffer.buffer + y * hysteresis_buffer.stride)[x];
    rgb8* rgb_value = (rgb8*)((std::byte*)rgb_buffer.buffer + y * rgb_buffer.stride);

    rgb_value[x].r = rgb_value[x].r / 2 + (hyst_value ? 127 : 0);
    rgb_value[x].g = rgb_value[x].g / 2;
    rgb_value[x].b = rgb_value[x].b / 2;
}

void red_mask_cu(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    red_mask_kernel<<<blocksPerGrid, threadsPerBlock>>>(hysteresis_buffer, rgb_buffer, width, height);
}