#include <iostream>
#include <cuda_runtime.h>
#include "red_mask.hpp"


__global__ void red_mask_kernel(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        if (hysteresis_buffer[y * stride + x]) {
                    rgb_buffer[y * stride + x * 3] = min(255, rgb_buffer[y * stride + x * 3] + 0.5 * 255);
        }
    }
}

void red_mask_cu(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    red_mask_kernel<<<blocksPerGrid, threadsPerBlock>>>(hysteresis_buffer, rgb_buffer, width, height, stride);
}