#include <iostream>
#include <cuda_runtime.h>


__global__ void red_mask(bool* hysteresis_buffer, rgb* rgb_buffer, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        if (hysteresis_buffer[y * stride + x]) {
                    rgb_buffer[y * stride + x * 3] = min(255, rgb_buffer[y * stride + x * 3] + 0.5 * 255);
        }
    }
}