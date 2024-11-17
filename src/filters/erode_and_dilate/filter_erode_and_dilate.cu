#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cfloat> // For FLT_MAX and FLT_MIN
#include "filter_erode_and_dilate.hpp"

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call)                                              \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

template <int opening_size>
__global__ void erode_shared(ImageView<float> input, ImageView<float> output, int width, int height) {
    extern __shared__ float smem[];

    const int smem_width = blockDim.x + 2 * opening_size;
    const int smem_height = blockDim.y + 2 * opening_size;

    // Compute base global indices for the shared memory tile
    int base_x = blockIdx.x * blockDim.x - opening_size;
    int base_y = blockIdx.y * blockDim.y - opening_size;

    int smem_x = threadIdx.x;
    int smem_y = threadIdx.y;

    // Load shared memory with coalesced access
    for (int y = smem_y; y < smem_height; y += blockDim.y) {
        int global_y = base_y + y;
        bool valid_y = (global_y >= 0) && (global_y < height);

        for (int x = smem_x; x < smem_width; x += blockDim.x) {
            int global_x = base_x + x;
            bool valid_x = (global_x >= 0) && (global_x < width);

            float value = FLT_MAX;
            if (valid_x && valid_y) {
                float* lineptr = (float*)((std::byte*)input.buffer + global_y * input.stride);
                value = lineptr[global_x];
            }
            smem[y * smem_width + x] = value;
        }
    }

    __syncthreads();

    // Now perform erosion using shared memory
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int smem_x = threadIdx.x + opening_size;
        int smem_y = threadIdx.y + opening_size;

        float min_val = FLT_MAX;
        #pragma unroll
        for (int dy = -opening_size; dy <= opening_size; ++dy) {
            #pragma unroll
            for (int dx = -opening_size; dx <= opening_size; ++dx) {
                int idx = (smem_y + dy) * smem_width + (smem_x + dx);
                float val = smem[idx];
                min_val = fminf(min_val, val);
            }
        }
        float* lineptr_out = (float*)((std::byte*)output.buffer + y * output.stride);
        lineptr_out[x] = min_val;
    }
}

template <int opening_size>
__global__ void dilate_shared(ImageView<float> input, ImageView<float> output, int width, int height) {
    extern __shared__ float smem[];

    const int smem_width = blockDim.x + 2 * opening_size;
    const int smem_height = blockDim.y + 2 * opening_size;

    // Compute base global indices for the shared memory tile
    int base_x = blockIdx.x * blockDim.x - opening_size;
    int base_y = blockIdx.y * blockDim.y - opening_size;

    int smem_x = threadIdx.x;
    int smem_y = threadIdx.y;

    // Load shared memory with coalesced access
    for (int y = smem_y; y < smem_height; y += blockDim.y) {
        int global_y = base_y + y;
        bool valid_y = (global_y >= 0) && (global_y < height);

        for (int x = smem_x; x < smem_width; x += blockDim.x) {
            int global_x = base_x + x;
            bool valid_x = (global_x >= 0) && (global_x < width);

            float value = -FLT_MAX;

            if (valid_x && valid_y) {
                float* lineptr = (float*)((std::byte*)input.buffer + global_y * input.stride);
                value = lineptr[global_x];
            }

            smem[y * smem_width + x] = value;
        }
    }

    __syncthreads();

    // Now perform dilation using shared memory
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int smem_x = threadIdx.x + opening_size;
        int smem_y = threadIdx.y + opening_size;

        float max_val = -FLT_MAX;
        #pragma unroll
        for (int dy = -opening_size; dy <= opening_size; ++dy) {
            #pragma unroll
            for (int dx = -opening_size; dx <= opening_size; ++dx) {
                int idx = (smem_y + dy) * smem_width + (smem_x + dx);
                float val = smem[idx];
                max_val = fmaxf(max_val, val);
            }
        }
        float* lineptr_out = (float*)((std::byte*)output.buffer + y * output.stride);
        lineptr_out[x] = max_val;
    }
}

void erode_cu(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size)
{
    dim3 threadsPerBlock(32, 32); // Adjusted thread block size
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int smem_size = (threadsPerBlock.x + 2 * opening_size) * (threadsPerBlock.y + 2 * opening_size) * sizeof(float);

    // Instantiate the kernel with the specific opening size
    switch (opening_size) {
        case 1:
            erode_shared<1><<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height);
            break;
        case 2:
            erode_shared<2><<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height);
            break;
        // Add more cases as needed
        default:
            // Handle unsupported opening sizes
            printf("Unsupported opening size: %d\n", opening_size);
            exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void dilate_cu(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size)
{
    dim3 threadsPerBlock(32, 32); // Adjusted thread block size
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int smem_size = (threadsPerBlock.x + 2 * opening_size) * (threadsPerBlock.y + 2 * opening_size) * sizeof(float);

    // Instantiate the kernel with the specific opening size
    switch (opening_size) {
        case 1:
            dilate_shared<1><<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height);
            break;
        case 2:
            dilate_shared<2><<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height);
            break;
        // Add more cases as needed
        default:
            // Handle unsupported opening sizes
            printf("Unsupported opening size: %d\n", opening_size);
            exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}
