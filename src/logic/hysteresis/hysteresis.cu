#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "hysteresis.hpp"

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define BLOCK_SIZE 30
#define HYSTERESIS_TILE_WIDTH (BLOCK_SIZE + 2)
#define LOWER_THRESHOLD 4.0
#define UPPER_THRESHOLD 30.0


__global__ void hysteresis_thresholding(ImageView<float> input, ImageView<bool> output, int width, int height, float threshold)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_x = blockIdx.x * blockDim.x;
    int tile_y = blockIdx.y * blockDim.y;

    int x = tile_x + tx;
    int y = tile_y + ty;

    if (x >= width || y >= height)
        return;

    // On charge la tuile
    float* input_lineptr = (float *)((std::byte*)input.buffer + y * input.stride);

    // On applique le seuil
    bool out_val = input_lineptr[x] > threshold;

    // On stocke le résultat dans la sortie
    bool *output_lineptr = (bool *)((std::byte*)output.buffer + y * output.stride);
    output_lineptr[x] = out_val;
}

__global__ void hysteresis_kernel(ImageView<bool> upper, ImageView<bool> lower, int width, int height, bool *has_changed_global)
{
    __shared__ bool tile_upper[HYSTERESIS_TILE_WIDTH][HYSTERESIS_TILE_WIDTH];
    __shared__ bool tile_lower[HYSTERESIS_TILE_WIDTH][HYSTERESIS_TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate global coordinates adjusted for halo
    int x = blockIdx.x * BLOCK_SIZE + tx - 1;
    int y = blockIdx.y * BLOCK_SIZE + ty - 1;

    // Load data into shared memory with boundary checks
    bool upper_value = false;
    bool lower_value = true;
    
    bool* upper_lineptr = (bool *)((std::byte*)upper.buffer + y * upper.stride);

    if (x >= 0 && x < width && y >= 0 && y < height)
    {
        bool* lower_lineptr = (bool *)((std::byte*)lower.buffer + y * lower.stride);
        upper_value = upper_lineptr[x];
        lower_value = lower_lineptr[x];
    }

    tile_upper[ty][tx] = upper_value;
    tile_lower[ty][tx] = lower_value;

    if (x >= width - 1 || y >= height - 1 || x == 0 || y == 0)
        return;

    if (tile_upper[ty][tx])
        return;

    if (!tile_lower[ty][tx])
        return;

    __syncthreads();

    // Only process inner pixels
    if (tx > 0 && tx < HYSTERESIS_TILE_WIDTH - 1 && ty > 0 && ty < HYSTERESIS_TILE_WIDTH - 1)
    {

        if (tile_upper[ty][tx - 1])
        {
            upper_lineptr[x] = true;
            *has_changed_global = true;
        }

        if (tile_upper[ty][tx + 1])
        {
            upper_lineptr[x] = true;
            *has_changed_global = true;
        }
        if (tile_upper[ty - 1][tx])
        {
            upper_lineptr[x] = true;
            *has_changed_global = true;
        }
        if (tile_upper[ty + 1][tx])
        {
            upper_lineptr[x] = true;
            *has_changed_global = true;
        }
        return;
    }
    if (upper_lineptr[x - 1])
    {
        upper_lineptr[x] = true;
        *has_changed_global = true;
    }
    if (upper_lineptr[x - 1])
    {
        upper_lineptr[x] = true;
        *has_changed_global = true;
    }
    if ((bool *)((std::byte*)upper.buffer + (y - 1) * upper.stride)[x])
    {
        upper_lineptr[x] = true;
        *has_changed_global = true;
    }
    if ((bool *)((std::byte*)upper.buffer + (y + 1) * upper.stride)[x])
    {
        upper_lineptr[x] = true;
        *has_changed_global = true;
    }

}



void hysteresis_cu(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold)
{
    dim3 blockSize(32, 32);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

    Image<bool> lower_threshold_input(width, height, true);

    // seuil inf et sup
    hysteresis_thresholding<<<gridSize, blockSize>>>(opened_input, lower_threshold_input, width, height, lower_threshold);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    hysteresis_thresholding<<<gridSize, blockSize>>>(opened_input, hysteresis, width, height, upper_threshold);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    bool h_has_changed = 1;

    // flag de changement
    bool *d_has_changed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_has_changed, sizeof(bool)));


    // on propage sur l'image.
    while (h_has_changed)
    {
        CHECK_CUDA_ERROR(cudaMemset(d_has_changed, false, sizeof(bool)));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        hysteresis_kernel<<<gridSize, blockSize>>>(hysteresis, lower_threshold_input, width, height, d_has_changed);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(&h_has_changed, d_has_changed, sizeof(bool), cudaMemcpyDeviceToHost));
    }

    //printf("%d\n", i);
    cudaFree(d_has_changed);
}

