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

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define RADIUS 1 // For a 1-pixel halo
#define TILE_SIZE_X (BLOCK_SIZE_X + 2 * RADIUS)
#define TILE_SIZE_Y (BLOCK_SIZE_Y + 2 * RADIUS)
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
    __shared__ bool tile_upper[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ bool tile_lower[TILE_SIZE_Y][TILE_SIZE_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE_X + tx;
    int y = blockIdx.y * BLOCK_SIZE_Y + ty;

    int smem_x = tx + RADIUS;
    int smem_y = ty + RADIUS;

    // Initialize shared memory with zeros
    tile_upper[smem_y][smem_x] = false;
    tile_lower[smem_y][smem_x] = false;

    // Load central data
    if (x < width && y < height)
    {
        bool* upper_lineptr = (bool *)((std::byte*)upper.buffer + y * upper.stride);
        bool* lower_lineptr = (bool *)((std::byte*)lower.buffer + y * lower.stride);
        tile_upper[smem_y][smem_x] = upper_lineptr[x];
        tile_lower[smem_y][smem_x] = lower_lineptr[x];
    }

    // Load halo regions
    // Left and right halos
    if (tx < RADIUS)
    {
        int halo_x_left = x - RADIUS;
        int halo_x_right = x + BLOCK_SIZE_X;

        if (halo_x_left >= 0 && y < height)
        {
            tile_upper[smem_y][smem_x - RADIUS] = upper.buffer[y * upper.stride + halo_x_left];
            tile_lower[smem_y][smem_x - RADIUS] = lower.buffer[y * lower.stride + halo_x_left];
        }
        if (halo_x_right < width && y < height)
        {
            tile_upper[smem_y][smem_x + BLOCK_SIZE_X] = upper.buffer[y * upper.stride + halo_x_right];
            tile_lower[smem_y][smem_x + BLOCK_SIZE_X] = lower.buffer[y * lower.stride + halo_x_right];
        }
    }

    // Top and bottom halos
    if (ty < RADIUS)
    {
        int halo_y_top = y - RADIUS;
        int halo_y_bottom = y + BLOCK_SIZE_Y;

        if (halo_y_top >= 0 && x < width)
        {
            tile_upper[smem_y - RADIUS][smem_x] = upper.buffer[halo_y_top * upper.stride + x];
            tile_lower[smem_y - RADIUS][smem_x] = lower.buffer[halo_y_top * lower.stride + x];
        }
        if (halo_y_bottom < height && x < width)
        {
            tile_upper[smem_y + BLOCK_SIZE_Y][smem_x] = upper.buffer[halo_y_bottom * upper.stride + x];
            tile_lower[smem_y + BLOCK_SIZE_Y][smem_x] = lower.buffer[halo_y_bottom * lower.stride + x];
        }
    }

    // Load corner halos
    if (tx < RADIUS && ty < RADIUS)
    {
        // Top-left corner
        int halo_x = x - RADIUS;
        int halo_y = y - RADIUS;
        if (halo_x >= 0 && halo_y >= 0)
        {
            tile_upper[smem_y - RADIUS][smem_x - RADIUS] = upper.buffer[halo_y * upper.stride + halo_x];
            tile_lower[smem_y - RADIUS][smem_x - RADIUS] = lower.buffer[halo_y * lower.stride + halo_x];
        }
        // Bottom-right corner
        halo_x = x + BLOCK_SIZE_X;
        halo_y = y + BLOCK_SIZE_Y;
        if (halo_x < width && halo_y < height)
        {
            tile_upper[smem_y + BLOCK_SIZE_Y][smem_x + BLOCK_SIZE_X] = upper.buffer[halo_y * upper.stride + halo_x];
            tile_lower[smem_y + BLOCK_SIZE_Y][smem_x + BLOCK_SIZE_X] = lower.buffer[halo_y * lower.stride + halo_x];
        }
        // Top-right corner
        halo_x = x + BLOCK_SIZE_X;
        halo_y = y - RADIUS;
        if (halo_x < width && halo_y >= 0)
        {
            tile_upper[smem_y - RADIUS][smem_x + BLOCK_SIZE_X] = upper.buffer[halo_y * upper.stride + halo_x];
            tile_lower[smem_y - RADIUS][smem_x + BLOCK_SIZE_X] = lower.buffer[halo_y * lower.stride + halo_x];
        }
        // Bottom-left corner
        halo_x = x - RADIUS;
        halo_y = y + BLOCK_SIZE_Y;
        if (halo_x >= 0 && halo_y < height)
        {
            tile_upper[smem_y + BLOCK_SIZE_Y][smem_x - RADIUS] = upper.buffer[halo_y * upper.stride + halo_x];
            tile_lower[smem_y + BLOCK_SIZE_Y][smem_x - RADIUS] = lower.buffer[halo_y * lower.stride + halo_x];
        }
    }

    __syncthreads();


    if (tile_upper[smem_y][smem_x])
        return;

        // Si le pixel n'est pas marqué dans l'image inférieure, on passe au suivant
    if (!tile_lower[smem_y][smem_x])
        return;

    if (tx > 0 && tile_upper[smem_y][smem_x - 1]) {
        upper_lineptr[x] = true;
        *has_changed_global = true;
    }

    if (tx < HYSTERESIS_TILE_WIDTH - 1 && tile_upper[smem_y][smem_x + 1]) {
        upper_lineptr[x] = true;
        *has_changed_global = true;
    }

    if (ty > 0 && tile_upper[smem_y - 1][smem_x]) {
        upper_lineptr[x] = true;
        *has_changed_global = true;
    }

    if (ty < HYSTERESIS_TILE_WIDTH - 1 && tile_upper[smem_y + 1][smem_x]) {
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

