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

#define HYSTERESIS_TILE_WIDTH 34 // block size de 32 x 32 et on rajoute 2 pixels de padding
#define LOWER_THRESHOLD 4.0
#define UPPER_THRESHOLD 30.0

// -----------------------------------------------------------

__global__ void hysteresis_thresholding(ImageView<float> input, ImageView<bool> output, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float *input_lineptr = (float *)((std::byte*)input.buffer + y * input.stride);
    float in_val = input_lineptr[x];

    // Applique le seuil et on stocke le résultat dans la sortie
    bool *output_lineptr = (bool *)((std::byte*)output.buffer + y * output.stride);
    output_lineptr[x] = in_val > threshold;
}


__global__ void hysteresis_kernel(ImageView<bool> upper, ImageView<bool> lower, int width, int height, bool *has_changed_global)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    bool has_changed = true;

    while (has_changed)
    {
        has_changed = false;
        __syncthreads();

        bool *upper_lineptr = (bool *)((std::byte*)upper.buffer + y * upper.stride);
        bool *lower_lineptr = (bool *)((std::byte*)lower.buffer + y * lower.stride);

        // Si le pixel est déjà marqué dans l'image supérieure, on passe au suivant
        if (upper_lineptr[x])
            break;

        // Si le pixel n'est pas marqué dans l'image inférieure, on passe au suivant
        if (!lower_lineptr[x])
            break;

        // on vérifie les pixels voisins pour propager le marquage
        if ((x > 0 && upper_lineptr[x - 1]) ||
            (x < width - 1 && upper_lineptr[x + 1]) ||
            (y > 0 && ((bool *)((std::byte*)upper.buffer + (y - 1) * upper_pitch))[x]) ||
            (y < height - 1 && ((bool *)((std::byte*)upper.buffer + (y + 1) * upper_pitch))[x]))
        {
            upper_lineptr[x] = true;
            has_changed = true;
            *has_changed_global = true;
            break;
        }

        __syncthreads();
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

    bool h_has_changed = true;

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

    cudaFree(d_has_changed);
}
