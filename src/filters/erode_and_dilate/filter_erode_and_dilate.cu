#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cfloat> // For FLT_MAX and FLT_MIN
#include "filter_erode_and_dilate.hpp"

// Run with: nvcc filter_erode_and_dilate.cu -o filter_erode_and_dilate

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

__global__ void erode_shared(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    extern __shared__ float smem[];

    int smem_width = blockDim.x + 2 * opening_size;
    int smem_height = blockDim.y + 2 * opening_size;
    int smem_size = smem_width * smem_height;

    int num_threads = blockDim.x * blockDim.y;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // Compute starting global x and y indices for shared memory
    int base_x = blockIdx.x * blockDim.x - opening_size;
    int base_y = blockIdx.y * blockDim.y - opening_size;

    // Load shared memory
    for (int i = thread_id; i < smem_size; i += num_threads) {
        int smem_x = i % smem_width;
        int smem_y = i / smem_width;

        int global_x = base_x + smem_x;
        int global_y = base_y + smem_y;

        float value;
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + global_y * input.stride);
            value = lineptr[global_x];
        } else {
            value = FLT_MAX; // For erosion
        }
        smem[smem_y * smem_width + smem_x] = value;
    }

    __syncthreads();

    // Now perform erosion using shared memory
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int smem_x = threadIdx.x + opening_size;
        int smem_y = threadIdx.y + opening_size;

        float min_val = FLT_MAX;
        for (int dy = -opening_size; dy <= opening_size; ++dy) {
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

__global__ void dilate_shared(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    extern __shared__ float smem[];

    int smem_width = blockDim.x + 2 * opening_size;
    int smem_height = blockDim.y + 2 * opening_size;
    int smem_size = smem_width * smem_height;

    int num_threads = blockDim.x * blockDim.y;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // Compute starting global x and y indices for shared memory
    int base_x = blockIdx.x * blockDim.x - opening_size;
    int base_y = blockIdx.y * blockDim.y - opening_size;

    // Load shared memory
    for (int i = thread_id; i < smem_size; i += num_threads) {
        int smem_x = i % smem_width;
        int smem_y = i / smem_width;

        int global_x = base_x + smem_x;
        int global_y = base_y + smem_y;

        float value;
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + global_y * input.stride);
            value = lineptr[global_x];
        } else {
            value = FLT_MIN; // For dilation
        }
        smem[smem_y * smem_width + smem_x] = value;
    }

    __syncthreads();

    // Now perform dilation using shared memory
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int smem_x = threadIdx.x + opening_size;
        int smem_y = threadIdx.y + opening_size;

        float max_val = FLT_MIN;
        for (int dy = -opening_size; dy <= opening_size; ++dy) {
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
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int smem_size = (threadsPerBlock.x + 2 * opening_size) * (threadsPerBlock.y + 2 * opening_size) * sizeof(float);
    erode_shared<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height, opening_size);
}

void dilate_cu(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int smem_size = (threadsPerBlock.x + 2 * opening_size) * (threadsPerBlock.y + 2 * opening_size) * sizeof(float);
    dilate_shared<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height, opening_size);
}

/*
// Helper function to initialize test data
void initializeTestData(lab* data, int width, int height, std::ptrdiff_t stride) {
    for (int y = 0; y < height; y++) {
        lab* row = (lab*)((std::byte*)data + y * stride);
        for (int x = 0; x < width; x++) {
            // Create a simple pattern: higher values in the center
            float centerX = width / 2.0f;
            float centerY = height / 2.0f;
            float distance = sqrtf(powf(x - centerX, 2) + powf(y - centerY, 2));
            row[x].L = 100.0f * (1.0f - distance / sqrtf(centerX * centerX + centerY * centerY));
            row[x].a = 50.0f;
            row[x].b = 50.0f;
        }
    }
}

// Helper function to print a small section of the image
void printImageSection(lab* data, int width, int height, std::ptrdiff_t stride, int startX, int startY, int sectionSize) {
    printf("\nImage section (L channel only):\n");
    for (int y = startY; y < std::min(startY + sectionSize, height); y++) {
        lab* row = (lab*)((std::byte*)data + y * stride);
        for (int x = startX; x < std::min(startX + sectionSize, width); x++) {
            printf("%.1f ", row[x].L);
        }
        printf("\n");
    }
}


int main() {
    // Define dimensions once at the start
    const int width = 1024;
    const int height = 1024;
    const std::ptrdiff_t stride = width * sizeof(lab);
    const int NUM_ITERATIONS = 100;

    // Allocate host memory
    lab* h_input = new lab[width * height];
    lab* h_output = new lab[width * height];

    // Initialize test data
    initializeTestData(h_input, width, height, stride);

    // Allocate device memory
    lab *d_input, *d_output, *d_temp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, height * stride));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, height * stride));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, height * stride));

    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, height * stride, cudaMemcpyHostToDevice));

    // Set up grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Print a section of the original image:");
    printImageSection(h_input, width, height, stride, 0, 0, 5);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Warmup phase
    for(int i = 0; i < 1000; i++) {
        erode<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, stride);
    }
    cudaDeviceSynchronize();

    // Time erosion
    cudaEventRecord(start);
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        erode<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, stride);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nGPU Erosion time (averaged over %d iterations): %.2f microseconds", 
           NUM_ITERATIONS, (milliseconds * 1000.0f) / NUM_ITERATIONS);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, height * stride, cudaMemcpyDeviceToHost));
    printf("\nAfter erosion:");
    printImageSection(h_output, width, height, stride, 0, 0, 5);

    // Time dilation
    cudaEventRecord(start);
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        dilate<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, stride);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nGPU Dilation time (averaged over %d iterations): %.2f microseconds", 
           NUM_ITERATIONS, (milliseconds * 1000.0f) / NUM_ITERATIONS);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, height * stride, cudaMemcpyDeviceToHost));
    printf("\nAfter dilation:");
    printImageSection(h_output, width, height, stride, 0, 0, 5);

    // Time closing operation
    cudaEventRecord(start);
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        erode<<<numBlocks, threadsPerBlock>>>(d_input, d_temp, width, height, stride);
        dilate<<<numBlocks, threadsPerBlock>>>(d_temp, d_output, width, height, stride);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nGPU Closing time (averaged over %d iterations): %.2f microseconds", 
           NUM_ITERATIONS, (milliseconds * 1000.0f) / NUM_ITERATIONS);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, height * stride, cudaMemcpyDeviceToHost));
    printf("\nAfter closing (erosion + dilation):");
    printImageSection(h_output, width, height, stride, 0, 0, 5);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_temp));

    return 0;
} **/