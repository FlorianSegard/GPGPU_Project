#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
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

__global__ void erode(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float* lineptr = (float*)((std::byte*)input.buffer + y * stride);
        float* lineptr_out = (float*)((std::byte*)output.buffer + y * stride);

        float min_val = lineptr[x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float* neighbor = (float*)((std::byte*)input.buffer + ny * stride);
                    min_val = fminf(min_val, neighbor[nx]);
                }
            }
        }
        lineptr_out[x] = min_val;
    }
}

__global__ void dilate(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float* lineptr = (float*)((std::byte*)input.buffer + y * stride);
        float* lineptr_out = (float*)((std::byte*)output.buffer + y * stride);

        float max_val = lineptr[x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float* neighbor = (float*)((std::byte*)input.buffer + ny * stride);
                    max_val = fmaxf(max_val, neighbor[nx]);
                }
            }
        }
        lineptr_out[x] = max_val;
    }
}


void erode_cu(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    erode<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height, stride);
}

void dilate_cu(ImageView<float> input, ImageView<float> output, int width, int height, std::ptrdiff_t stride)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dilate<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height, stride);
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