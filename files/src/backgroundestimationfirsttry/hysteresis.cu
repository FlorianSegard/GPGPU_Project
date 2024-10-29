#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Run with: nvcc hysteresis.cu -o hysteresis

// Simple LAB color structure
struct lab {
    float L;
    float a;
    float b;
};

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

// -----------------------------------------------------------

__device__ bool has_changed;

__global__ void hysteresis_reconstruction(const lab* input, bool* marker, bool* output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    lab* lineptr = (lab*)((std::byte*)input + y * stride);
    int current_idx = y * width + x;

    if (output[current_idx] || !marker[current_idx]) // already processed or too low
        return;

    // Check 8-connected neighbors
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_idx = ny * width + x;
                if (output[neighbor_idx]) {
                    output[current_idx] = true;
                    has_changed = true;
                    return;
                }
            }
        }
    }
}

int main() {
    const int width = 64;
    const int height = 64;
    const std::ptrdiff_t stride = width * sizeof(lab);

    // Allocate memory
    std::vector<lab> input(width * height);
    std::vector<bool> marker(width * height, false);
    std::vector<bool> output(width * height, false);

    // Initialize test data
    initializeTestData(input.data(), width, height, stride);

    // Set some markers for testing
    marker[width / 2 + height / 2 * width] = true; // Center pixel

    // Allocate device memory
    lab* d_input;
    bool* d_marker;
    bool* d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, height * stride));
    CHECK_CUDA_ERROR(cudaMalloc(&d_marker, width * height * sizeof(bool)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, width * height * sizeof(bool)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), height * stride, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_marker, marker.data(), width * height * sizeof(bool), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, output.data(), width * height * sizeof(bool), cudaMemcpyHostToDevice));

    // Set up grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Run hysteresis reconstruction
    do {
        has_changed = false;
        hysteresis_reconstruction<<<numBlocks, threadsPerBlock>>>(d_input, d_marker, d_output, width, height, stride);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&has_changed, has_changed, sizeof(bool), 0, cudaMemcpyDeviceToHost));
    } while (has_changed);

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, width * height * sizeof(bool), cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Hysteresis result (center 5x5 section):\n";
    for (int y = height / 2 - 2; y <= height / 2 + 2; ++y) {
        for (int x = width / 2 - 2; x <= width / 2 + 2; ++x) {
            std::cout << output[y * width + x] << " ";
        }
        std::cout << "\n";
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_marker);
    cudaFree(d_output);

    return 0;
}
