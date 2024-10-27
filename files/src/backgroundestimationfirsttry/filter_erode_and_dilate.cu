#include <iostream>
#include <cuda_runtime.h>

// Run with: nvcc filter_erode_and_dilate.cu -o filter_erode_and_dilate

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

__global__ void erode(lab* input, lab* output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        lab* lineptr = (lab*)((std::byte*)input + y * stride);
        lab* lineptr_out = (lab*)((std::byte*)output + y * stride);

        lab min_val = lineptr[x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    lab* neighbor = (lab*)((std::byte*)input + ny * stride);
                    min_val.L = fminf(min_val.L, neighbor[nx].L);
                    min_val.a = fminf(min_val.a, neighbor[nx].a);
                    min_val.b = fminf(min_val.b, neighbor[nx].b);
                }
            }
        }
        lineptr_out[x] = min_val;
    }
}

__global__ void dilate(lab* input, lab* output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        lab* lineptr = (lab*)((std::byte*)input + y * stride);
        lab* lineptr_out = (lab*)((std::byte*)output + y * stride);

        lab max_val = lineptr[x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    lab* neighbor = (lab*)((std::byte*)input + ny * stride);
                    max_val.L = fmaxf(max_val.L, neighbor[nx].L);
                    max_val.a = fmaxf(max_val.a, neighbor[nx].a);
                    max_val.b = fmaxf(max_val.b, neighbor[nx].b);
                }
            }
        }
        lineptr_out[x] = max_val;
    }
}

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
    const int width = 64;
    const int height = 64;
    const std::ptrdiff_t stride = width * sizeof(lab);  // Simple stride calculation

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
    dim3 threadsPerBlock(16, 16); // May need to be changed ... to be tested
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Print a section of the original image:");
    printImageSection(h_input, width, height, stride, 0, 0, 5);

    // Erosion
    erode<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, stride);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, height * stride, cudaMemcpyDeviceToHost));
    printf("\nAfter erosion:");
    printImageSection(h_output, width, height, stride, 0, 0, 5);

    // Test dilation
    dilate<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, stride);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, height * stride, cudaMemcpyDeviceToHost));
    printf("\nAfter dilation:");
    printImageSection(h_output, width, height, stride, 0, 0, 5);

    // Test erosion followed by dilation (closing operation)
    erode<<<numBlocks, threadsPerBlock>>>(d_input, d_temp, width, height, stride);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    dilate<<<numBlocks, threadsPerBlock>>>(d_temp, d_output, width, height, stride);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, height * stride, cudaMemcpyDeviceToHost));
    printf("\nAfter closing (erosion + dilation):");
    printImageSection(h_output, width, height, stride, 0, 0, 5);

    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_temp));

    return 0;
}