#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
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

    const int kernel_radius = 1;
    const int BLOCK_DIM = 16;

    // Shared memory block avec padding pour les halos
    __shared__ lab shared_block[18][18]; // (BLOCK_DIM + 2) × (BLOCK_DIM + 2)

    // Local indices in shared memory
    int local_x = threadIdx.x + kernel_radius;
    int local_y = threadIdx.y + kernel_radius;

    // Load center pixels
    if (x < width && y < height) {
        lab* lineptr = (lab*)((std::byte*)input + y * stride);
        shared_block[local_y][local_x] = lineptr[x];
    }

    // Load halo regions
    // Left and right halos
    if (threadIdx.x < kernel_radius) {
        // Left
        if (x > 0) {
            lab* lineptr = (lab*)((std::byte*)input + y * stride);
            shared_block[local_y][threadIdx.x] = lineptr[x - 1];
        } else {
            shared_block[local_y][threadIdx.x] = shared_block[local_y][local_x];
        }
        
        // Right
        if (x + BLOCK_DIM < width) {
            lab* lineptr = (lab*)((std::byte*)input + y * stride);
            shared_block[local_y][threadIdx.x + BLOCK_DIM + kernel_radius] = lineptr[x + BLOCK_DIM];
        } else {
            shared_block[local_y][threadIdx.x + BLOCK_DIM + kernel_radius] = shared_block[local_y][local_x];
        }
    }

    // Top and bottom halos
    if (threadIdx.y < kernel_radius) {
        // Top
        if (y > 0) {
            lab* lineptr = (lab*)((std::byte*)input + (y - 1) * stride);
            shared_block[threadIdx.y][local_x] = lineptr[x];
        } else {
            shared_block[threadIdx.y][local_x] = shared_block[local_y][local_x];
        }
        
        // Bottom
        if (y + BLOCK_DIM < height) {
            lab* lineptr = (lab*)((std::byte*)input + (y + BLOCK_DIM) * stride);
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][local_x] = lineptr[x];
        } else {
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][local_x] = shared_block[local_y][local_x];
        }
    }

    // Load corner halos
    if (threadIdx.x < kernel_radius && threadIdx.y < kernel_radius) {
        // Top-left
        if (x > 0 && y > 0) {
            lab* lineptr = (lab*)((std::byte*)input + (y - 1) * stride);
            shared_block[threadIdx.y][threadIdx.x] = lineptr[x - 1];
        } else {
            shared_block[threadIdx.y][threadIdx.x] = shared_block[local_y][local_x];
        }

        // Top-right
        if (x + BLOCK_DIM < width && y > 0) {
            lab* lineptr = (lab*)((std::byte*)input + (y - 1) * stride);
            shared_block[threadIdx.y][threadIdx.x + BLOCK_DIM + kernel_radius] = lineptr[x + BLOCK_DIM];
        } else {
            shared_block[threadIdx.y][threadIdx.x + BLOCK_DIM + kernel_radius] = shared_block[local_y][local_x];
        }

        // Bottom-left
        if (x > 0 && y + BLOCK_DIM < height) {
            lab* lineptr = (lab*)((std::byte*)input + (y + BLOCK_DIM) * stride);
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x] = lineptr[x - 1];
        } else {
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x] = shared_block[local_y][local_x];
        }

        // Bottom-right
        if (x + BLOCK_DIM < width && y + BLOCK_DIM < height) {
            lab* lineptr = (lab*)((std::byte*)input + (y + BLOCK_DIM) * stride);
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x + BLOCK_DIM + kernel_radius] = lineptr[x + BLOCK_DIM];
        } else {
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x + BLOCK_DIM + kernel_radius] = shared_block[local_y][local_x];
        }
    }

    __syncthreads();

    // Calcul érosion
    if (x < width && y < height) {
        lab min_val = shared_block[local_y][local_x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                lab neighbor = shared_block[local_y + dy][local_x + dx];
                min_val.L = fminf(min_val.L, neighbor.L);
                min_val.a = fminf(min_val.a, neighbor.a);
                min_val.b = fminf(min_val.b, neighbor.b);
            }
        }

        lab* lineptr_out = (lab*)((std::byte*)output + y * stride);
        lineptr_out[x] = min_val;
    }
}

__global__ void dilate(lab* input, lab* output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int kernel_radius = 1;
    const int BLOCK_DIM = 16;

    // Shared memory block avec padding pour les halos
    __shared__ lab shared_block[18][18]; // (BLOCK_DIM + 2) × (BLOCK_DIM + 2)

    // Local indices in shared memory
    int local_x = threadIdx.x + kernel_radius;
    int local_y = threadIdx.y + kernel_radius;

    // Load center pixels
    if (x < width && y < height) {
        lab* lineptr = (lab*)((std::byte*)input + y * stride);
        shared_block[local_y][local_x] = lineptr[x];
    }

    // Load halo regions
    // Left and right halos
    if (threadIdx.x < kernel_radius) {
        // Left
        if (x > 0) {
            lab* lineptr = (lab*)((std::byte*)input + y * stride);
            shared_block[local_y][threadIdx.x] = lineptr[x - 1];
        } else {
            shared_block[local_y][threadIdx.x] = shared_block[local_y][local_x];
        }
        
        // Right
        if (x + BLOCK_DIM < width) {
            lab* lineptr = (lab*)((std::byte*)input + y * stride);
            shared_block[local_y][threadIdx.x + BLOCK_DIM + kernel_radius] = lineptr[x + BLOCK_DIM];
        } else {
            shared_block[local_y][threadIdx.x + BLOCK_DIM + kernel_radius] = shared_block[local_y][local_x];
        }
    }

    // Top and bottom halos
    if (threadIdx.y < kernel_radius) {
        // Top
        if (y > 0) {
            lab* lineptr = (lab*)((std::byte*)input + (y - 1) * stride);
            shared_block[threadIdx.y][local_x] = lineptr[x];
        } else {
            shared_block[threadIdx.y][local_x] = shared_block[local_y][local_x];
        }
        
        // Bottom
        if (y + BLOCK_DIM < height) {
            lab* lineptr = (lab*)((std::byte*)input + (y + BLOCK_DIM) * stride);
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][local_x] = lineptr[x];
        } else {
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][local_x] = shared_block[local_y][local_x];
        }
    }

    // Load corner halos
    if (threadIdx.x < kernel_radius && threadIdx.y < kernel_radius) {
        // Top-left
        if (x > 0 && y > 0) {
            lab* lineptr = (lab*)((std::byte*)input + (y - 1) * stride);
            shared_block[threadIdx.y][threadIdx.x] = lineptr[x - 1];
        } else {
            shared_block[threadIdx.y][threadIdx.x] = shared_block[local_y][local_x];
        }

        // Top-right
        if (x + BLOCK_DIM < width && y > 0) {
            lab* lineptr = (lab*)((std::byte*)input + (y - 1) * stride);
            shared_block[threadIdx.y][threadIdx.x + BLOCK_DIM + kernel_radius] = lineptr[x + BLOCK_DIM];
        } else {
            shared_block[threadIdx.y][threadIdx.x + BLOCK_DIM + kernel_radius] = shared_block[local_y][local_x];
        }

        // Bottom-left
        if (x > 0 && y + BLOCK_DIM < height) {
            lab* lineptr = (lab*)((std::byte*)input + (y + BLOCK_DIM) * stride);
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x] = lineptr[x - 1];
        } else {
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x] = shared_block[local_y][local_x];
        }

        // Bottom-right
        if (x + BLOCK_DIM < width && y + BLOCK_DIM < height) {
            lab* lineptr = (lab*)((std::byte*)input + (y + BLOCK_DIM) * stride);
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x + BLOCK_DIM + kernel_radius] = lineptr[x + BLOCK_DIM];
        } else {
            shared_block[threadIdx.y + BLOCK_DIM + kernel_radius][threadIdx.x + BLOCK_DIM + kernel_radius] = shared_block[local_y][local_x];
        }
    }

    __syncthreads();

    // Compute dilation
    if (x < width && y < height) {
        lab max_val = shared_block[local_y][local_x];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                lab neighbor = shared_block[local_y + dy][local_x + dx];
                max_val.L = fmaxf(max_val.L, neighbor.L);
                max_val.a = fmaxf(max_val.a, neighbor.a);
                max_val.b = fmaxf(max_val.b, neighbor.b);
            }
        }

        lab* lineptr_out = (lab*)((std::byte*)output + y * stride);
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
    const int width = 1024;
    const int height = 1024;
    const std::ptrdiff_t stride = width * sizeof(lab);  // Simple stride calculation
    const int NUM_ITERATIONS = 100;  // Run multiple iterations

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

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Add GPU warmup phase
    for(int i = 0; i < 1000; i++) {  // More warmup iterations
        erode<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, stride);
    }
    cudaDeviceSynchronize();

    // Reset GPU before timing
    cudaDeviceReset();
    
    // Increase problem size
    const int width = 1024;  // Larger image
    const int height = 1024;
    
    // Add CPU-GPU sync before timing
    cudaDeviceSynchronize();
    
    // Time erosion with proper synchronization
    cudaEventRecord(start);
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        erode<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, stride);
        cudaDeviceSynchronize();  // Force synchronization each iteration
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
}