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
    // Coordonnées du pixel traité par ce thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Taille du voisinage 3x3 (1 pixel de part et d'autre)
    const int kernel_radius = 1;

    // Définir un bloc de mémoire partagée pour stocker les pixels voisins
    __shared__ lab shared_block[32 + 2][32 + 2]; // Par exemple, un bloc de 32x32 + bords

    // Indices locaux dans le bloc partagé
    int local_x = threadIdx.x + kernel_radius;
    int local_y = threadIdx.y + kernel_radius;

    // Charger le pixel dans la mémoire partagée
    if (x < width && y < height) {
        lab* lineptr = (lab*)((std::byte*)input + y * stride);
        shared_block[local_y][local_x] = lineptr[x];
    }

    // Charger les pixels voisins nécessaires pour le calcul du voisinage
    if (threadIdx.x < kernel_radius) {
        // Charger le bord gauche
        if (x >= kernel_radius) {
            shared_block[local_y][local_x - kernel_radius] = input[(y * stride) / sizeof(lab) + x - kernel_radius];
        }
        // Charger le bord droit
        if (x + blockDim.x < width) {
            shared_block[local_y][local_x + blockDim.x] = input[(y * stride) / sizeof(lab) + x + blockDim.x];
        }
    }
    if (threadIdx.y < kernel_radius) {
        // Charger le bord supérieur
        if (y >= kernel_radius) {
            shared_block[local_y - kernel_radius][local_x] = input[((y - kernel_radius) * stride) / sizeof(lab) + x];
        }
        // Charger le bord inférieur
        if (y + blockDim.y < height) {
            shared_block[local_y + blockDim.y][local_x] = input[((y + blockDim.y) * stride) / sizeof(lab) + x];
        }
    }

    __syncthreads();

    // Calculer le minimum dans le voisinage 3x3
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

        // Stocker le résultat dans la mémoire globale
        lab* lineptr_out = (lab*)((std::byte*)output + y * stride);
        lineptr_out[x] = min_val;
    }
}

__global__ void dilate(lab* input, lab* output, int width, int height, std::ptrdiff_t stride) {
    // Coordonnées du pixel traité par ce thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int kernel_radius = 1;

    __shared__ lab shared_block[32 + 2][32 + 2];

    int local_x = threadIdx.x + kernel_radius;
    int local_y = threadIdx.y + kernel_radius;

    if (x < width && y < height) {
        lab* lineptr = (lab*)((std::byte*)input + y * stride);
        shared_block[local_y][local_x] = lineptr[x];
    }

    if (threadIdx.x < kernel_radius) {
        if (x >= kernel_radius) {
            shared_block[local_y][local_x - kernel_radius] = input[(y * stride) / sizeof(lab) + x - kernel_radius];
        }
        if (x + blockDim.x < width) {
            shared_block[local_y][local_x + blockDim.x] = input[(y * stride) / sizeof(lab) + x + blockDim.x];
        }
    }
    if (threadIdx.y < kernel_radius) {
        if (y >= kernel_radius) {
            shared_block[local_y - kernel_radius][local_x] = input[((y - kernel_radius) * stride) / sizeof(lab) + x];
        }
        if (y + blockDim.y < height) {
            shared_block[local_y + blockDim.y][local_x] = input[((y + blockDim.y) * stride) / sizeof(lab) + x];
        }
    }

    __syncthreads();

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