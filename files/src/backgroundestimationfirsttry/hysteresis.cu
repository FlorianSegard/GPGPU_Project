#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// Run with: nvcc hysteresis.cu -o hysteresis
// ./hysteresis <your-image.jpg>

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

// CUDA kernel for converting RGB image to LAB
__global__ void rgb_to_lab(const uchar3* input, lab* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    uchar3 pixel = input[idx];
    float r = pixel.x / 255.0f;
    float g = pixel.y / 255.0f;
    float b = pixel.z / 255.0f;

    printf("rgb2lab 1");

    float var_R = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    float var_G = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    float var_B = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    printf("rgb2lab 2");
    float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
    float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
    float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

    printf("rgb2lab 3");
    float L = (Y > 0.008856) ? 116.0 * pow(Y, 1.0 / 3.0) - 16.0 : 903.3 * Y;
    float A = 500.0 * (((X > 0.008856) ? pow(X, 1.0 / 3.0) : (7.787 * X + 16.0 / 116.0)) -
                      ((Y > 0.008856) ? pow(Y, 1.0 / 3.0) : (7.787 * Y + 16.0 / 116.0)));
    float B = 200.0 * (((Y > 0.008856) ? pow(Y, 1.0 / 3.0) : (7.787 * Y + 16.0 / 116.0)) -
                      ((Z > 0.008856) ? pow(Z, 1.0 / 3.0) : (7.787 * Z + 16.0 / 116.0)));

    printf("rgb2lab 4");
    output[idx] = {L, A, B};
}

// -----------------------------------------------------------

__device__ bool has_changed;

__global__ void hysteresis_reconstruction(const lab* input, bool* marker, bool* output, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    lab* lineptr;
    lineptr = (lab*)((std::byte*)input + y * stride);
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image.jpg>" << std::endl;
        return 1;
    }

    // Load the image
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << argv[1] << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read file: " << argv[1] << std::endl;
        return 1;
    }

    file.close();

    int width = 1280; // 480p
    int height = 720; // 480p
    std::ptrdiff_t stride = width * sizeof(lab);

    // Allocate memory on the host
    std::vector<lab> input(width * height);
    std::vector<uint8_t> marker(width * height, false);
    std::vector<uint8_t> output(width * height, false);

    // Allocate memory on the device
    lab* d_input;
    bool* d_marker;
    bool* d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, height * stride));
    CHECK_CUDA_ERROR(cudaMalloc(&d_marker, width * height * sizeof(bool)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, width * height * sizeof(bool)));

    // Copy data to the device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), height * stride, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_marker, marker.data(), width * height * sizeof(bool), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, output.data(), width * height * sizeof(bool), cudaMemcpyHostToDevice));

    // Set up grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Convert the image to LAB
    uchar3* rgb_data = reinterpret_cast<uchar3*>(buffer.data());
    rgb_to_lab<<<numBlocks, threadsPerBlock>>>(rgb_data, d_input, width, height);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Run hysteresis reconstruction
    bool host_has_changed;
    do {
        host_has_changed = false;
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(has_changed, &host_has_changed, sizeof(bool), 0, cudaMemcpyHostToDevice));
        hysteresis_reconstruction<<<numBlocks, threadsPerBlock>>>(d_input, d_marker, d_output, width, height, stride);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&host_has_changed, has_changed, sizeof(bool), 0, cudaMemcpyDeviceToHost));
    } while (host_has_changed);

    // Copy the result back to the host
    std::vector<uint8_t> host_output(width * height);
    CHECK_CUDA_ERROR(cudaMemcpy(host_output.data(), d_output, width * height * sizeof(bool), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_marker);
    cudaFree(d_output);

    return 0;
}
