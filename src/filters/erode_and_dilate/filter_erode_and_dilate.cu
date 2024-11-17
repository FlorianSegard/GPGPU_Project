#include <iostream>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX and FLT_MIN
#include "filter_erode_and_dilate.hpp"

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

// Erosion kernel using shared memory
__global__ void erode_shared(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    extern __shared__ float sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    // Compute shared memory dimensions
    int smem_width = blockDim.x + 2 * opening_size;
    int smem_height = blockDim.y + 2 * opening_size;

    // Compute global coordinates
    int x_global = bx + tx - opening_size;
    int y_global = by + ty - opening_size;

    // Compute shared memory coordinates
    int x_shared = tx + opening_size;
    int y_shared = ty + opening_size;

    // Load data into shared memory with boundary checks
    float value = FLT_MAX;
    if (x_global >= 0 && x_global < width && y_global >= 0 && y_global < height) {
        float* lineptr = (float*)((std::byte*)input.buffer + y_global * input.stride);
        value = lineptr[x_global];
    }
    sdata[y_shared * smem_width + x_shared] = value;

    // Load halo regions (top, bottom, left, right)
    // Load top halo
    if (ty < opening_size) {
        for (int i = -opening_size; i < blockDim.x + opening_size; ++i) {
            int x_halo = bx + i;
            int y_halo = by + ty - opening_size;
            int x_s = tx + i + opening_size;
            int y_s = ty;

            float halo_value = FLT_MAX;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load bottom halo
    if (ty >= blockDim.y - opening_size) {
        for (int i = -opening_size; i < blockDim.x + opening_size; ++i) {
            int x_halo = bx + i;
            int y_halo = by + ty + opening_size;
            int x_s = tx + i + opening_size;
            int y_s = ty + 2 * opening_size;

            float halo_value = FLT_MAX;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load left halo
    if (tx < opening_size) {
        for (int i = -opening_size; i < blockDim.y + opening_size; ++i) {
            int x_halo = bx + tx - opening_size;
            int y_halo = by + i;
            int x_s = tx;
            int y_s = ty + i + opening_size;

            float halo_value = FLT_MAX;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load right halo
    if (tx >= blockDim.x - opening_size) {
        for (int i = -opening_size; i < blockDim.y + opening_size; ++i) {
            int x_halo = bx + tx + opening_size;
            int y_halo = by + i;
            int x_s = tx + 2 * opening_size;
            int y_s = ty + i + opening_size;

            float halo_value = FLT_MAX;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load corner halos
    // Top-left
    if (tx < opening_size && ty < opening_size) {
        int x_halo = bx + tx - opening_size;
        int y_halo = by + ty - opening_size;
        int x_s = tx;
        int y_s = ty;

        float halo_value = FLT_MAX;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    // Top-right
    if (tx >= blockDim.x - opening_size && ty < opening_size) {
        int x_halo = bx + tx + opening_size;
        int y_halo = by + ty - opening_size;
        int x_s = tx + 2 * opening_size;
        int y_s = ty;

        float halo_value = FLT_MAX;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    // Bottom-left
    if (tx < opening_size && ty >= blockDim.y - opening_size) {
        int x_halo = bx + tx - opening_size;
        int y_halo = by + ty + opening_size;
        int x_s = tx;
        int y_s = ty + 2 * opening_size;

        float halo_value = FLT_MAX;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    // Bottom-right
    if (tx >= blockDim.x - opening_size && ty >= blockDim.y - opening_size) {
        int x_halo = bx + tx + opening_size;
        int y_halo = by + ty + opening_size;
        int x_s = tx + 2 * opening_size;
        int y_s = ty + 2 * opening_size;

        float halo_value = FLT_MAX;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    __syncthreads();

    // Perform erosion using shared memory
    int x_out = bx + tx;
    int y_out = by + ty;

    if (x_out < width && y_out < height) {
        float min_val = FLT_MAX;
        for (int dy = -opening_size; dy <= opening_size; ++dy) {
            for (int dx = -opening_size; dx <= opening_size; ++dx) {
                int x_s = x_shared + dx;
                int y_s = y_shared + dy;
                float val = sdata[y_s * smem_width + x_s];
                min_val = fminf(min_val, val);
            }
        }
        float* lineptr_out = (float*)((std::byte*)output.buffer + y_out * output.stride);
        lineptr_out[x_out] = min_val;
    }
}

// Dilation kernel using shared memory
__global__ void dilate_shared(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    extern __shared__ float sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    // Compute shared memory dimensions
    int smem_width = blockDim.x + 2 * opening_size;
    int smem_height = blockDim.y + 2 * opening_size;

    // Compute global coordinates
    int x_global = bx + tx - opening_size;
    int y_global = by + ty - opening_size;

    // Compute shared memory coordinates
    int x_shared = tx + opening_size;
    int y_shared = ty + opening_size;

    // Load data into shared memory with boundary checks
    float value = FLT_MIN;
    if (x_global >= 0 && x_global < width && y_global >= 0 && y_global < height) {
        float* lineptr = (float*)((std::byte*)input.buffer + y_global * input.stride);
        value = lineptr[x_global];
    }
    sdata[y_shared * smem_width + x_shared] = value;

    // Load halo regions (similar to erode_shared kernel)
    // Load top, bottom, left, right, and corner halos
    // Replace FLT_MAX with FLT_MIN for dilation

    // Load top halo
    if (ty < opening_size) {
        for (int i = -opening_size; i < blockDim.x + opening_size; ++i) {
            int x_halo = bx + i;
            int y_halo = by + ty - opening_size;
            int x_s = tx + i + opening_size;
            int y_s = ty;

            float halo_value = FLT_MIN;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load bottom halo
    if (ty >= blockDim.y - opening_size) {
        for (int i = -opening_size; i < blockDim.x + opening_size; ++i) {
            int x_halo = bx + i;
            int y_halo = by + ty + opening_size;
            int x_s = tx + i + opening_size;
            int y_s = ty + 2 * opening_size;

            float halo_value = FLT_MIN;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load left halo
    if (tx < opening_size) {
        for (int i = -opening_size; i < blockDim.y + opening_size; ++i) {
            int x_halo = bx + tx - opening_size;
            int y_halo = by + i;
            int x_s = tx;
            int y_s = ty + i + opening_size;

            float halo_value = FLT_MIN;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load right halo
    if (tx >= blockDim.x - opening_size) {
        for (int i = -opening_size; i < blockDim.y + opening_size; ++i) {
            int x_halo = bx + tx + opening_size;
            int y_halo = by + i;
            int x_s = tx + 2 * opening_size;
            int y_s = ty + i + opening_size;

            float halo_value = FLT_MIN;
            if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
                float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
                halo_value = lineptr[x_halo];
            }
            sdata[y_s * smem_width + x_s] = halo_value;
        }
    }

    // Load corner halos (top-left, top-right, bottom-left, bottom-right)
    // Similar to erode_shared kernel, replace FLT_MAX with FLT_MIN

    // Top-left
    if (tx < opening_size && ty < opening_size) {
        int x_halo = bx + tx - opening_size;
        int y_halo = by + ty - opening_size;
        int x_s = tx;
        int y_s = ty;

        float halo_value = FLT_MIN;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    // Top-right
    if (tx >= blockDim.x - opening_size && ty < opening_size) {
        int x_halo = bx + tx + opening_size;
        int y_halo = by + ty - opening_size;
        int x_s = tx + 2 * opening_size;
        int y_s = ty;

        float halo_value = FLT_MIN;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    // Bottom-left
    if (tx < opening_size && ty >= blockDim.y - opening_size) {
        int x_halo = bx + tx - opening_size;
        int y_halo = by + ty + opening_size;
        int x_s = tx;
        int y_s = ty + 2 * opening_size;

        float halo_value = FLT_MIN;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    // Bottom-right
    if (tx >= blockDim.x - opening_size && ty >= blockDim.y - opening_size) {
        int x_halo = bx + tx + opening_size;
        int y_halo = by + ty + opening_size;
        int x_s = tx + 2 * opening_size;
        int y_s = ty + 2 * opening_size;

        float halo_value = FLT_MIN;
        if (x_halo >= 0 && x_halo < width && y_halo >= 0 && y_halo < height) {
            float* lineptr = (float*)((std::byte*)input.buffer + y_halo * input.stride);
            halo_value = lineptr[x_halo];
        }
        sdata[y_s * smem_width + x_s] = halo_value;
    }

    __syncthreads();

    // Perform dilation using shared memory
    int x_out = bx + tx;
    int y_out = by + ty;

    if (x_out < width && y_out < height) {
        float max_val = FLT_MIN;
        for (int dy = -opening_size; dy <= opening_size; ++dy) {
            for (int dx = -opening_size; dx <= opening_size; ++dx) {
                int x_s = x_shared + dx;
                int y_s = y_shared + dy;
                float val = sdata[y_s * smem_width + x_s];
                max_val = fmaxf(max_val, val);
            }
        }
        float* lineptr_out = (float*)((std::byte*)output.buffer + y_out * output.stride);
        lineptr_out[x_out] = max_val;
    }
}

void erode_cu(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int smem_size = (threadsPerBlock.x + 2 * opening_size) * (threadsPerBlock.y + 2 * opening_size) * sizeof(float);
    erode_shared<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height, opening_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void dilate_cu(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int smem_size = (threadsPerBlock.x + 2 * opening_size) * (threadsPerBlock.y + 2 * opening_size) * sizeof(float);
    dilate_shared<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, output, width, height, opening_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
