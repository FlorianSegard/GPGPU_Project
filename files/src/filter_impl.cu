#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "logic/labConverter.hpp"
#include "logic/backgroundestimation.hpp"
#include "logic/filter_erode_and_dilate.hpp"
#include "logic/hysteresis.hpp"
#include "logic/red_mask.hpp"
#include "filter_impl.h"

// Cuda error checking macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Separate kernel launch error checking function
inline void checkKernelLaunch() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel synchronization error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ============== CUDA FUNCTIONS ==============

__global__ void debug_bool_kernel(ImageView<bool> bf, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    bool bl = (bool*)((std::byte*)bf.buffer + y * bf.stride)[x];
    rgb8* rgb_value = (rgb8*)((std::byte*)rgb_buffer.buffer + y * rgb_buffer.stride);

    rgb_value[x].r = bl ? 255 : 0;//rgb_value[x].r / 2 + (bf ? 127 : 0);
    rgb_value[x].g = bl ? 255 : 0;//rgb_value[x].g / 2;
    rgb_value[x].b = bl ? 255 : 0;//rgb_value[x].b / 2;
}


__global__ void debug_float_kernel(ImageView<float> bf, ImageView<rgb8> rgb_buffer, int width, int height, std::ptrdiff_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* bl = (float*)((std::byte*)bf.buffer + y * bf.stride);
    rgb8* rgb_value = (rgb8*)((std::byte*)rgb_buffer.buffer + y * rgb_buffer.stride);

    rgb_value[x].r = round(fmaxf((int)(bl[x]), 255.0));
    rgb_value[x].g = round(fmaxf((int)(bl[x]), 255.0));
    rgb_value[x].b = round(fmaxf((int)(bl[x]), 255.0));
}


Image<lab> current_background;
Image<lab> candidate_background;
Image<int> current_time_pixels;
bool isInitialized = false;

void initializeGlobals(int width, int height) {
    if (!isInitialized) {
        current_background = Image<lab>(width, height, true);
        candidate_background = Image<lab>(width, height, true);
        current_time_pixels = Image<int>(width, height, true);
        isInitialized = true;
    }
}

// TODO: what to do when background_ref / candidate_background null?
// TODO: is it possible to reuse buffers instead of always creating new ones?
// Check error after each initialization
extern "C" {
void filter_impl_cu(uint8_t* pixels_buffer, int width, int height, int plane_stride)
{
    // Init device and global variables
    Parameters params;
    params.device = GPU;
    initializeGlobals(width, height);



    // GPU properties for kernel calls
    cudaError_t error;
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Clone pixels_buffer inside new allocated rgb_buffer
    Image<rgb8> rgb_image(width, height, true);
    error = cudaMemcpy2D(rgb_image.buffer, rgb_image.stride, pixels_buffer, plane_stride,
                         width * sizeof(rgb8), height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(error);



    // Allocate lab converted image buffer
    labConv_init(&params);
    Image<lab> lab_image(width, height, true);

    // Convert RGB to LAB -> result stored inside lab_buffer
    labConv_process_frame(rgb_image, lab_image);
    cudaDeviceSynchronize();
    checkKernelLaunch();
    std::cout << "labConv call succeeded" << std::endl;



    // Update background and get residual image
    background_init(&params);
    Image<float> residual_image(width, height, true);

    background_process_frame(lab_image, current_background, candidate_background, current_time_pixels, residual_image);
	cudaDeviceSynchronize();
    checkKernelLaunch();
    std::cout << "background call succeeded" << std::endl;



    // Alloc and perform eroding operation
    filter_init(&params);
    Image<float> erode_image(width, height, true);

    erode_process_frame(
            residual_image, erode_image,
         width, height, plane_stride
    );
    cudaDeviceSynchronize();
    checkKernelLaunch();
    std::cout << "erode call succeeded" << std::endl;



    // Alloc and perform eroding operation
    Image<float> dilate_image(width, height, true);

    dilate_process_frame(
            erode_image, dilate_image,
            width, height, plane_stride
    );
    cudaDeviceSynchronize();
    checkKernelLaunch();
    std::cout << "dilate call succeeded" << std::endl;

    debug_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(dilate_image, rgb_image, width, height, plane_stride);
    /*

    // Alloc and perform hysteresis operation
    hysteresis_init(&params);
    Image<bool> hysteresis_image(width, height, true);

    //TODO: retrieve threshold values
    hysteresis_process_frame(
            dilate_image, hysteresis_image,
            width, height, 3, 30
    );
    cudaDeviceSynchronize();
    checkKernelLaunch();
    std::cout << "hysteresis call succeeded" << std::endl;


    // Alloc and red mask operation
    mask_process_frame(hysteresis_image, rgb_image, width, height, plane_stride);
    cudaDeviceSynchronize();
    checkKernelLaunch();
    std::cout << "red mask call succeeded" << std::endl;

    */



    // // Copy result back to pixels_buffer
    error = cudaMemcpy2D(pixels_buffer, plane_stride, rgb_image.buffer, rgb_image.stride,
                         width * sizeof(rgb8), height, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(error);
    std::cout << "copy back to pixels_buffer" << std::endl;

    // // Clean up temporary buffers
    // cudaFree(rgb_buffer);
    // cudaFree(lab_buffer);
    // cudaFree(residual_buffer);
    // cudaFree(eroded_buffer);
    // cudaFree(dilated_buffer);
}
}