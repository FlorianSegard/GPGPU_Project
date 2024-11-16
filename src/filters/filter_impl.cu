#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "../logic/lab_converter/lab_converter.hpp"
#include "../logic/background/background_estimation.hpp"
#include "erode_and_dilate/filter_erode_and_dilate.hpp"
#include "../logic/hysteresis/hysteresis.hpp"
#include "../logic/red_mask/red_mask.hpp"
#include "filter_impl.hpp"

// Cuda error checking macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Separate kernel launch error checking function
inline void checkKernelLaunch(bool is_gpu) {
    if (!is_gpu)
        return;

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

// ============== CUDA FUNCTIONS FOR DEBUG ==============

// GPU properties for cuda debug purpose kernel calls
//cudaError_t error;
//dim3 threadsPerBlock(32, 32);
//dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

__global__ void debug_bool_kernel(ImageView<bool> bf, ImageView<rgb8> rgb_buffer, int width, int height) {
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


__global__ void debug_float_kernel(ImageView<float> bf, ImageView<rgb8> rgb_buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* bl = (float*)((std::byte*)bf.buffer + y * bf.stride);
    rgb8* rgb_value = (rgb8*)((std::byte*)rgb_buffer.buffer + y * rgb_buffer.stride);

    rgb_value[x].r = (uint8_t) round(fminf((bl[x]), 255.0));
    rgb_value[x].g = (uint8_t) round(fminf((bl[x]), 255.0));
    rgb_value[x].b = (uint8_t) round(fminf((bl[x]), 255.0));
}

// ============== MAIN IMAGE PROCESSING ==============

Image<lab> current_background;
Image<lab> candidate_background;
Image<int> current_time_pixels;
bool isInitialized = false;

void initializeGlobals(int width, int height, ImageView<lab> lab_image, bool is_gpu) {
    current_background = Image<lab>(width, height, is_gpu);
    candidate_background = Image<lab>(width, height, is_gpu);
    current_time_pixels = Image<int>(width, height, is_gpu);
    isInitialized = true;

    if (is_gpu) {
        cudaError_t error;
        error = cudaMemcpy2D(current_background.buffer, current_background.stride, lab_image.buffer, lab_image.stride,
                             width * sizeof(lab), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(error);
        error = cudaMemcpy2D(candidate_background.buffer, candidate_background.stride, lab_image.buffer,
                             lab_image.stride,
                             width * sizeof(lab), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(error);
        std::cout << "Running on GPU" << std::endl;
    }
    else {
        for (int y = 0; y < lab_image.height; ++y)
            memcpy((char*)current_background.buffer + y * current_background.stride,
                   (char*)lab_image.buffer + y * lab_image.stride,
                   lab_image.width * sizeof(lab));

        for (int y = 0; y < lab_image.height; ++y)
            memcpy((char*)candidate_background.buffer + y * candidate_background.stride,
                   (char*)lab_image.buffer + y * lab_image.stride,
                   lab_image.width * sizeof(lab));

        std::cout << "Running on CPU" << std::endl;
    }
}


extern "C" {
    void filter_impl(uint8_t* pixels_buffer, int width, int height, int plane_stride, e_device_t device,
                        const char* bg_uri, int opening_size, int th_low, int th_high, int bg_sampling_rate, int bg_number_frame)
    {
        // Init device and device variables
        Parameters params;
        params.device = device;
        bool is_gpu = device == GPU;

        cudaError_t error;
        lab_conv_init(&params);
        background_init(&params);
        filter_init(&params);
        hysteresis_init(&params);
        mask_init(&params);


        // Clone pixels_buffer inside new allocated rgb_buffer
        Image<rgb8> rgb_image(width, height, is_gpu);
        if (is_gpu) {
            error = cudaMemcpy2D(rgb_image.buffer, rgb_image.stride, pixels_buffer, plane_stride,
                                 width * sizeof(rgb8), height, cudaMemcpyDefault);
            CHECK_CUDA_ERROR(error);
        }
        else {
            for (int y = 0; y < rgb_image.height; ++y)
                memcpy((char*)rgb_image.buffer + y * rgb_image.stride,
                        (char*)pixels_buffer + y * plane_stride,
                        rgb_image.width * sizeof(rgb8));
        }


        // Allocate lab converted image buffer
        Image<lab> lab_image(width, height, is_gpu);

        // Convert RGB to LAB -> result stored inside lab_buffer
        lab_conv_process_frame(rgb_image, lab_image);
        checkKernelLaunch(is_gpu);

        if (!isInitialized)
            initializeGlobals(width, height, lab_image, is_gpu);

        // Update background and get residual image
        Image<float> residual_image(width, height, is_gpu);

        background_process_frame(lab_image, current_background, candidate_background,
                                 current_time_pixels, residual_image, bg_number_frame);
        checkKernelLaunch(is_gpu);
        //debug_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(residual_image, rgb_image, width, height);


        // Alloc and perform eroding operation
        Image<float> erode_image(width, height, is_gpu);
        erode_process_frame(
                residual_image, erode_image,
                width, height, opening_size / 2
        );
        checkKernelLaunch(is_gpu);
        //debug_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(erode_image, rgb_image, width, height);


        // Keep old residual_image alloc and perform dilatation operation
        dilate_process_frame(
                erode_image, residual_image,
                width, height, opening_size / 2
        );
        checkKernelLaunch(is_gpu);
        //debug_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(dilate_image, rgb_image, width, height);


        // Alloc and perform hysteresis operation
        Image<bool> hysteresis_image(width, height, is_gpu);
        hysteresis_process_frame(
                residual_image, hysteresis_image,
                width, height, th_low, th_high
        );
        checkKernelLaunch(is_gpu);
        //debug_bool_kernel<<<blocksPerGrid, threadsPerBlock>>>(hysteresis_image, rgb_image, width, height);


        // Alloc and red mask operation
        mask_process_frame(hysteresis_image, rgb_image, width, height);
        checkKernelLaunch(is_gpu);
        std::cout << "Running on 5" << std::endl;


        // Copy result back to pixels_buffer
        if (is_gpu) {
            error = cudaMemcpy2D(pixels_buffer, plane_stride, rgb_image.buffer, rgb_image.stride,
                                 width * sizeof(rgb8), height, cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERROR(error);
        }
        else {
            for (int y = 0; y < rgb_image.height; ++y)
                memcpy(pixels_buffer + y * plane_stride,
                       (char*)rgb_image.buffer + y * rgb_image.stride,
                       rgb_image.width * sizeof(rgb8));
        }
    }
}