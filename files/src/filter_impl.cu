#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "logic/labConverter.hpp"
#include "logic/backgroundestimation.hpp"
#include "logic/filter_erode_and_dilate.hpp"
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



    // // Perform dilatation operation
    // size_t dilated_pitch;
    // lab* dilated_buffer; // type: lab array pointer
    // error = cudaMallocPitch(&dilated_buffer, &dilated_pitch,
    //                         width * sizeof(lab), height);
    // CHECK_CUDA_ERROR(error);

    // dilate<<<blocksPerGrid, threadsPerBlock>>>(
    //     eroded_buffer, dilated_buffer,
    //     width, height, eroded_pitch
    // );
    // checkKernelLaunch();

    // // Perform hysteresis operation
    // size_t hysteresis_pitch;
    // bool* hysteresis_buffer; // type: bool array pointer
    // error = cudaMallocPitch(&hysteresis_buffer, &hysteresis_pitch,
    //                         width * sizeof(bool), height);
    // CHECK_CUDA_ERROR(error);

    // hysteresis_reconstruction<<<blocksPerGrid, threadsPerBlock>>>(
    //     dilated_buffer, hysteresis_buffer,
    //     width, height, dilated_pitch
    // );
    // checkKernelLaunch();

    // // TODO: Apply the new created hysteresis mask to rgb_buffer
    // // - hysteresis_buffer, hysteresis_pitch      : the mask buffer
    // // - rgb_buffer, rgb_pitch                    : the buffer to change
    // // - heigt and widt h
    // apply_mask<<<blocksPerGrid, threadsPerBlock>>>();
    // checkKernelLaunch();

    // // Copy result back to pixels_buffer
    // error = cudaMemcpy2D(pixels_buffer, plane_stride, rgb_buffer, rgb_pitch,
    //                      width * sizeof(rgb8), height, cudaMemcpyDeviceToHost);
    // CHECK_CUDA_ERROR(error);

    // // Clean up temporary buffers
    // cudaFree(rgb_buffer);
    // cudaFree(lab_buffer);
    // cudaFree(residual_buffer);
    // cudaFree(eroded_buffer);
    // cudaFree(dilated_buffer);
}
}