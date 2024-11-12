#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "backgroundestimationfirsttry/labConverter.hpp" //maybe do it better like do a library in the makefile or somehtign
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








size_t background_ref_pitch;
lab* background_ref = nullptr;
size_t candidate_bg_pitch;
lab* candidate_background = nullptr;

// TODO: what to do when background_ref / candidate_background null?
// TODO: is it possible to reuse buffers instead of always creating new ones?

void filter_impl(uint8_t* pixels_buffer, int width, int height, int plane_stride, int pixel_stride)
{


    Parameters params;    
    params.device = GPU;

    // GPU properties for kernel calls
    cudaError_t error;
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Alloc memory and copy input RGB buffer
    // -> cudaMemcpy2D 'kind' param - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
    
    
    // size_t rgb_pitch;
    // rgb8* rgb_buffer; // type: rgb8 array pointer
    // error = cudaMallocPitch(&rgb_buffer, &rgb_pitch,
    //                         width * sizeof(rgb8), height);

    // CHECK_CUDA_ERROR(error);

    Image<rgb8> rgb_image(width, height, true);


    error = cudaMemcpy2D(rgb_image.buffer, rgb_image.stride, pixels_buffer, plane_stride,
                         width * sizeof(rgb8), height, cudaMemcpyDefault);

    CHECK_CUDA_ERROR(error);

    // Allocate LAB converted image buffer

    // size_t lab_pitch;
    // lab* lab_buffer; // type: lab array pointer

    // error = cudaMallocPitch(&lab_buffer, &lab_pitch,
    //                         width * sizeof(lab), height);
    // CHECK_CUDA_ERROR(error);

    Image<lab> lab_image(width, height, true);


    // Convert RGB to LAB

    labConv_init(&params);

    labConv_process_frame(rgb_image, lab_image);

    checkKernelLaunch();

    // Residual image
    size_t residual_pitch;
    lab* residual_buffer; // type: lab array pointer
    error = cudaMallocPitch(&residual_buffer, &residual_pitch,
                            width * sizeof(lab), height);
    CHECK_CUDA_ERROR(error);

    // TODO: GPU residual image to code with the following args
    // - background_ref, background_ref_pitch     : the background reference
    // - lab_buffer, lab_pitch                    : the current image
    // - residual_buffer, residual_pitch          : the buffer to fill
    // - heigt and width





    // residual_image<<<blocksPerGrid, threadsPerBlock>>>();


    // checkKernelLaunch();





    // // Update background model
    // check_background_GPU<<<blocksPerGrid, threadsPerBlock>>>(
    //     lab_buffer, lab_pitch,
    //     background_ref, background_ref_pitch,
    //     candidate_background, candidate_bg_pitch,
    //     (int*)current_time_pixels, time_pixels_pitch,
    //     width, height
    // );
    // checkKernelLaunch();

    // // Perform eroding operation
    // size_t eroded_pitch;
    // lab* eroded_buffer; // type: lab array pointer
    // error = cudaMallocPitch(&eroded_buffer, &eroded_pitch,
    //                         width * sizeof(lab), height);
    // CHECK_CUDA_ERROR(error);

    // erode<<<blocksPerGrid, threadsPerBlock>>>(
    //     residual_buffer, eroded_buffer,
    //     width, height, residual_pitch
    // );
    // checkKernelLaunch();

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