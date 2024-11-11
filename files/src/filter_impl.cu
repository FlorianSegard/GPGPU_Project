#include <iostream>

struct rgb {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct lab {
    float L;
    float a;
    float b;
};

// Cuda error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

size_t background_ref_pitch;
std::byte* background_ref = nullptr;
size_t candidate_bg_pitch;
std::byte* candidate_background = nullptr;

// TODO: what to do when background_ref / candidate_background null?
// TODO: is it possible to reuse buffers instead of always creating new ones?


extern "C"
{
    void filter_impl(uint8_t* pixels_buffer, int width, int height, int plane_stride, int pixel_stride, GstClockTime timestamp)
    {
        // GPU properties for kernel calls
        cudaError_t error;
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((width + blockSize.x - 1) / blockSize.x,
                           (height + blockSize.y - 1) / blockSize.y);

        // Alloc memory and copy input RGB buffer
        // -> cudaMemcpy2D 'kind' param - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
        size_t pitch;
        std::byte* rgb_buffer; // type: rgb array pointer
        error = cudaMallocPitch(&rgb_buffer, &pitch,
                                width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(error);
        error = cudaMemcpy2D(rgb_buffer, pitch, pixels_buffer, plane_stride,
                             width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(error);

        // Allocate LAB converted image buffer
        size_t lab_pitch;
        std::byte* lab_buffer; // type: lab array pointer
        error = cudaMallocPitch(&lab_buffer, &lab_pitch,
                                width * sizeof(lab), height));
        CHECK_CUDA_ERROR(error)

        // Convert RGB to LAB
        rgbtolab_converter_GPU<<<blocksPerGrid, threadsPerBlock>>>(
            (rgb8*)rgb_buffer, rgb_pitch,
            (lab*)lab_buffer, lab_pitch,
            width, height
        )
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Residual image
        size_t residual_pitch;
        std::byte* residual_buffer; // type: lab array pointer
        error = cudaMallocPitch(&residual_buffer, &residual_pitch,
                                width * sizeof(Lab), height);
        CHECK_CUDA_ERROR(error);

        // TODO: GPU residual image to code with the following args
        // - background_ref, background_ref_pitch     : the background reference
        // - lab_buffer, lab_pitch                    : the current image
        // - residual_buffer, residual_pitch          : the buffer to fill
        // - heigt and width
        residual_image<<<blocksPerGrid, threadsPerBlock>>>();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Update background model
        check_background_GPU<<<blocksPerGrid, threadsPerBlock>>>(
            (lab*)lab_buffer, lab_pitch,
            (lab*)background_ref, background_ref_pitch,
            (lab*)candidate_background, candidate_bg_pitch,
            (int*)current_time_pixels, time_pixels_pitch,
            width, height
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Perform eroding operation
        size_t eroded_pitch;
        std::byte* eroded_buffer; // type: lab array pointer
        error = cudaMallocPitch(&eroded_buffer, &eroded_pitch,
                                width * sizeof(lab), height);
        CHECK_CUDA_ERROR(error);

        erode<<<blocksPerGrid, threadsPerBlock>>>(
            (lab*)residual_buffer, (lab*)eroded_buffer,
            width, height, residual_pitch
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Perform dilatation operation
        size_t dilated_pitch;
        std::byte* dilated_buffer; // type: lab array pointer
        error = cudaMallocPitch(&dilated_buffer, &dilated_pitch,
                                width * sizeof(lab), height);
        CHECK_CUDA_ERROR(error);

        dilate<<<blocksPerGrid, threadsPerBlock>>>(
            (lab*)eroded_buffer, (lab*)dilated_buffer,
            width, height, eroded_pitch
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Perform hysteresis operation
        size_t hysteresis_pitch;
        std::byte* hysteresis_buffer; // type: bool array pointer
        error = cudaMallocPitch(&hysteresis_buffer, &hysteresis_pitch,
                                width * sizeof(bool), height);
        CHECK_CUDA_ERROR(error);

        hysteresis_reconstruction<<<blocksPerGrid, threadsPerBlock>>>(
            (lab*)dilated_buffer, (lab*)hysteresis_buffer,
            width, height, dilated_pitch
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // TODO: Apply the new created hysteresis mask to rgb_buffer
        // - hysteresis_buffer, hysteresis_pitch      : the mask buffer
        // - rgb_buffer, rgb_pitch                    : the buffer to change
        // - heigt and widt h
        apply_mask<<<blocksPerGrid, threadsPerBlock>>>();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy result back to pixels_buffer
        error = cudaMemcpy2D(pixels_buffer, plane_stride, rgb_buffer, rgb_pitch,
                             width * sizeof(rgb8), height, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(error)

        // Clean up temporary buffers
        cudaFree(rgb_buffer);
        cudaFree(lab_buffer);
        cudaFree(residual_buffer);
        cudaFree(eroded_buffer);
        cudaFree(dilated_buffer);
        cudaFree(marker_buffer);
        cudaFree(output_buffer);
    }
}