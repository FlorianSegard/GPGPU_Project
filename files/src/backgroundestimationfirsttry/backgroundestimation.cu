#include <iostream>
#include "backgroundestimation.hpp"


// prob optional because in the backgroundestimation.cpp but need to test to check if it works because these are not gpu functions
// -------------------------------------------------------------------------------------------------------------------------------
__host__ __device__ lab averageLAB_GPU(lab p1, lab p2) {
    lab result;
    result.L = (p1.L + p2.L) / 2;
    result.a = (p1.a + p2.a) / 2;
    result.b = (p1.b + p2.b) / 2;
    return result;
}


// Distance euclidienne
__host__ __device__ float labDistance_GPU(lab p1, lab p2) {
    return sqrt(pow(p1.L - p2.L, 2) + pow(p1.a - p2.a, 2) + pow(p1.b - p2.b, 2));
}
// -------------------------------------------------------------------------------------------------------------------------------

__global__ void check_background_GPU(ImageView<lab> in, ImageView<lab> currentBackground, 
                                    ImageView<lab> candidateBackground, ImageView<int> currentTimePixels,
                                    ImageView<float> currentDistancePixels, int width, int height)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        lab* lineptr_lab = (lab*)((std::byte*)in.buffer + y * in.stride);
        lab* lineptr_lab_background = (lab*)((std::byte*)currentBackground.buffer + y * currentBackground.stride);
        lab* lineptr_lab_candidate = (lab*)((std::byte*)candidateBackground.buffer + y * candidateBackground.stride);
        int* lineptr_time = (int*)((std::byte*)currentTimePixels.buffer + y * currentTimePixels.stride);
        float* lineptr_distance = (float*)((std::byte*)currentDistancePixels.buffer + y * currentDistancePixels.stride);

        float distance = labDistance_GPU(lineptr_lab_background[x], lineptr_lab[x]);
        lineptr_distance[x] = distance;

        int currentpixel_time = lineptr_time[x];
        lab currentpixel = lineptr_lab[x];
        lab currentpixel_candidate = lineptr_lab_candidate[x];
        lab currentpixel_background = lineptr_lab_background[x];
        if (distance < 25)
        {
            if (currentpixel_time == 0)
            {
                lineptr_lab_candidate[x] = currentpixel;
                lineptr_time[x]++;
            }   
            else if (currentpixel_time < 100)
            {
                lineptr_lab_candidate[x] = averageLAB_GPU(currentpixel, currentpixel_candidate);
                lineptr_time[x]++;
            }
            else
            {
                lineptr_lab_background[x] = currentpixel_candidate;
                lineptr_time[x]++;
            }
        }
        else
        {
            lineptr_lab_background[x] = averageLAB_GPU(currentpixel, currentpixel_background);
            lineptr_time[x] = 0;
        }
    }

}


__global__ void initialize_buffers_GPU(lab* zero_image_buffer, int* currentTimePixels_buffer, int width, int height, std::ptrdiff_t stride_image, std::ptrdiff_t stride_time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        lab* lineptr = (lab*)((std::byte*)zero_image_buffer + y * stride_image);
        int* lineptr_time = (int*)((std::byte*)currentTimePixels_buffer + y * stride_time);

        lineptr[x] = {1, 10, 1};  
        lineptr_time[x] = 0; 
    }
}




int main()
{

    int width = 100;
    int height = 100;
    Image<lab> zero_image(width, height, true);
    Image<int> currentTimePixels(width, height, true);
    Image<float> currentDistancePixels(width, height, true);


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // just initializing for the testing (zero image and initializing time map)
    initialize_buffers_GPU<<<threadsPerBlock, blocksPerGrid>>>(zero_image.buffer, currentTimePixels.buffer, 
                                                            zero_image.width, zero_image.height, 
                                                            zero_image.stride, currentTimePixels.stride);

    cudaDeviceSynchronize();

    Image<lab> currentBackground(width, height, true);
    Image<lab> candidateBackground(width, height, true);

    cudaMemcpy2D(currentBackground.buffer, currentBackground.stride,
                 zero_image.buffer, zero_image.stride,
                 width * sizeof(lab), height,
                 cudaMemcpyDeviceToDevice);

    cudaMemcpy2D(candidateBackground.buffer, candidateBackground.stride,
                 zero_image.buffer, zero_image.stride,
                 width * sizeof(lab), height,
                 cudaMemcpyDeviceToDevice);


    while (true) {
    check_background_GPU<<<threadsPerBlock, blocksPerGrid>>>(zero_image, currentBackground, candidateBackground, currentTimePixels, currentDistancePixels, width, height);
    
    }
    cudaDeviceSynchronize();

    
}
