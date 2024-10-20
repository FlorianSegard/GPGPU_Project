#include <iostream>
#include "backgroundestimation.hpp"


// prob optional because in the backgroundestimation.cpp but need to test to check if it works because these are not gpu functions
// -------------------------------------------------------------------------------------------------------------------------------
lab averageLAB(lab p1, lab p2) {
    lab result;
    result.L = (p1.L + p2.L) / 2;
    result.a = (p1.a + p2.a) / 2;
    result.b = (p1.b + p2.b) / 2;
    return result;
}


// Distance euclidienne
float labDistance(lab p1, lab p2) {
    return sqrt(pow(p1.L - p2.L, 2) + pow(p1.a - p2.a, 2) + pow(p1.b - p2.b, 2));
}
// -------------------------------------------------------------------------------------------------------------------------------

__global__ void check_background_GPU(lab* in_buffer, std::ptrdiff_t stride_in,
                                    lab* currentBackground_buffer, std::ptrdiff_t stride_currentBackground, 
                                    lab* candidateBackground_buffer, std::ptrdiff_t stride_candidateBackground,
                                    int* currentTimePixels_buffer, std::ptrdiff_t stride_time,
                                    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        lab* lineptr_lab = (lab*)((std::byte*)in_buffer + y * stride_in);
        lab* lineptr_lab_background = (lab*)((std::byte*)currentBackground_buffer + y * stride_currentBackground);
        lab* lineptr_lab_candidate = (lab*)((std::byte*)candidateBackground_buffer + y * stride_candidateBackground);
        int* lineptr_time = (int*)((std::byte*)currentTimePixels_buffer + y * stride_time);

        if (distance < 25)
        {
            if (currentpixel_time == 0)
            {
                lineptr_lab_candidate[x] = currentpixel;
                lineptr_time[x]++;
            }   
            else if (currentpixel_time < 100)
            {
                lineptr_lab_candidate[x] = averageLAB(currentpixel, currentpixel_candidate);
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
            lineptr_lab_background[x] = averageLAB(currentpixel, currentpixel_background);
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


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // just initializing for the testing (zero image and initializing time map)
    initialize_buffers<<<threadsPerBlock, blocksPerGrid>>>(zero_image.buffer, currentTimePixels.buffer, 
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


    check_background<<<threadsPerBlock, blocksPerGrid>>>(zero_image.buffer, zero_image.stride,
                                                            currentBackground.buffer, currentBackground.stride,
                                                            candidateBackground.buffer, candidateBackground.stride,
                                                            currentTimePixels.buffer, currentTimePixels.stride,
                                                            zero_image.width, zero_image.height);

    cudaDeviceSynchronize();

    
}