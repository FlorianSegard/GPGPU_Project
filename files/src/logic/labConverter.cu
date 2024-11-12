#include <iostream>
#include "labConverter.hpp"
#include "labConverterUtils.hpp"

__global__ void rgbtolab_converter_GPU(ImageView<rgb8> rgb_image, ImageView<lab> backgroundLAB, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        rgb8* lineptr = (rgb8*)((std::byte*)rgb_image.buffer + y * rgb_image.stride);
        lab* lineptr_lab = (lab*)((std::byte*)backgroundLAB.buffer + y * backgroundLAB.stride);

        rgb8 currentpixel = lineptr[x];


        float r_normalized = currentpixel.r / 255.f;
        float g_normalized = currentpixel.g / 255.f;
        float b_normalized = currentpixel.b / 255.f;

        float r_linear = get_linear_GPU(r_normalized);
        float g_linear = get_linear_GPU(g_normalized);
        float b_linear = get_linear_GPU(b_normalized);
        
        float result[3];
        XYZ_color_space_GPU(r_linear, g_linear, b_linear, result);

        float X = result[0];
        float Y = result[1];
        float Z = result[2];

        float X_n = X / 0.95047;
        float Y_n = Y / 1.0;
        float Z_n = Z / 1.08883;

        float X_n_sqrt = f_GPU(X_n);
        float Y_n_sqrt = f_GPU(Y_n);
        float Z_n_sqrt = f_GPU(Z_n);

        float L = 116 * Y_n_sqrt - 16;
        float a = 500 * (X_n_sqrt - Y_n_sqrt);
        float b = 200 * (Y_n_sqrt - Z_n_sqrt);

        lab currentpixel_lab = {L, a, b};
        lineptr_lab[x] = currentpixel_lab;
    }
}

extern "C"
void rgbtolab_converter_cu(ImageView<rgb8> rgb_image, ImageView<lab> backgroundLAB, int width, int height)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgbtolab_converter_GPU<<<blocksPerGrid, threadsPerBlock>>>(rgb_image, backgroundLAB, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}


// __global__ void initialize_buffers_GPU(rgb8* zero_image_buffer, int width, int height, std::ptrdiff_t stride_image) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height) {
//         rgb8* lineptr = (rgb8*)((std::byte*)zero_image_buffer + y * stride_image);

//         lineptr[x] = {1, 10, 1};
//     }
// }



// int main()
// {

//     int width = 100;
//     int height = 100;
//     Image<rgb8> zero_image(width, height, true);
//     Image<lab> converted_image(width, height, true);


//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
//                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     // just initializing for the testing (zero image and initializing time map)
//     initialize_buffers_GPU<<<threadsPerBlock, blocksPerGrid>>>(zero_image.buffer, zero_image.width, zero_image.height, zero_image.stride);
//     cudaDeviceSynchronize();
    
//     rgbtolab_converter_GPU<<<threadsPerBlock, blocksPerGrid>>>(zero_image, lab_image, width, height);
//     cudaDeviceSynchronize();

// }
