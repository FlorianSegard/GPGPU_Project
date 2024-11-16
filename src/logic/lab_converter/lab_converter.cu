#include <iostream>
#include "lab_converter.hpp"
#include "lab_converter_utils.hpp"

__global__ void rgbtolab_converter_kernel(ImageView<rgb8> rgb_image, ImageView<lab> backgroundLAB, int width, int height)
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

        float r_result;
        float g_result;
        float b_result;
        get_linear(r_normalized, &r_result);
        get_linear(g_normalized, &g_result);
        get_linear(b_normalized, &b_result);

        float r_linear = r_result;
        float g_linear = g_result;
        float b_linear = b_result;

        float result[3];
        XYZ_color_space(r_linear, g_linear, b_linear, result);
        //XYZ_color_space(r_normalized, g_normalized, b_normalized, result);

        float X = result[0];
        float Y = result[1];
        float Z = result[2];

        float X_n = X / 0.95047f;
        float Y_n = Y / 1.0f;
        float Z_n = Z / 1.08883f;


        float X_n_result_f;
        float Y_n_result_f;
        float Z_n_result_f;
        f(X_n, &X_n_result_f);
        f(Y_n, &Y_n_result_f);
        f(Z_n, &Z_n_result_f);

        float X_n_sqrt = X_n_result_f;
        float Y_n_sqrt = Y_n_result_f;
        float Z_n_sqrt = Z_n_result_f;

        float L = 116.0f * Y_n_sqrt - 16.0f;
        float a = 500.0f * (X_n_sqrt - Y_n_sqrt);
        float b = 200.0f * (Y_n_sqrt - Z_n_sqrt);

        lab currentpixel_lab = {L, a, b};
        lineptr_lab[x] = currentpixel_lab;
    }
}

extern "C"
void rgbtolab_converter_cu(ImageView<rgb8> rgb_image, ImageView<lab> backgroundLAB, int width, int height)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgbtolab_converter_kernel<<<blocksPerGrid, threadsPerBlock>>>(rgb_image, backgroundLAB, width, height);

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
